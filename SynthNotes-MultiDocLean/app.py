"""
SynthNotes MultiDocLeanLean — MultiDoc + pre-extraction stage to strip annual-report
boilerplate before synthesis. Two upload buckets (ARs + transcripts), one button.

Workflow:
  1. User uploads annual reports (.txt) in one bucket and quarterly transcripts in another
  2. User reviews/edits THREE prompts: AR extraction, transcript extraction, synthesis
  3. User picks target output length
  4. Stage 0 — Extraction (parallel, cheap model):
       - AR chunks → keep narrative sections (Chairman, MD&A, business review,
         strategy/outlook), skip financial statements, notes to accounts, RPT,
         compliance, schedules, accounting policies
       - Transcript chunks → keep mgmt opening remarks + Q&A substance, skip operator
         greetings, safe-harbour disclaimers, pleasantries
  5. Stage 1 — Map (parallel, cheap model): per-chunk notes on extracted content
  6. Stage 2 — Reduce (plan-then-write, quality model): outline + parallel sections
       → final document
  7. Outputs: final document (.md, .pdf), interim Map notes (.txt for resume),
       combined extracted text (.txt, portable across other tools)

Borrows the chunking, retry, streaming, plan-then-write, and cost-tracking logic
from SynthNotes-MultiDoc; this is a separate self-contained app.
"""

import streamlit as st
import google.generativeai as genai
import os, re, time, json, html as html_module
from datetime import datetime
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import streamlit.components.v1 as components


# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY", "")
if _api_key:
    genai.configure(api_key=_api_key)

MAX_OUTPUT_TOKENS  = 65536
MAX_FILES          = 500          # File count cap (real ceiling is reduce-stage token budget; see below)
MAX_FILE_SIZE_MB   = 10           # Per-file size cap
PARALLEL_WORKERS   = 3            # Concurrent chunk-processing workers (safe for rate limits)

# ── Hierarchical reduce parameters ─────────────────────────────────────────────
# If the combined Map-stage output exceeds WORDS_PER_REDUCE_BATCH, the Reduce stage
# splits the work into batches, summarises each batch (intermediate reduce), then
# recurses. This lets the app handle arbitrary corpora — including the 500+ file
# case the file-count limit would otherwise block.
#
# Budget rationale: Gemini's 1M-token context window must hold prompt + input + output.
# Reserve ~65K tokens for output, ~5K for prompt scaffolding → ~930K available.
# At ~1.3 tokens/word, that's ~715K words. We pick a conservative 500K-word budget
# per batch to leave headroom for prompt overhead, model quirks, and safety margin.
WORDS_PER_REDUCE_BATCH = 500_000
MAX_REDUCE_DEPTH       = 3        # Recursion cap (depth 3 handles inputs up to ~125M words)

MODELS = {
    "Gemini 2.5 Flash (Fast)":       "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite (Cheap)": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro (Best)":         "gemini-2.5-pro",
    "Gemini 3.0 Flash":              "gemini-3-flash-preview",
    "Gemini 3.5 Flash":              "gemini-3.5-flash",
    "Gemini 2.0 Flash":              "gemini-2.0-flash-lite",
    "Gemini 1.5 Flash":              "gemini-1.5-flash",
}

# Approximate pricing per 1M tokens (USD), <200K context. Verified May 2026.
MODEL_PRICING = {
    "gemini-2.5-pro":         (1.25, 10.00),
    "gemini-2.5-flash":       (0.30,  2.50),
    "gemini-2.5-flash-lite":  (0.10,  0.40),
    "gemini-3.5-flash":       (0.50,  3.00),
    "gemini-3-flash-preview": (0.50,  3.00),
    "gemini-3.1-flash-lite":  (0.25,  1.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.0-flash-lite":  (0.075, 0.30),
    "gemini-1.5-flash":       (0.075, 0.30),
}

# Length presets the user can pick from for the final synthesised document.
LENGTH_PRESETS = {
    "Short (~2000 words)":    2000,
    "Standard (~4000 words)": 4000,
    "Long (~6000 words)":     6000,
    "Maximum (~8000 words)":  8000,
    "Custom":                 None,
}

# Adaptive chunk sizing: bigger target → smaller chunks (more granular extraction).
# Each row: (target_max_words, chunk_size, chunk_overlap).
# The first row whose target_max >= user's target is selected.
CHUNK_SIZE_TABLE: List[Tuple[int, int, int]] = [
    (1500,  8000, 600),
    (3000,  6000, 500),
    (5000,  5000, 450),
    (8000,  4000, 400),
    (15000, 3000, 300),
]


def compute_chunk_params(target_word_count: int) -> Tuple[int, int]:
    """Return (chunk_size, overlap) appropriate for the given output target."""
    for max_target, chunk_size, overlap in CHUNK_SIZE_TABLE:
        if target_word_count <= max_target:
            return chunk_size, overlap
    return 3000, 300


# ── Extraction-stage chunk sizes ───────────────────────────────────────────────
# We deliberately use VERY large chunks for extraction. Reasoning:
#   - Gemini's 1M-token context window easily fits an entire AR (~200K-300K words)
#   - Fewer chunks = fewer LLM calls = dramatically lower wall-clock latency
#   - Output token cap (65K) is what forces SOME chunking on the largest ARs;
#     at ~30% retention, an 80K-word chunk's extract ≈ 24K words = ~34K tokens,
#     fits comfortably in the output budget with headroom
#   - create_chunks_with_overlap returns a SINGLE chunk when the file is smaller
#     than chunk_size — so small files (typical transcripts, small ARs) become
#     one LLM call automatically. That's the "adaptive" behaviour for free.
#
# Result: a 30K-word transcript → 1 chunk; a 60K-word AR → 1 chunk;
#         a 150K-word AR → 2 chunks; a 250K-word AR → 4 chunks.
EXTRACTION_CHUNK_SIZE_AR        = 80_000
EXTRACTION_CHUNK_OVERLAP_AR     = 2_000
EXTRACTION_CHUNK_SIZE_TRANSCRIPT    = 30_000
EXTRACTION_CHUNK_OVERLAP_TRANSCRIPT = 1_000

# Extraction stage uses its own worker pool size. Flash-Lite has high rate limits
# (4000 RPM, 4M TPM on paid tier) and extraction calls don't depend on each other,
# so we parallelise more aggressively than the synthesis Map stage.
EXTRACTION_PARALLEL_WORKERS = 6

# Marker the extraction prompt is told to emit when a chunk has nothing worth
# keeping. We use this to filter empty chunks from the combined extract.
EXTRACTION_SKIP_MARKER = "[chunk skipped — no analyst-relevant content]"


# Default user prompt shipped in the UI text-area. The user can edit or replace it.
# Designed for equity-research notes on ONE company across transcripts + presentations
# + annual reports — but the tool itself is general; swap this out for other use cases.
DEFAULT_USER_PROMPT = """ROLE
You are an experienced equity analyst explaining ONE company to a sharp colleague who
knows markets but not this company. Write the way you'd actually talk it through: in plain
English, as a STORY, landing the things that matter and leaving out the noise. Your reader
is an investment analyst, not an engineer — explain the business in business terms, never
in procurement or product-brochure language.

SOURCES
All documents in this project: ~5 years of earnings-call transcripts, the last few
investor presentations, and recent annual reports. Read them together. The transcripts —
especially the analyst Q&A — are where management explains the WHY, so they carry the
story. Annual reports are only for hard facts (segment splits, capacity, balance sheet);
never reproduce their narrative or outlook prose, which is generic boilerplate.

TWO ABSOLUTE RULES
1. FACT before INTERPRETATION, always visually separate. State what management said or what
   the numbers show as plain narrative. Then, where you add your own judgement, start a new
   paragraph beginning "My read:" — the ONLY place your inference may appear. A reader must
   be able to skip every "My read:" and still have a complete, accurate account of what the
   company and management actually said.
2. No outside data, no invented quotes. Not in the documents = say so. Quotes ≤25 words,
   exact; otherwise paraphrase.

READABILITY RULES — how to write so it's retained
- Flowing paragraphs. No big tables, no bullet dumps, no reference codes, no
  quarter-by-quarter logs.
- FORMATTING FOR NAVIGATION: under each numbered section, break the prose with short bold
  sub-headings (a few words each) that signal what the next paragraph(s) cover, so the
  reader can scan the structure and find their place. Sub-headings label the content; they
  don't replace the prose — write full paragraphs under each, not bullets. Use 2-5
  sub-headings per section as the material needs; don't force them where a section is short.
- Depth comes from EXPLAINING A FEW THINGS FULLY, not from listing many. For every claim
  that matters, give the mechanism — the WHY behind it — in plain words. A reader should
  finish each section understanding not just what happened but why.
- When a point recurs across many quarters, say so plainly ("management has said for six
  straight quarters that…") — repetition is the signal of what's load-bearing, and it's
  what the reader will remember.
- Round numbers, ranges, direction. You're telling a story, not reconciling a model.
- Cut anything that only makes sense with heavy context you haven't given. If it matters,
  give the context in half a sentence; if it doesn't, drop it.
- The test for every paragraph: could the reader repeat this back to someone tomorrow? If
  it's a pile of disconnected details, rewrite it as a point with a reason.

WRITE THE NOTE IN THIS ORDER, AS FLOWING PROSE UNDER EACH HEADING (with bold sub-headings
inside each section per the formatting rule above):

1. WHAT THIS COMPANY DOES
   In plain English: what does it sell, and to whom? Break revenue into its main buckets in
   words (roughly what % each is), and for each, say who the customer is (state utilities /
   private industrial capex / EPC contractors / OEMs / exports) and what that product does
   for them. Where product type drives the economics — e.g. higher-voltage transformers are
   harder to make, so fewer players and better margins — explain it that way, in cause-and-
   effect business terms, not spec-sheet terms. Domestic vs export, plainly.
   My read: where the genuinely attractive part of the mix sits, and why.

2. THE STORY OF THE LAST 5 YEARS  (the heart of the note — give this the most space)
   Tell it as a narrative, not a ledger. Cover, and connect, these threads:
   - GROWTH: how sales and EBITDA grew, and crucially WHERE the growth came from — which
     products/end-markets pulled it and WHY that demand appeared (a capex cycle, a policy
     push, exports, a competitor stumbling, share gains). Don't just name the driver —
     explain the mechanism behind it.
   - MARGINS: what happened to margins and WHY — pricing power because the market was tight?
     richer mix? or squeezed by a specific raw material? Say which, and whether management
     framed the gains as durable or temporary.
   - ORDER BOOK: how it grew and what it's signalling, since it leads sales — and whether
     management said anything about the QUALITY (margin) of the book, not just its size.
   - CAPACITY: how capacity expansion played out — did they add it, did it arrive in time,
     did demand absorb it, or did they expand into a hot market late?
   Weave these into one story, anchored on the few drivers that genuinely mattered, so the
   reader can repeat the arc back. (Natural sub-headings here: the growth drivers, margins,
   order book, capacity — use them as your bold sub-heads.)
   My read: how much of this growth is structural versus a cyclical/peak moment, and which
   parts I'd trust to persist.

3. WHY THIS COMPANY WINS  (competitive advantage)
   What actually lets them win business and hold margin — technology, approvals and track
   record, customer relationships, scale, being one of few who can do the hard work?
   Explain it as why a customer picks them over the next supplier.
   My read: whether that edge is durable or just today's tightness flattering everyone.

4. COMPETITION, AND HOW MANAGEMENT TALKS ABOUT IT
   Who they compete with, and what management says when analysts push on new competition,
   new capacity coming in, imports, or pricing pressure — do they sound relaxed or guarded,
   and have they admitted any share loss or pricing slippage? Whether they engage the
   question or deflect it is itself informative.
   My read: whether the competitive threat is real and how honestly management is facing it.

5. WHAT MANAGEMENT EXPECTS NEXT  (their words, kept clearly as their words)
   Pull together what management has guided or signalled on demand and sales growth, order-
   book outlook, and margins — and the REASONS they give for each, since the reasoning
   matters more than the number. Include their stated plans on capacity and capital
   allocation. Keep this strictly "management says," never blended with your own view.
   My read: which expectations look well-supported versus optimistic, and what they're
   quietly assuming.

6. WHAT TO REMEMBER
   Close with a short paragraph: the handful of things a busy investor should actually
   carry away — the load-bearing points that recurred and matter. Plain sentences, no grab-
   bag. This is the part the reader keeps.

LENGTH & TONE
Long enough to do justice to section 2, short enough to read in one sitting and summarise
from memory. Depth on the things that matter, silence on the things that don't. If a
section is thin in the sources, keep it short and say what's missing — never pad with
generic industry talk.
"""


# ── Extraction prompts (defaults shipped — editable in the UI) ─────────────────
# These prompts filter source documents BEFORE synthesis. Each chunk is classified
# as "keep" (analyst-relevant narrative) or "skip" (financial schedules, statutory
# compliance, boilerplate) and only kept content flows downstream.

DEFAULT_EXTRACTION_AR_PROMPT = """You are extracting ONLY the analyst-relevant narrative portions from a chunk of an Indian annual report.

KEEP these sections — output them VERBATIM (do not paraphrase, summarise, or alter):
- Chairman's Letter / Managing Director's Letter / CEO's Letter
- Management Discussion and Analysis (MD&A) — all sub-sections
- Business Performance / Business Review / Operations Review
- Segment / Geography / Business Vertical Performance (narrative form)
- Strategy / Strategic Direction / Future Outlook / Five-Year View
- Director's Report — narrative business commentary only (skip the routine governance text)
- Business Responsibility Report — only if it contains substantive business content

SKIP these sections entirely (output nothing for these):
- Notice of AGM
- Standalone / Consolidated Financial Statements (balance sheet, P&L, cash flow)
- Notes to Accounts
- Significant Accounting Policies
- Schedules, annexures, related documents
- Related Party Transactions
- Statutory disclosures, secretarial audit, compliance reports
- Auditor's Report and Independent Auditor's Report
- Corporate Governance Report — skip unless substantive business content
- Shareholder information / investor information / dividend history
- Subsidiary company details
- Cover pages, contents pages, awards/recognition pages, photograph captions

INSTRUCTIONS
1. Read the chunk below.
2. Identify which portions are KEEP and which are SKIP per the rules above.
3. Output the KEEP portions VERBATIM. Do NOT paraphrase or summarise — preserve exact wording.
4. If a chunk crosses a boundary (e.g., MD&A ends and Notes to Accounts begins), output only the relevant part and end with `[remainder skipped — non-narrative]`.
5. If the WHOLE chunk is SKIP material, output exactly: `[chunk skipped — no analyst-relevant content]`
6. When in doubt, KEEP. Loss of a narrative passage is worse than retaining a borderline one.
7. Do NOT add commentary about what you found. Do NOT add section labels you've invented.
"""


DEFAULT_EXTRACTION_TRANSCRIPT_PROMPT = """You are extracting the analyst-relevant substance from a chunk of a quarterly earnings call transcript.

KEEP these portions — output them VERBATIM:
- Management opening remarks about business performance, segment performance, strategy, outlook
- Q&A: analyst questions and management responses
- Forward-looking statements, guidance updates, capex commentary
- Specific data points, segment commentary, customer/product/geography mentions

SKIP these portions (output nothing):
- Operator greetings, intro logistics, registration/dial-in housekeeping
- Safe-harbour disclaimers and forward-looking-statement legal disclaimers
- Repetitive analyst pleasantries — "congrats on the quarter", "thanks for taking my question", "this is XYZ from ABC Securities and Investments"
- Closing remarks that are just thanks/wrap-up
- Logistical interruptions — "the next question comes from...", "may I request a follow-up question please"

INSTRUCTIONS
1. Output the KEEP portions VERBATIM. Do NOT paraphrase or summarise.
2. If the WHOLE chunk is SKIP material, output exactly: `[chunk skipped — no analyst-relevant content]`
3. When in doubt, KEEP. Analyst Q&A is the most valuable part of a transcript — bias toward retention.
4. Do NOT add commentary or section labels.
"""


EXTRACTION_WRAPPER_TEMPLATE = """{extraction_prompt}

---

CONTEXT
- Source document: **{filename}**
- This is chunk {chunk_n} of {total_chunks} from this document.

CHUNK CONTENT
{chunk_text}
"""


# ── 2. PROMPT TEMPLATES ────────────────────────────────────────────────────────

# The user provides their own prompt at runtime. These templates WRAP the user's
# prompt with structural requirements (source attribution, chronological-marker
# preservation, length target, chronology inference) without dictating content.

MAP_PROMPT_TEMPLATE = """You are extracting structured notes from a section of a source document, following the user's specifications.

### USER'S INSTRUCTIONS (defines WHAT notes to make and HOW)
{user_prompt}

### CONTEXT
- Source document: **{filename}**
- This is section {chunk_n} of {total_chunks} from this document
- This document is item {file_position} of {total_files} in the collection

### YOUR TASK
1. Process the SOURCE CONTENT below following the USER'S INSTRUCTIONS above.
2. Capture ALL substantive content in this section — examples, data, named entities, reasoning, claims, dates.
3. **Preserve every chronological marker** — dates, fiscal years, quarters (Q1/Q2/Q3/Q4), time periods, sequence references ("the previous", "after the", "as of"), version numbers, milestones. These are critical for ordering across documents in the synthesis stage.
4. Begin your output with a single heading line for source attribution:
   `**From {filename} — section {chunk_n}/{total_chunks}**`
5. Do NOT skip or condense substantive content.
6. Do NOT include meta-commentary like "this section discusses…" or "the document mentions…".
7. Apply the user's instructions to format the body (bullets, prose, sub-headings — whatever they asked for).

### SOURCE CONTENT
{chunk_text}
"""


REDUCE_PROMPT_TEMPLATE = """You are synthesising a final consolidated document from notes extracted across multiple source documents.

### USER'S ORIGINAL INSTRUCTIONS (this is what the final document should be)
{user_prompt}

### LENGTH TARGET
Approximately **{target_word_count} words** for the final document. Stay within ±15% of this target. Prioritise depth on the most important material over shallow breadth across everything. If the source notes contain more detail than fits, choose what to keep based on the user's instructions above.

### CHRONOLOGY DIRECTIVE
The notes below come from **{num_files} source document(s)**, processed section-by-section. The filenames are listed, but the **chronological order across documents is NOT given**. You must infer it from the content:

- Look for dates, fiscal years, quarters (Q1/Q2/Q3/Q4), specific time periods
- Look for sequence cues: "the previous quarter", "after the merger", "before the launch", "as of March"
- Look for evolution cues: changes in numbers, references to past events, tone shifts
- Filenames may also carry hints (date strings, version numbers) — use them, but content trumps filename

If chronology can be established, **organise the final document chronologically**.

If chronology genuinely cannot be inferred from content, group by topic instead and prepend this single line at the very top of the document:
> _Note: Chronological order could not be reliably inferred from the source content; this document is organised by topic._

### SYNTHESIS RULES
1. Read ALL the per-section notes below in full before writing.
2. Establish chronological order across documents from content cues (per the directive above).
3. Synthesise into ONE coherent document following the user's instructions.
4. Preserve key facts, data points, and reasoning from the source notes.
5. Eliminate redundancy where the same point appears across multiple sections.
6. Maintain narrative flow appropriate to the user's instructions.
7. Target approximately **{target_word_count} words**.
8. **Do NOT add information not present in the source notes** — no inference, no external knowledge, no filling gaps.
9. Use clear headings to make the document navigable.

### SOURCE DOCUMENTS (filename list)
{filename_list}

### PER-SECTION NOTES (sections are in arbitrary order — establish chronology yourself)
{combined_notes}

---

Now produce the final consolidated document. Begin immediately with the document itself — no preamble like "Here is the synthesis…".
"""


OUTLINE_PROMPT = """You are designing the STRUCTURE of a consolidated document that synthesises notes from multiple source documents. This is the **planning step** — you will NOT write the document yet.

### USER'S INSTRUCTIONS (defines what the document is)
{user_prompt}

### TARGET LENGTH
**~{target_word_count} words total**. Section word budgets MUST sum to approximately this number.

### CHRONOLOGY DIRECTIVE
The notes below come from **{num_files} source document(s)**. Examine the content for date markers, fiscal years, quarters (Q1/Q2/Q3/Q4), sequence references, and contextual ordering. If chronology is inferable, order sections chronologically. If not, group by topic.

### SOURCE DOCUMENTS
{filename_list}

### PER-SECTION NOTES
{combined_notes}

---

### YOUR TASK
Produce a structured outline of the final document. Use EXACTLY this format:

# [Document Title — chosen by you to fit the user's instructions]

## [Section 1 heading]
- Coverage: [1–2 sentence description of what this section covers, including which source notes it draws from]
- Word budget: ~[N] words

## [Section 2 heading]
- Coverage: [...]
- Word budget: ~[N] words

(continue for all sections)

TOTAL: ~[sum of all section budgets — must be close to {target_word_count}] words
CHRONOLOGY_NOTE: [one sentence about how sections are ordered, e.g. "Sections are in chronological order from Q1 2023 to Q4 2024" OR "Chronology could not be inferred; sections are organised by topic."]

### RULES
1. Section count: typically **5–10 sections**, scaled to content and length target. With ~{target_word_count} words and dense source material, lean toward more sections (8–10) rather than fewer.
2. Word budgets MUST sum to approximately **{target_word_count}** (±10%).
3. Each section should be coherent, self-contained, and cover distinct material (no overlap between sections).
4. Section headings should reflect the user's instructions in form and tone.
5. Do NOT write any prose body — only the outline structure.
6. The COVERAGE line for each section should be specific enough that a separate writer could write JUST that section knowing only its coverage description and the source notes.

Produce the outline now. Begin immediately with `# [title]`.
"""


SECTION_PROMPT = """You are writing ONE SECTION of a larger consolidated document. Other sections are being written separately — your job is to write your section well.

### USER'S ORIGINAL INSTRUCTIONS (context — what the final document is)
{user_prompt}

### THIS SECTION'S ASSIGNMENT
**Heading**: {section_heading}
**Coverage**: {section_coverage}
**Word budget**: approximately **{section_word_budget} words**
**Position**: section {section_n} of {total_sections}

### FULL DOCUMENT OUTLINE (for context — DO NOT cover material assigned to other sections)
{outline_text}

### LENGTH COMPLIANCE — IMPORTANT
Aim for **~{section_word_budget} words**. This is a firm target, not a suggestion:
- If your natural draft is much shorter, you are under-using the source notes — go back to the notes below and pull more substantive detail until you hit the budget.
- If your draft is significantly longer, you may be covering material that belongs in OTHER sections — trim to your assigned coverage scope.

### CONTENT RULES
1. **Stay strictly within your section's coverage scope.** The outline above lists other sections — their material is theirs, not yours.
2. Use ONLY information from the per-section notes below — no external knowledge, no inference.
3. Preserve hard data (numbers, percentages, dates, named entities, monetary values, named geographies) from the source notes.
4. Apply the user's instructions for formatting (prose, bullets, sub-headings) within this section.
5. Begin your output with the heading line exactly: `## {section_heading}`
6. Do NOT include preamble like "This section covers…" — start with content immediately after the heading.
7. Do NOT include a conclusion that summarises other sections — the final document has its own flow.

### PER-SECTION NOTES (full set — extract content relevant to YOUR section)
{combined_notes}

---

Write your assigned section now. Start with `## {section_heading}` and produce only the section body.
"""


INTERMEDIATE_REDUCE_PROMPT = """You are compressing a BATCH of per-section notes into a denser intermediate summary that will be combined with other batches in a later step.

**This is NOT the final document.** Your output will be one of several intermediate summaries that get synthesised together later. Preserve information richly; the final compression happens downstream.

### USER'S ORIGINAL INSTRUCTIONS (context — what the eventual final document is supposed to be)
{user_prompt}

### YOUR TASK
Compress the per-section notes below to approximately **{target_word_count} words** while preserving:

- **ALL hard data** — numbers, percentages, monetary values, named entities, dates
- **ALL chronological markers** — dates, quarters, fiscal years, sequence references ("the previous", "as of X")
- **ALL source attribution** — keep the `**From [filename] — section X/Y**` headings exactly as they appear; group output by source document
- **The substance of all distinct claims, arguments, and reasoning** raised in this batch

Eliminate ONLY:
- Redundancy where the same point appears multiple times within this batch
- Filler language, meta-commentary, conversational artifacts

Do NOT:
- Add inference or external knowledge — only what's in the notes below
- Synthesise across the batch into a narrative (that's the final step's job)
- Drop substantive content to hit the word target — better to overshoot slightly than to lose information
- Re-order chronologically (the final step does that)

### PER-SECTION NOTES IN THIS BATCH
{combined_notes}

---

Produce the intermediate compressed summary now. Preserve source-attribution headings exactly.
"""


# ── 3. UTILITIES (borrowed from SynthNotes-Pro; self-contained here) ───────────

def get_model(display_name: str) -> genai.GenerativeModel:
    """Cache and return a GenerativeModel for the given UI label."""
    cache = st.session_state.setdefault("_model_cache", {})
    model_id = MODELS.get(display_name, "gemini-2.5-flash")
    if model_id not in cache:
        cache[model_id] = genai.GenerativeModel(model_id)
    return cache[model_id]


def _record_usage(model_id: str, response, stage: str = "") -> None:
    """Append a usage entry to st.session_state['usage_log']. Silent on any failure."""
    try:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        input_tokens  = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        if input_tokens == 0 and output_tokens == 0:
            return
        log = st.session_state.setdefault("usage_log", [])
        log.append({
            "model":         model_id or "unknown",
            "stage":         stage or "other",
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        })
    except Exception:
        pass


def compute_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    pricing = MODEL_PRICING.get(model_id)
    if not pricing:
        return 0.0
    in_price, out_price = pricing
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


# ── Pre-flight cost estimation ─────────────────────────────────────────────────
# Rough estimate shown to the user BEFORE the run, so they can sanity-check
# expected spend at current input size and model selection. Real cost typically
# lands within ±30% of this — the biggest uncertainty is per-chunk Map output
# verbosity, which depends heavily on the user prompt's verbosity requirements.

_TOKENS_PER_WORD_EST = 1.4  # mixed prose with light tagging tokenises slightly above 1


def estimate_pipeline_cost(
    ar_words: int, transcript_words: int, target_word_count: int,
    extract_model_id: str, map_model_id: str, reduce_model_id: str,
    retention_pct: float = 0.30,
):
    """Estimate per-stage cost in USD for a full MultiDocLean pipeline run.

    Architecture being estimated:
      - Extraction (Stage 0): chunk both buckets, classify keep/skip per chunk
        with cheap model. Output ~30% of input on average (configurable).
      - Map (Stage 1): chunk EXTRACTED content, run per-chunk notes (cheap model).
      - Outline (Stage 2a): 1 call receiving full Map output (quality model).
      - Section writing (Stage 2b): ~8 parallel section calls, each receiving the
        full Map output (no section-routing in MultiDocLean yet).
    """
    # Extraction stage — separate chunk sizes per bucket
    p_overhead_extr = 800  # extraction prompt + wrapper

    ar_step = EXTRACTION_CHUNK_SIZE_AR - EXTRACTION_CHUNK_OVERLAP_AR
    ar_chunks = max(0, (ar_words + ar_step - 1) // ar_step) if ar_words else 0
    ar_extr_in_tokens  = int(ar_chunks * (EXTRACTION_CHUNK_SIZE_AR + p_overhead_extr) * _TOKENS_PER_WORD_EST)
    ar_extr_out_tokens = int(ar_words * retention_pct * _TOKENS_PER_WORD_EST)

    tx_step = EXTRACTION_CHUNK_SIZE_TRANSCRIPT - EXTRACTION_CHUNK_OVERLAP_TRANSCRIPT
    tx_chunks = max(0, (transcript_words + tx_step - 1) // tx_step) if transcript_words else 0
    tx_extr_in_tokens  = int(tx_chunks * (EXTRACTION_CHUNK_SIZE_TRANSCRIPT + p_overhead_extr) * _TOKENS_PER_WORD_EST)
    # Transcripts retain more (less boilerplate) — assume 70% retained for the estimate
    tx_extr_out_tokens = int(transcript_words * 0.70 * _TOKENS_PER_WORD_EST)

    extract_in_tokens  = ar_extr_in_tokens + tx_extr_in_tokens
    extract_out_tokens = ar_extr_out_tokens + tx_extr_out_tokens
    extract_cost = compute_cost(extract_in_tokens, extract_out_tokens, extract_model_id)

    # Estimated input to synthesis after extraction
    extracted_words = int(ar_words * retention_pct + transcript_words * 0.70)

    # Map stage — runs on the EXTRACTED content (much smaller than raw)
    chunk_size, overlap = compute_chunk_params(target_word_count)
    map_step = chunk_size - overlap
    n_map_chunks = max(1, (extracted_words + map_step - 1) // map_step) if extracted_words else 0

    prompt_overhead_words = 1100      # user prompt + wrapper template
    map_in_tokens  = int(n_map_chunks * (chunk_size + prompt_overhead_words) * _TOKENS_PER_WORD_EST)
    map_out_tokens = int(n_map_chunks * 2000 * _TOKENS_PER_WORD_EST)
    map_cost = compute_cost(map_in_tokens, map_out_tokens, map_model_id)

    # Outline pass — single call, all Map output as input
    outline_in_tokens  = map_out_tokens + int(500 * _TOKENS_PER_WORD_EST)
    outline_out_tokens = int(500 * _TOKENS_PER_WORD_EST)
    outline_cost = compute_cost(outline_in_tokens, outline_out_tokens, reduce_model_id)

    # Section writing — typical ~8 parallel sections; each receives the FULL Map
    # output (no section-routing in MultiDocLean yet, unlike FactbaseNote).
    n_sections = 8
    section_in_per_call = map_out_tokens + int((500 + 800) * _TOKENS_PER_WORD_EST)
    section_total_in    = n_sections * section_in_per_call
    section_total_out   = int(target_word_count * _TOKENS_PER_WORD_EST)
    section_cost = compute_cost(section_total_in, section_total_out, reduce_model_id)

    return {
        "ar_chunks":         ar_chunks,
        "transcript_chunks": tx_chunks,
        "extracted_words":   extracted_words,
        "n_map_chunks":      n_map_chunks,
        "chunk_size":        chunk_size,
        "extract_cost":      extract_cost,
        "map_cost":          map_cost,
        "outline_cost":      outline_cost,
        "section_cost":      section_cost,
        "total_cost":        extract_cost + map_cost + outline_cost + section_cost,
    }


def generate_with_retry(model, prompt, max_retries: int = 3, stream: bool = False,
                        generation_config=None, stage: str = ""):
    """Call the model with retry on transient errors and auto-record usage."""
    kwargs = {"stream": stream}
    if generation_config:
        kwargs["generation_config"] = generation_config
    model_id = getattr(model, "model_name", "") or ""
    if model_id.startswith("models/"):
        model_id = model_id[len("models/"):]
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, **kwargs)
            if not stream:
                _record_usage(model_id, response, stage)
            else:
                try:
                    response._tracked_model_id = model_id
                    response._tracked_stage    = stage
                except (AttributeError, TypeError):
                    pass
            return response
        except Exception as e:
            err = str(e).lower()
            is_transient = any(k in err for k in ["429", "503", "500", "deadline", "timeout", "unavailable", "resource_exhausted"])
            if is_transient and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def stream_and_collect(response, placeholder=None) -> Tuple[str, int]:
    """Iterate a streamed response, collect text, auto-record usage if tagged."""
    full_text, counter = "", 0
    for chunk in response:
        if chunk.parts:
            full_text += chunk.text
            counter += 1
            if placeholder and counter % 5 == 0:
                placeholder.caption(f"Streaming… {len(full_text.split()):,} words")
    if placeholder:
        placeholder.empty()
    tracked_model = getattr(response, "_tracked_model_id", "")
    tracked_stage = getattr(response, "_tracked_stage", "")
    if tracked_model:
        _record_usage(tracked_model, response, tracked_stage)
    tokens = 0
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = getattr(response.usage_metadata, "total_token_count", 0)
    except Exception:
        pass
    return full_text, tokens


def create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks


# ── Filename + auto-download helpers ──────────────────────────────────────────
# Auto-download mitigates Streamlit session loss: as soon as a run completes,
# the output is written to the user's disk so even if the session times out or
# the tab is refreshed, the file is already saved locally.

def _sanitize_filename_component(s: str, fallback: str = "untitled") -> str:
    """Make a string safe for use in a filename. Replaces runs of non-alphanumerics
    with a single underscore; falls back to 'untitled' for empty/blank input."""
    if s is None:
        return fallback
    s = s.strip()
    if not s:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return safe or fallback


def filename_for(company_name: str, kind: str, ext: str) -> str:
    """Build a filename like '20260628_Hitachi_Energy_extract.txt'.
    Date is today's date in YYYYMMDD; kind is the document type label."""
    date_str = datetime.now().strftime("%Y%m%d")
    company  = _sanitize_filename_component(company_name)
    kind_safe = _sanitize_filename_component(kind, fallback="output")
    ext_safe  = ext.lstrip(".")
    return f"{date_str}_{company}_{kind_safe}.{ext_safe}"


def auto_download_files(files: List[Tuple[str, str, str]]) -> None:
    """Trigger browser downloads for one or more text files via injected JS.
    See SynthNotes-MultiDoc for design notes — same mechanism."""
    if not files:
        return
    blocks = []
    for i, (filename, content, mime) in enumerate(files):
        delay_ms = i * 600
        blocks.append(
            f"setTimeout(function() {{\n"
            f"  var blob = new Blob([{json.dumps(content)}], {{type: {json.dumps(mime)}}});\n"
            f"  var url = URL.createObjectURL(blob);\n"
            f"  var a = document.createElement('a');\n"
            f"  a.href = url;\n"
            f"  a.download = {json.dumps(filename)};\n"
            f"  document.body.appendChild(a);\n"
            f"  a.click();\n"
            f"  setTimeout(function() {{ URL.revokeObjectURL(url); document.body.removeChild(a); }}, 200);\n"
            f"}}, {delay_ms});"
        )
    js = "\n".join(blocks)
    components.html(f"<script>{js}</script>", height=0)


def copy_button(text: str, label: str = "Copy"):
    theme = st.context.theme
    bg = theme.get("primaryColor", "#FF4B4B")
    fg = theme.get("backgroundColor", "#FFFFFF")
    components.html(
        f"""
        <button onclick="doCopy()" style="background:{bg};color:{fg};border:none;padding:0.4rem 1.2rem;
            border-radius:0.3rem;cursor:pointer;font-size:0.875rem;width:100%;min-height:38px;">
          {html_module.escape(label)}
        </button>
        <script>
        function doCopy() {{
            var btn = document.querySelector('button');
            navigator.clipboard.writeText({json.dumps(text)}).then(() => {{
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = {json.dumps(label)}, 2000);
            }}).catch(() => {{
                btn.textContent = 'Failed — try again';
                setTimeout(() => btn.textContent = {json.dumps(label)}, 2000);
            }});
        }}
        </script>
        """,
        height=45,
    )


def api_key_check():
    if not _api_key:
        st.error("**GEMINI_API_KEY not set.** Add it to a `.env` file in this folder or set it in your environment.")
        st.code("GEMINI_API_KEY=your_key_here", language="bash")
        st.stop()


def render_usage_panel():
    """Per-stage token & cost breakdown rendered as a collapsible expander."""
    log = st.session_state.get("usage_log", [])
    if not log:
        return
    by_stage: dict = {}
    total_in, total_out, total_cost = 0, 0, 0.0
    for entry in log:
        s     = entry["stage"] or "other"
        model = entry["model"]
        cost  = compute_cost(entry["input_tokens"], entry["output_tokens"], model)
        slot  = by_stage.setdefault(s, {"input": 0, "output": 0, "cost": 0.0, "models": set()})
        slot["input"]  += entry["input_tokens"]
        slot["output"] += entry["output_tokens"]
        slot["cost"]   += cost
        slot["models"].add(model)
        total_in   += entry["input_tokens"]
        total_out  += entry["output_tokens"]
        total_cost += cost

    with st.expander(f"💰 Usage & cost — ~${total_cost:.4f} session-to-date", expanded=False):
        st.caption(
            "Approximate cost based on the hardcoded `MODEL_PRICING` table — verify against your "
            "Google Cloud project's billing reports for authoritative numbers. "
            "Resets when the app session restarts."
        )
        lines = [
            "| Stage | Model(s) | Input tokens | Output tokens | Cost (USD) |",
            "|---|---|---:|---:|---:|",
        ]
        for stage, vals in by_stage.items():
            models = ", ".join(sorted(vals["models"]))
            lines.append(
                f"| {stage} | `{models}` | {vals['input']:,} | {vals['output']:,} | ${vals['cost']:.4f} |"
            )
        lines.append(
            f"| **Total** | — | **{total_in:,}** | **{total_out:,}** | **${total_cost:.4f}** |"
        )
        st.markdown("\n".join(lines))
        if st.button("Reset usage counter", key="reset_usage_btn"):
            st.session_state.pop("usage_log", None)
            st.rerun()


class _PipelineComplete(Exception):
    """Sentinel raised inside the pipeline try-block to short-circuit cleanly.
    Caught right before the generic Exception handler — no error UI, no st.stop().
    Used by the extraction-only short-circuit so the output renderer still runs
    on the same script execution and shows the combined-extract download."""


# ── 4. CORE PROCESSING (extract → map → reduce) ────────────────────────────────

# ── Stage 0: extraction (per-source-type) ──────────────────────────────────────

def _extract_one_chunk(
    chunk_text: str, chunk_n: int, total_chunks: int, filename: str,
    extraction_prompt: str, model,
) -> Optional[str]:
    """Run the extraction prompt on a single chunk. Returns the model's output,
    which is either VERBATIM kept content or the EXTRACTION_SKIP_MARKER."""
    prompt = EXTRACTION_WRAPPER_TEMPLATE.format(
        extraction_prompt=extraction_prompt.strip(),
        filename=filename,
        chunk_n=chunk_n,
        total_chunks=total_chunks,
        chunk_text=chunk_text,
    )
    try:
        resp = generate_with_retry(model, prompt, stage="Extraction")
        return resp.text
    except Exception as e:
        return f"[Extraction failed for chunk {chunk_n} of {filename}: {e}]"


def extract_pass(
    files: List[Tuple[str, str]], extraction_prompt: str,
    chunk_size: int, overlap: int, model, status_write, source_type_label: str,
) -> List[Tuple[str, str]]:
    """Run extraction on a list of (filename, content) files. Returns a list of
    (filename, extracted_content) tuples — same shape as input, ready to feed
    downstream Map stage. Files with no retained content are dropped (with a warning).

    Global parallelism: builds a flat task list of ALL chunks across ALL files,
    then dispatches them to one worker pool. Massively faster than per-file
    sequential processing when chunks-per-file varies (which it always does).
    Files are reconstructed in order at the end via (file_idx, chunk_idx) keys.

    Chunk sizing is adaptive automatically — create_chunks_with_overlap returns
    a single chunk when len(words) <= chunk_size, so small files become single
    LLM calls without any branching here.
    """
    if not files:
        return []

    # ── Build flat task list across all files ──────────────────────────────────
    # Each task: (file_idx, chunk_idx, chunk_text, filename, total_chunks_in_file)
    tasks: List[Tuple[int, int, str, str, int]] = []
    per_file_chunk_counts: List[int] = []
    for file_idx, (filename, content) in enumerate(files):
        chunks = create_chunks_with_overlap(content, chunk_size, overlap)
        per_file_chunk_counts.append(len(chunks))
        for chunk_idx, chunk_text in enumerate(chunks):
            tasks.append((file_idx, chunk_idx, chunk_text, filename, len(chunks)))

    n_total = len(tasks)
    # Per-file chunk-count log (shows adaptive sizing — small files → 1 chunk)
    chunk_summary = ", ".join(
        f"{files[i][0]} → {per_file_chunk_counts[i]}"
        for i in range(len(files))
    )
    status_write(
        f"  {source_type_label}: {len(files)} file(s) → {n_total} chunk(s) total "
        f"(parallel × {EXTRACTION_PARALLEL_WORKERS}). Per-file: {chunk_summary}"
    )

    # ── Process all chunks in one global parallel pool ─────────────────────────
    results: dict = {}  # (file_idx, chunk_idx) → extracted_text
    with ThreadPoolExecutor(max_workers=EXTRACTION_PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _extract_one_chunk, chunk_text, chunk_idx + 1, total_in_file,
                filename, extraction_prompt, model,
            ): (file_idx, chunk_idx, filename)
            for file_idx, chunk_idx, chunk_text, filename, total_in_file in tasks
        }
        done = 0
        for fut in as_completed(futures):
            file_idx, chunk_idx, filename = futures[fut]
            results[(file_idx, chunk_idx)] = fut.result()
            done += 1
            status_write(f"    • {source_type_label}: {done}/{n_total} chunk(s) done")

    # ── Group results back by file, preserve chunk order, filter skip markers ──
    out: List[Tuple[str, str]] = []
    for file_idx, (filename, content) in enumerate(files):
        n_chunks = per_file_chunk_counts[file_idx]
        kept = []
        for chunk_idx in range(n_chunks):
            output = results.get((file_idx, chunk_idx))
            if not output or not output.strip():
                continue
            stripped = output.strip()
            # Skip if the chunk's whole output is the skip marker
            if stripped == EXTRACTION_SKIP_MARKER or stripped.startswith("[chunk skipped"):
                continue
            kept.append(stripped)

        if not kept:
            status_write(f"  ⚠️  {filename}: extraction returned no retained content — skipping this file")
            continue

        combined = "\n\n".join(kept)
        # Trim any trailing "[remainder skipped — non-narrative]" marker lines
        combined = re.sub(r"\n*\[remainder skipped[^\]]*\]\s*$", "", combined).strip()

        orig_words = len(content.split())
        kept_words = len(combined.split())
        retention = (kept_words / orig_words * 100) if orig_words else 0
        status_write(
            f"  ✓ {filename}: {orig_words:,} → {kept_words:,} words ({retention:.0f}% retained)"
        )
        out.append((filename, combined))

    return out


def serialize_combined_extract(
    extracted_ar: List[Tuple[str, str]],
    extracted_transcripts: List[Tuple[str, str]],
) -> str:
    """Build a single human-readable .txt file with per-source headers. This is
    the portable artefact the user downloads — can be pasted into any other tool."""
    lines = ["==== SynthNotes MultiDocLean — Combined Extracted Content ===="]
    total_kept = sum(len(c.split()) for _, c in (extracted_ar + extracted_transcripts))
    lines.append(
        f"{len(extracted_ar)} AR file(s) + {len(extracted_transcripts)} transcript file(s)  "
        f"|  total {total_kept:,} words after extraction"
    )
    lines.append("")
    for filename, content in extracted_ar:
        lines.append(f"==== From: {filename}  (annual report) ====")
        lines.append(content.strip())
        lines.append("")
    for filename, content in extracted_transcripts:
        lines.append(f"==== From: {filename}  (transcript) ====")
        lines.append(content.strip())
        lines.append("")
    return "\n".join(lines)


# ── Stage 1: Map (per-chunk synthesis notes on extracted content) ──────────────

def process_chunk(
    chunk_text: str, chunk_n: int, total_chunks: int,
    filename: str, file_position: int, total_files: int,
    user_prompt: str, model,
) -> Optional[str]:
    """Map stage: run the user prompt on a single chunk."""
    prompt = MAP_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt.strip(),
        filename=filename,
        chunk_n=chunk_n,
        total_chunks=total_chunks,
        file_position=file_position,
        total_files=total_files,
        chunk_text=chunk_text,
    )
    try:
        resp = generate_with_retry(model, prompt, stage="Map (per chunk)")
        return resp.text
    except Exception as e:
        return f"_[Section {chunk_n} of {filename} failed: {e}]_"


def _final_reduce(
    notes_list: List[str], filenames: List[str], user_prompt: str,
    target_word_count: int, reduce_model, status_write,
) -> str:
    """Run the FINAL reduce — applies chronology inference, length target, narrative synthesis."""
    combined = "\n\n".join(notes_list)
    prompt = REDUCE_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt.strip(),
        target_word_count=target_word_count,
        num_files=len(filenames),
        filename_list="\n".join(f"- {f}" for f in filenames),
        combined_notes=combined,
    )
    ph = st.empty()
    resp = generate_with_retry(
        reduce_model, prompt, stream=True,
        generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS},
        stage="Final reduce (synthesis)",
    )
    text, _ = stream_and_collect(resp, ph)
    return text


# ── Plan-then-write final synthesis ────────────────────────────────────────────
# Instead of asking the model to produce a 6000-word document in one shot (where
# LLMs reliably under-comply on length), we do it in two stages:
#   1. Outline pass: the model proposes a section-by-section plan with per-section
#      word budgets. Chronology is decided here, once.
#   2. Write pass: each section is written in parallel by a separate LLM call,
#      receiving the full outline (so it knows what other sections cover) and
#      the full notes (so it can extract what's relevant).
# Net effect: each section's small word target is reliably hit (LLMs nail small
# targets), so the global target is reliably hit too. Trade-off: 2–4× more LLM
# calls than single-pass reduce.

def _generate_outline(
    combined_notes: str, filenames: List[str], user_prompt: str,
    target_word_count: int, model,
) -> str:
    """Plan stage — ask the model to propose a section-by-section outline."""
    prompt = OUTLINE_PROMPT.format(
        user_prompt=user_prompt.strip(),
        target_word_count=target_word_count,
        num_files=len(filenames),
        filename_list="\n".join(f"- {f}" for f in filenames),
        combined_notes=combined_notes,
    )
    resp = generate_with_retry(model, prompt, stage="Plan (outline)")
    return resp.text


def _parse_outline(outline_text: str) -> List[dict]:
    """Extract sections from the outline text. Returns a list of dicts:
    [{'heading': str, 'coverage': str, 'budget': int}, ...]

    Robust to minor format variations from the model. Returns [] if parsing
    fails (caller should fall back to single-pass reduce in that case)."""
    sections = []
    current: Optional[dict] = None

    for raw_line in outline_text.split("\n"):
        line = raw_line.strip()
        # New section starts on a '## ' heading (NOT '# ' which is the document title)
        if line.startswith("## "):
            if current and current.get("heading"):
                sections.append(current)
            current = {"heading": line[3:].strip(), "coverage": "", "budget": 0}
        elif current is not None:
            # Coverage line
            cov_match = re.match(r"-\s*Coverage\s*:\s*(.+)$", line, re.IGNORECASE)
            if cov_match:
                current["coverage"] = cov_match.group(1).strip()
                continue
            # Word budget line — extract first number
            bud_match = re.match(r"-\s*Word\s*budget\s*:\s*(.+)$", line, re.IGNORECASE)
            if bud_match:
                num = re.search(r"(\d+)", bud_match.group(1))
                if num:
                    current["budget"] = int(num.group(1))
                continue

    if current and current.get("heading"):
        sections.append(current)

    # Filter out malformed sections (no budget set)
    return [s for s in sections if s["heading"] and s["budget"] > 0]


def _extract_outline_metadata(outline_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the document title (# heading) and the CHRONOLOGY_NOTE line, if present."""
    title = None
    title_match = re.search(r"^#\s+(.+?)$", outline_text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
        # Don't mistake a `## section` heading for the title
        if title.startswith("#"):
            title = None

    chrono = None
    chrono_match = re.search(r"CHRONOLOGY_NOTE\s*:\s*(.+?)$", outline_text, re.MULTILINE)
    if chrono_match:
        chrono = chrono_match.group(1).strip()

    return title, chrono


def _write_section(
    section: dict, section_n: int, total_sections: int,
    outline_text: str, combined_notes: str,
    user_prompt: str, model,
) -> str:
    """Write stage — generate ONE section per the outline's assignment."""
    prompt = SECTION_PROMPT.format(
        user_prompt=user_prompt.strip(),
        section_heading=section["heading"],
        section_coverage=section.get("coverage", "(no specific coverage stated)"),
        section_word_budget=section.get("budget", 500),
        section_n=section_n,
        total_sections=total_sections,
        outline_text=outline_text,
        combined_notes=combined_notes,
    )
    resp = generate_with_retry(model, prompt, stage="Write (section)")
    return resp.text


def plan_then_write_final(
    notes_list: List[str], filenames: List[str], user_prompt: str,
    target_word_count: int, model, status_write,
) -> str:
    """Final synthesis via plan-then-write pattern: outline → parallel sections → stitch.
    Falls back to single-pass _final_reduce if the outline can't be parsed."""
    combined_notes = "\n\n".join(notes_list)

    # ── Step 1: Plan ───────────────────────────────────────────────────────────
    status_write(f"📋 PLAN stage — generating outline for ~{target_word_count}-word document…")
    outline_text = _generate_outline(combined_notes, filenames, user_prompt, target_word_count, model)
    sections = _parse_outline(outline_text)

    if not sections:
        status_write("⚠️  Could not parse outline — falling back to single-pass synthesis")
        return _final_reduce(notes_list, filenames, user_prompt, target_word_count, model, status_write)

    title, chronology_note = _extract_outline_metadata(outline_text)
    total_budget = sum(s["budget"] for s in sections)
    status_write(
        f"📋 Outline: **{len(sections)} sections**, total budget {total_budget:,} words"
        + (f", title: '{title}'" if title else "")
    )

    # ── Step 2: Write (parallel per-section generation) ───────────────────────
    status_write(f"✏️  WRITE stage — generating {len(sections)} sections in parallel…")
    results: List[Optional[str]] = [None] * len(sections)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _write_section, section, i + 1, len(sections),
                outline_text, combined_notes, user_prompt, model,
            ): i
            for i, section in enumerate(sections)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            wc = len(results[i].split()) if results[i] else 0
            s = sections[i]
            status_write(
                f"  • Section {done}/{len(sections)}: '{s['heading']}' "
                f"({wc:,} words written, budget was ~{s['budget']:,})"
            )

    # ── Step 3: Stitch ─────────────────────────────────────────────────────────
    body_parts = [r.strip() for r in results if r and r.strip()]
    document_parts: List[str] = []
    if title:
        document_parts.append(f"# {title}")
    if chronology_note:
        document_parts.append(f"*{chronology_note}*")
    document_parts.extend(body_parts)

    stitched = "\n\n".join(document_parts)
    actual_words = len(stitched.split())
    status_write(
        f"✓ Plan-then-write complete — **{actual_words:,} words** "
        f"({actual_words / target_word_count * 100:.0f}% of {target_word_count:,}-word target)"
    )
    return stitched


def _intermediate_reduce(
    notes_batch: List[str], user_prompt: str,
    target_words: int, reduce_model, depth: int,
) -> str:
    """Run ONE intermediate reduce — compress a batch of notes while preserving info richness."""
    combined = "\n\n".join(notes_batch)
    prompt = INTERMEDIATE_REDUCE_PROMPT.format(
        user_prompt=user_prompt.strip(),
        target_word_count=target_words,
        combined_notes=combined,
    )
    resp = generate_with_retry(
        reduce_model, prompt,
        stage=f"Intermediate reduce (depth {depth})",
    )
    return resp.text


def _group_into_batches(notes_list: List[str], max_words_per_batch: int) -> List[List[str]]:
    """Greedy bin-packing: walk notes in order, start a new batch when adding the next
    note would exceed the budget. Preserves the original order of notes."""
    batches: List[List[str]] = []
    current: List[str] = []
    current_words = 0
    for note in notes_list:
        wc = len(note.split())
        if current_words + wc > max_words_per_batch and current:
            batches.append(current)
            current = [note]
            current_words = wc
        else:
            current.append(note)
            current_words += wc
    if current:
        batches.append(current)
    return batches


def hierarchical_reduce(
    notes_list: List[str], filenames: List[str], user_prompt: str,
    target_word_count: int, reduce_model, status_write, depth: int = 0,
) -> str:
    """Reduce per-chunk notes into a final document, recursing through intermediate
    reduces when the input exceeds the per-call word budget.

    Algorithm:
      - If notes_list fits in WORDS_PER_REDUCE_BATCH or is just one note → final reduce
      - Otherwise → group into batches, intermediate-reduce each batch in parallel,
        recurse with the resulting intermediate summaries
      - Cap recursion at MAX_REDUCE_DEPTH to fail loudly on pathological input
    """
    total_words = sum(len(n.split()) for n in notes_list)
    fits_in_one_call = total_words <= WORDS_PER_REDUCE_BATCH or len(notes_list) <= 1

    if fits_in_one_call:
        if depth == 0:
            status_write(f"Notes fit in one synthesis pass (~{total_words:,} words). Starting plan-then-write…")
        else:
            status_write(f"[Depth {depth}] Notes fit in one synthesis pass (~{total_words:,} words after compression). Starting plan-then-write…")
        # Plan-then-write: produces a structured outline, then writes each section in parallel.
        # More reliable length compliance than single-pass synthesis; also generally higher quality.
        # Falls back to single-pass _final_reduce internally if the outline can't be parsed.
        return plan_then_write_final(notes_list, filenames, user_prompt, target_word_count, reduce_model, status_write)

    if depth >= MAX_REDUCE_DEPTH:
        raise ValueError(
            f"Input remains too large after {MAX_REDUCE_DEPTH} levels of hierarchical reduce "
            f"(still {total_words:,} words across {len(notes_list)} notes at depth {depth}). "
            "Consider: a smaller target word count, fewer source files, or running the workload as separate sub-runs."
        )

    # Split into batches and intermediate-reduce in parallel
    batches = _group_into_batches(notes_list, WORDS_PER_REDUCE_BATCH)
    status_write(
        f"[Depth {depth}] {total_words:,} words exceeds {WORDS_PER_REDUCE_BATCH:,}-word budget — "
        f"splitting into {len(batches)} batch(es) and compressing each…"
    )

    # Each intermediate summary should preserve detail; target ~5000 words per batch is generous
    intermediate_target = 5000
    results: List[Optional[str]] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _intermediate_reduce, batch, user_prompt, intermediate_target, reduce_model, depth,
            ): i
            for i, batch in enumerate(batches)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            status_write(f"[Depth {depth}] Batch {done}/{len(batches)} compressed")

    intermediates = [r for r in results if r and r.strip()]
    if not intermediates:
        raise ValueError(f"All intermediate reduces at depth {depth} returned empty output.")

    # Recurse with the (smaller) intermediate summaries
    return hierarchical_reduce(
        intermediates, filenames, user_prompt, target_word_count,
        reduce_model, status_write, depth + 1,
    )


# ── Interim notes: save/load ───────────────────────────────────────────────────
# Lets the user save the Map-stage output as a .txt file, then re-run the Reduce
# stage later (with possibly different prompt or target length) without re-paying
# for the Map stage. The serialisation format is human-readable and round-trips.

INTERIM_FILE_HEADER = "==== SynthNotes MultiDocLean — Interim Notes ===="
INTERIM_SECTION_SEPARATOR = "==== SECTION ===="


def serialize_interim(notes_list: List[str], filenames: List[str]) -> str:
    """Serialize per-chunk notes + filenames to a single .txt file."""
    header_lines = [
        INTERIM_FILE_HEADER,
        f"Generated from {len(filenames)} source file(s)",
        "Source filenames:",
    ]
    for f in filenames:
        header_lines.append(f"- {f}")
    header_lines.append("")  # blank line between header and first section
    header = "\n".join(header_lines)
    body = ("\n\n" + INTERIM_SECTION_SEPARATOR + "\n").join(notes_list)
    return header + "\n" + INTERIM_SECTION_SEPARATOR + "\n" + body + "\n"


def parse_interim(text: str) -> Tuple[List[str], List[str]]:
    """Parse a saved interim file. Returns (notes_list, filenames).
    Falls back gracefully if the file doesn't match the expected format."""
    filenames: List[str] = []
    # Extract filenames from header block, if present
    header_match = re.search(r"Source filenames:\s*\n((?:-\s.+\n?)+)", text)
    if header_match:
        for line in header_match.group(1).splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                filenames.append(stripped[2:].strip())

    # Split body on section separator
    if INTERIM_SECTION_SEPARATOR in text:
        parts = text.split(INTERIM_SECTION_SEPARATOR)
        # First part is header (everything before the first separator); rest are notes
        notes = [p.strip() for p in parts[1:] if p.strip()]
    else:
        # Fallback for plain-text input — treat the whole file as one note
        notes = [text.strip()]

    return notes, filenames or ["(filenames not recorded in interim file)"]


# ── PDF export ─────────────────────────────────────────────────────────────────

def markdown_to_pdf_bytes(md_text: str, title: str = "SynthNotes MultiDocLean Output") -> Optional[bytes]:
    """Render markdown to a PDF byte string via markdown → HTML → xhtml2pdf.
    Returns None if the optional PDF dependencies aren't installed, so the app
    degrades gracefully (the .md download remains available)."""
    try:
        import markdown as md_lib
        from xhtml2pdf import pisa
        import io
    except ImportError:
        return None

    html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
    safe_title = html_module.escape(title)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
  @page {{ size: A4; margin: 1.5cm; }}
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4; color: #1a1a2e; }}
  h1 {{ font-size: 18pt; border-bottom: 1px solid #ccc; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 14pt; margin-top: 18px; color: #1a1a2e; }}
  h3 {{ font-size: 12pt; margin-top: 14px; }}
  h4 {{ font-size: 11pt; margin-top: 12px; }}
  p  {{ margin: 6px 0; }}
  ul, ol {{ margin: 6px 0; padding-left: 22px; }}
  li {{ margin: 3px 0; }}
  code {{ background: #f4f4f4; padding: 1px 4px; font-family: 'Courier New', monospace; font-size: 9.5pt; }}
  pre {{ background: #f4f4f4; padding: 8px; font-family: 'Courier New', monospace; font-size: 9.5pt; }}
  blockquote {{ border-left: 3px solid #ccc; padding-left: 10px; margin-left: 0; color: #555; font-style: italic; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  td, th {{ border: 1px solid #ddd; padding: 5px 8px; font-size: 10pt; }}
  th {{ background: #f0f2f6; font-weight: bold; }}
  strong, b {{ font-weight: bold; }}
  em, i {{ font-style: italic; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    pdf_buf = io.BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_buf)
    if pisa_status.err:
        return None
    pdf_buf.seek(0)
    return pdf_buf.read()


# ── 5. UI ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SynthNotes MultiDocLean", layout="wide", page_icon="📚")
    st.title("📚 SynthNotes MultiDocLean")
    st.caption(
        "Multi-file note synthesis. Upload many .txt files, provide your own prompt, "
        "get one consolidated document with chronology inferred from content. "
        "Borrows the chunking and retry logic from SynthNotes Pro."
    )
    api_key_check()

    # ── Sidebar: model settings ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Model Settings")
        extract_model_name = st.selectbox(
            "Extraction model (Stage 0)",
            list(MODELS.keys()), index=1,  # Flash-Lite — cheapest, fine for keep/skip classification
            key="extract_model",
            help=(
                "Filters source documents — keeps narrative sections, drops boilerplate. "
                "Mechanical keep/skip classification — Flash-Lite is plenty. "
                "Cost is roughly ~$0.10 per AR at default settings."
            ),
        )
        map_model_name = st.selectbox(
            "Per-section model (Map stage)",
            list(MODELS.keys()), index=0,
            key="map_model",
            help=(
                "Used to extract notes from each chunk of the EXTRACTED content. "
                "Cheap, fast models work well here since the work is mechanical extraction."
            ),
        )
        reduce_model_name = st.selectbox(
            "Synthesis model (Reduce stage)",
            list(MODELS.keys()), index=3,  # Gemini 3.0 Flash — Flash-tier pricing, capable reasoning
            key="reduce_model",
            help=(
                "Used for outline generation and per-section writing during synthesis. "
                "Default is **Gemini 3.0 Flash** (`gemini-3-flash-preview`) — Flash-tier "
                "pricing ($0.50 / $3.00 per 1M tokens) with strong reasoning. ~3× cheaper "
                "than 2.5 Pro. Switch to 2.5 Pro if sections come out thin or the outline "
                "feels shallow. Note: preview model — could be deprecated by Google with notice."
            ),
        )

    # ── Mode toggle ────────────────────────────────────────────────────────────
    # ── Company name (used in download filenames; auto-downloads use this) ─────
    company_name = st.text_input(
        "Company name (used in auto-downloaded filenames; optional)",
        placeholder="e.g. Hitachi Energy",
        key="company_name",
        help=(
            "Used to name auto-downloaded files: "
            "**YYYYMMDD_<company>_synthnotes.md/.txt** for the final document, "
            "**YYYYMMDD_<company>_extract.txt** for the combined extracted text. "
            "If blank, files are named with 'untitled' instead."
        ),
    )

    st.markdown("### 1. Input mode")
    mode = st.radio(
        "Start from",
        ["Source .txt files (full pipeline)", "Saved interim notes (skip Map, re-run synthesis only)"],
        key="input_mode",
        label_visibility="collapsed",
        help=(
            "**Source files** — full pipeline: Map (per-chunk extraction) → Reduce (synthesis). "
            "Use this when you have raw source documents.\n\n"
            "**Saved interim notes** — load an interim .txt downloaded from a previous run, "
            "and re-run only the Reduce stage. Cheap and fast for iterating on prompt or length."
        ),
    )
    is_interim_mode = mode.startswith("Saved interim")

    # ── Conditional input section ──────────────────────────────────────────────
    uploaded_ar: List = []
    uploaded_transcripts: List = []
    interim_uploaded = None
    combine_files = False  # legacy — extraction supersedes this concept

    if not is_interim_mode:
        st.markdown("### 2. Annual reports")
        uploaded_ar = st.file_uploader(
            f"Upload annual-report .txt files (up to {MAX_FILES})",
            type=["txt"], accept_multiple_files=True,
            help="Extraction will keep narrative sections (Chairman's letter, MD&A, business review, strategy, outlook) and drop financial statements, notes to accounts, related-party items, statutory reports, and accounting policies.",
            key="ar_uploader",
        )
        if uploaded_ar:
            ar_total_words = 0
            with st.expander(f"✓ {len(uploaded_ar)} AR file(s) loaded — view list", expanded=False):
                for f in uploaded_ar:
                    try:
                        sample = f.getvalue().decode("utf-8", errors="replace")
                        wc = len(sample.split())
                        ar_total_words += wc
                        st.text(f"  • {f.name}  ({f.size/1024:.1f} KB, ~{wc:,} words)")
                    except Exception:
                        st.text(f"  • {f.name}  ({f.size/1024:.1f} KB)")
            st.caption(f"Total AR input: ~{ar_total_words:,} words across {len(uploaded_ar)} file(s). Typical retention after extraction: ~25-35%.")

        st.markdown("### 3. Quarterly call transcripts")
        uploaded_transcripts = st.file_uploader(
            f"Upload transcript .txt files (up to {MAX_FILES})",
            type=["txt"], accept_multiple_files=True,
            help="Extraction will keep management opening remarks and Q&A substance; drop operator greetings, safe-harbour disclaimers, analyst pleasantries.",
            key="transcripts_uploader",
        )
        if uploaded_transcripts:
            tx_total_words = 0
            with st.expander(f"✓ {len(uploaded_transcripts)} transcript file(s) loaded — view list", expanded=False):
                for f in uploaded_transcripts:
                    try:
                        sample = f.getvalue().decode("utf-8", errors="replace")
                        wc = len(sample.split())
                        tx_total_words += wc
                        st.text(f"  • {f.name}  ({f.size/1024:.1f} KB, ~{wc:,} words)")
                    except Exception:
                        st.text(f"  • {f.name}  ({f.size/1024:.1f} KB)")
            st.caption(f"Total transcript input: ~{tx_total_words:,} words across {len(uploaded_transcripts)} file(s). Typical retention: ~65-75%.")
    else:
        st.markdown("### 2. Interim notes file")
        interim_uploaded = st.file_uploader(
            "Upload a saved interim notes .txt (from a previous run)",
            type=["txt"], accept_multiple_files=False,
            key="interim_uploader",
            help=(
                "The interim file is what you downloaded from a previous output as "
                "**'Download interim notes (.txt)'**. Loading it skips extraction AND Map; "
                "only Reduce + plan-then-write runs."
            ),
        )
        if interim_uploaded:
            size_kb = interim_uploaded.size / 1024
            st.info(f"✓ Interim file loaded: **{interim_uploaded.name}** ({size_kb:.1f} KB)")

    # ── AR extraction prompt (editable) ────────────────────────────────────────
    st.markdown("### 4. AR extraction prompt — what to keep from annual reports")
    with st.expander("View / edit AR extraction prompt (default shipped)", expanded=False):
        ar_extraction_prompt = st.text_area(
            "AR extraction prompt",
            value=DEFAULT_EXTRACTION_AR_PROMPT,
            height=400,
            key="ar_extraction_prompt",
            label_visibility="collapsed",
            help="Defines what counts as analyst-relevant in annual reports. Edit to match your shop's view of which sections matter.",
        )

    # ── Transcript extraction prompt (editable) ────────────────────────────────
    st.markdown("### 5. Transcript extraction prompt — what to keep from transcripts")
    with st.expander("View / edit transcript extraction prompt (default shipped)", expanded=False):
        transcript_extraction_prompt = st.text_area(
            "Transcript extraction prompt",
            value=DEFAULT_EXTRACTION_TRANSCRIPT_PROMPT,
            height=350,
            key="transcript_extraction_prompt",
            label_visibility="collapsed",
            help="Defines what to keep from transcripts. Defaults retain mgmt opening + Q&A; drop greetings and disclaimers.",
        )

    # ── Synthesis prompt ───────────────────────────────────────────────────────
    st.markdown("### 6. Synthesis prompt — what the final note should look like")
    st.caption(
        "A default prompt is shipped — an equity-analyst story-style company note. "
        "Edit it or replace it entirely for other use cases. Used in BOTH Map and Reduce stages "
        "of synthesis (i.e., after extraction has filtered the source documents)."
    )
    with st.expander("View / edit synthesis prompt (default shipped)", expanded=False):
        user_prompt = st.text_area(
            "Synthesis prompt",
            value=DEFAULT_USER_PROMPT,
            height=500,
            key="user_prompt",
            label_visibility="collapsed",
        )

    # ── Length & chunk size ────────────────────────────────────────────────────
    st.markdown("### 7. Output length")
    length_label = st.radio(
        "Target length",
        list(LENGTH_PRESETS.keys()),
        horizontal=True,
        key="length_preset",
        label_visibility="collapsed",
    )
    word_count = LENGTH_PRESETS[length_label]
    if word_count is None:
        word_count = st.number_input(
            "Custom target word count",
            min_value=500, max_value=15000, value=8000, step=500,
            key="custom_wc",
        )

    chunk_size, overlap = compute_chunk_params(word_count)
    st.caption(
        f"📐 Internal chunk size auto-set to **{chunk_size:,} words** per section, "
        f"**{overlap}-word overlap** between sections. "
        f"Smaller targets use larger chunks (more compression); larger targets use smaller chunks (more granular extraction)."
    )

    # ── Pre-flight cost estimate ───────────────────────────────────────────────
    # Source-files mode only (interim mode skips extraction + Map and is much cheaper).
    if not is_interim_mode and (uploaded_ar or uploaded_transcripts):
        ar_words_total = sum(
            len(f.getvalue().decode("utf-8", errors="replace").split()) for f in (uploaded_ar or [])
        )
        tx_words_total = sum(
            len(f.getvalue().decode("utf-8", errors="replace").split()) for f in (uploaded_transcripts or [])
        )
        extract_model_id = MODELS.get(extract_model_name, "gemini-2.5-flash-lite")
        map_model_id     = MODELS.get(map_model_name,     "gemini-2.5-flash")
        reduce_model_id  = MODELS.get(reduce_model_name,  "gemini-2.5-pro")
        est = estimate_pipeline_cost(
            ar_words_total, tx_words_total, word_count,
            extract_model_id, map_model_id, reduce_model_id,
        )
        # Scope cost to whatever stages will actually run.
        # NOTE: stop_after_extraction is defined just below this block, so we
        # re-derive it from session state here (radio key="pipeline_scope").
        _scope_value = st.session_state.get("pipeline_scope", "Full pipeline")
        _ext_only    = isinstance(_scope_value, str) and _scope_value.startswith("Extraction only")
        if _ext_only:
            run_cost = est["extract_cost"]
            run_desc = "Extraction only (Stage 0)"
        else:
            run_cost = est["total_cost"]
            run_desc = "Full pipeline (Extract → Map → Reduce)"
        with st.expander(
            f"💰 Estimated cost for this run: **~${run_cost:.2f}**  ({run_desc})",
            expanded=False,
        ):
            st.caption(
                "Rough estimate at current input size and model selection. Pipeline is "
                "Extraction → Map → Outline → ~8 section calls. Actual cost typically lands "
                "within ±30% — biggest variables are per-chunk Map output verbosity and the "
                "actual retention percentage of extraction (assumed ~30% for ARs and ~70% for "
                "transcripts; varies by source)."
            )
            cost_lines = [
                "| Stage | Model | Est. cost (USD) |",
                "|---|---|---:|",
                f"| Extraction — ARs ({est['ar_chunks']:,} chunks) + transcripts ({est['transcript_chunks']:,} chunks) | `{extract_model_id}` | ${est['extract_cost']:.4f} |",
                f"| Map — {est['n_map_chunks']:,} chunks on extracted content @ {est['chunk_size']:,}-word chunks | `{map_model_id}` | ${est['map_cost']:.4f} |",
                f"| Outline (1 call, all Map output) | `{reduce_model_id}` | ${est['outline_cost']:.4f} |",
                f"| Section writing (~8 sections, full Map output each) | `{reduce_model_id}` | ${est['section_cost']:.4f} |",
                f"| **Total** | — | **${est['total_cost']:.4f}** |",
            ]
            st.markdown("\n".join(cost_lines))
            st.caption(
                f"Input scale: ARs {ar_words_total:,} words ({len(uploaded_ar or [])} file(s)), "
                f"transcripts {tx_words_total:,} words ({len(uploaded_transcripts or [])} file(s)). "
                f"Extracted estimate: ~{est['extracted_words']:,} words feeding synthesis."
            )

    # ── Pipeline scope (extraction-only short-circuit) ─────────────────────────
    # Only relevant in source-files mode (interim mode always runs Reduce-only).
    stop_after_extraction = False
    if not is_interim_mode:
        st.markdown("### 8. Pipeline scope")
        scope_choice = st.radio(
            "How far to run",
            [
                "Full pipeline — Extract → Map → Reduce → final document",
                "Extraction only — just produce the combined extracted .txt and stop",
            ],
            index=0,
            key="pipeline_scope",
            label_visibility="collapsed",
            help=(
                "**Full pipeline (default)** — runs the whole thing end-to-end.\n\n"
                "**Extraction only** — runs Stage 0 (extraction) and stops. Produces the "
                "combined extracted .txt file, which you can download and use for anything: "
                "feed into MultiDoc, paste into Gemini AI Studio, hand to a colleague, archive. "
                "Skips Map + Reduce entirely — synthesis prompt and target length are ignored "
                "in this mode."
            ),
        )
        stop_after_extraction = scope_choice.startswith("Extraction only")

    # ── Process ────────────────────────────────────────────────────────────────
    st.divider()
    button_label = "Generate consolidated document"
    if stop_after_extraction:
        button_label = "Run extraction only (produce combined .txt and stop)"
    if st.button(button_label, type="primary", use_container_width=True):
        if not stop_after_extraction and not user_prompt.strip():
            st.error("Please provide a synthesis prompt.")
            st.stop()

        # ── Branch on mode: gather (all_notes, filenames) ─────────────────────
        all_notes: List[str] = []
        filenames: List[str] = []
        extracted_ar: List[Tuple[str, str]] = []
        extracted_transcripts: List[Tuple[str, str]] = []

        if is_interim_mode:
            if not interim_uploaded:
                st.error("Please upload a saved interim notes file.")
                st.stop()
            try:
                interim_text = interim_uploaded.getvalue().decode("utf-8", errors="replace")
                all_notes, filenames = parse_interim(interim_text)
                if not all_notes:
                    st.error("The uploaded interim file produced no parsable section notes.")
                    st.stop()
            except Exception as e:
                st.error(f"Could not parse interim file: {e}")
                st.stop()
        else:
            if not uploaded_ar and not uploaded_transcripts:
                st.error("Please upload at least one annual report or one transcript.")
                st.stop()
            if not ar_extraction_prompt.strip() or not transcript_extraction_prompt.strip():
                st.error("Please provide both extraction prompts.")
                st.stop()

            # Decode files (UTF-8 with BOM/Latin-1 fallback) into (name, content) tuples
            def _decode_bucket(uploads):
                out = []
                for f in (uploads or []):
                    size_mb = f.size / (1024 * 1024)
                    if size_mb > MAX_FILE_SIZE_MB:
                        st.error(f"`{f.name}`: file too large ({size_mb:.1f} MB; limit {MAX_FILE_SIZE_MB} MB).")
                        st.stop()
                    data = f.getvalue()
                    content = None
                    for enc in ("utf-8", "utf-8-sig", "latin-1"):
                        try:
                            content = data.decode(enc)
                            break
                        except UnicodeDecodeError:
                            continue
                    if content is None:
                        st.error(f"`{f.name}`: could not decode (UTF-8/Latin-1).")
                        st.stop()
                    if content.strip():
                        out.append((f.name, content))
                    else:
                        st.warning(f"`{f.name}` is empty — skipping.")
                return out

            ar_files = _decode_bucket(uploaded_ar)
            tx_files = _decode_bucket(uploaded_transcripts)
            if not ar_files and not tx_files:
                st.error("All uploaded files were empty after decoding. Please upload non-empty .txt files.")
                st.stop()

        extract_model = get_model(extract_model_name)
        map_model     = get_model(map_model_name)
        reduce_model  = get_model(reduce_model_name)

        # Fresh usage log per run
        st.session_state["usage_log"] = []

        with st.status("Processing…", expanded=True) as status:
            try:
                if not is_interim_mode:
                    # ── STAGE 0: EXTRACTION (per-bucket) ──────────────────────
                    st.write(
                        f"**STAGE 0 — EXTRACTION** — "
                        f"{len(ar_files)} AR file(s), {len(tx_files)} transcript file(s)"
                    )
                    if ar_files:
                        extracted_ar = extract_pass(
                            ar_files, ar_extraction_prompt,
                            EXTRACTION_CHUNK_SIZE_AR, EXTRACTION_CHUNK_OVERLAP_AR,
                            extract_model, st.write, source_type_label="AR",
                        )
                    if tx_files:
                        extracted_transcripts = extract_pass(
                            tx_files, transcript_extraction_prompt,
                            EXTRACTION_CHUNK_SIZE_TRANSCRIPT, EXTRACTION_CHUNK_OVERLAP_TRANSCRIPT,
                            extract_model, st.write, source_type_label="Transcript",
                        )

                    if not extracted_ar and not extracted_transcripts:
                        raise ValueError(
                            "Extraction returned no content for any file. Check that your "
                            "source files have analyst-relevant narrative material."
                        )

                    extracted_words = sum(
                        len(c.split()) for _, c in extracted_ar + extracted_transcripts
                    )
                    st.write(
                        f"✓ Extraction complete — {len(extracted_ar)} AR + {len(extracted_transcripts)} "
                        f"transcript file(s) retained, ~{extracted_words:,} total words feeding synthesis"
                    )

                    # Save combined extract to session state for download
                    st.session_state["combined_extract_text"] = serialize_combined_extract(
                        extracted_ar, extracted_transcripts,
                    )
                    # Track sources for the output renderer
                    st.session_state["ar_filenames"] = [f[0] for f in extracted_ar]
                    st.session_state["transcript_filenames"] = [f[0] for f in extracted_transcripts]

                    # ── EARLY EXIT: stop after extraction ─────────────────────
                    # We use a sentinel exception (NOT st.stop) so the script still
                    # runs to completion — including the output-rendering block below
                    # the button handler, which is what shows the download UI.
                    if stop_after_extraction:
                        st.session_state["stopped_after"]  = "extraction"
                        # Clear any prior final document so the output renderer
                        # doesn't show a stale full-pipeline result
                        st.session_state.pop("final_document", None)
                        st.session_state.pop("interim_notes_text", None)
                        status.update(label="Done — stopped after extraction", state="complete")
                        st.write(
                            f"**✓ Combined extracted .txt ready** — {extracted_words:,} words. "
                            f"Download from the output section below."
                        )
                        # Stage auto-download — fires once on the next render
                        st.session_state["pending_auto_download"] = [
                            (
                                filename_for(company_name, "extract", "txt"),
                                st.session_state["combined_extract_text"],
                                "text/plain",
                            ),
                        ]
                        raise _PipelineComplete()

                    # Feed extracted content into the existing Map pipeline as if they
                    # were original files. Per-source attribution preserved naturally.
                    files = extracted_ar + extracted_transcripts
                    filenames = [f[0] for f in files]

                    # Build a flat task list across all chunks of all files
                    tasks: List[Tuple[int, int, int, str, str]] = []
                    for file_idx, (filename, content) in enumerate(files, start=1):
                        chunks = create_chunks_with_overlap(content, chunk_size, overlap)
                        for i, chunk_text in enumerate(chunks, start=1):
                            tasks.append((file_idx, i, len(chunks), filename, chunk_text))

                    st.write(
                        f"**STAGE 1 — MAP** — {len(tasks)} chunk(s) across "
                        f"{len(files)} extracted file(s), parallel × {PARALLEL_WORKERS}"
                    )

                    results: List[Optional[str]] = [None] * len(tasks)
                    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                        futures = {
                            executor.submit(
                                process_chunk,
                                chunk_text, chunk_n, total_chunks, filename,
                                file_position, len(files), user_prompt, map_model,
                            ): idx
                            for idx, (file_position, chunk_n, total_chunks, filename, chunk_text)
                            in enumerate(tasks)
                        }
                        done = 0
                        for fut in as_completed(futures):
                            idx = futures[fut]
                            results[idx] = fut.result()
                            done += 1
                            st.write(f"  • Map: {done}/{len(tasks)} chunks complete")

                    all_notes = [r for r in results if r and r.strip()]
                    if not all_notes:
                        raise ValueError("Map stage produced no notes — check input files and prompt.")

                    map_words = sum(len(n.split()) for n in all_notes)
                    st.write(
                        f"✓ Map stage complete — {len(all_notes)} chunk notes, "
                        f"~{map_words:,} words of intermediate content"
                    )
                else:
                    interim_words = sum(len(n.split()) for n in all_notes)
                    st.write(
                        f"⏩ Skipping extraction + Map — using {len(all_notes)} pre-extracted "
                        f"section notes (~{interim_words:,} words) from interim file"
                    )

                # ── Save interim notes to session state (for download) ────────
                st.session_state["interim_notes_text"] = serialize_interim(all_notes, filenames)

                # ── STAGE 2: REDUCE (hierarchical + plan-then-write) ──────────
                st.write(f"**STAGE 2 — REDUCE** — synthesising ~{word_count:,}-word final document")
                final_doc = hierarchical_reduce(
                    all_notes, filenames, user_prompt, word_count,
                    reduce_model, st.write, depth=0,
                )
                if not final_doc.strip():
                    raise ValueError("Synthesis returned empty output.")

                st.session_state["final_document"]   = final_doc
                st.session_state["source_filenames"] = filenames
                st.session_state["target_words"]     = word_count

                # ── Stage auto-download — fires once on the next render ─────
                # Full pipeline: auto-download the final document as .md + .txt,
                # plus the combined extract as .txt (in case session is lost
                # before user manually downloads).
                _pending: List[Tuple[str, str, str]] = [
                    (filename_for(company_name, "synthnotes", "md"),  final_doc, "text/markdown"),
                    (filename_for(company_name, "synthnotes", "txt"), final_doc, "text/plain"),
                ]
                _combined_ext = st.session_state.get("combined_extract_text", "")
                if _combined_ext:
                    _pending.append(
                        (filename_for(company_name, "extract", "txt"), _combined_ext, "text/plain")
                    )
                st.session_state["pending_auto_download"] = _pending

                actual_words = len(final_doc.split())
                pct_of_target = actual_words / word_count * 100
                status.update(label="Done!", state="complete")
                st.write(
                    f"✓ Final document: **{actual_words:,} words** "
                    f"({pct_of_target:.0f}% of {word_count:,}-word target)"
                )

            except _PipelineComplete:
                # Normal early exit (e.g., "extraction only" mode). Already logged
                # inside the try block; nothing else to do here.
                pass
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # ── Auto-download trigger — fires exactly once after a successful run ─────
    # Popped so it doesn't re-fire on subsequent reruns. Works for both full
    # pipeline (final doc .md+.txt, plus extract .txt) and extraction-only
    # (just the combined extract .txt).
    _pending_dl = st.session_state.pop("pending_auto_download", None)
    if _pending_dl:
        auto_download_files(_pending_dl)
        st.success(
            "✓ Auto-downloaded to your downloads folder: " +
            " · ".join(f"`{f[0]}`" for f in _pending_dl) +
            "  *(first multi-file download per site may prompt your browser to allow it)*"
        )

    # ── Output ─────────────────────────────────────────────────────────────────
    stopped = st.session_state.get("stopped_after", "")
    if "final_document" in st.session_state or stopped == "extraction":
        st.divider()
        sources      = st.session_state.get("source_filenames", [])
        target       = st.session_state.get("target_words", 0)
        interim_text  = st.session_state.get("interim_notes_text", "")
        combined_extract_text = st.session_state.get("combined_extract_text", "")
        ar_names = st.session_state.get("ar_filenames", [])
        tx_names = st.session_state.get("transcript_filenames", [])

        if stopped == "extraction":
            # ── Extraction-only mode — combined extract is THE deliverable ──
            ext_words = len(combined_extract_text.split())
            col_title, col_copy, col_dl = st.columns([2, 1, 1])
            with col_title:
                st.subheader(f"Combined Extracted Content  ({ext_words:,} words)")
            with col_copy:
                copy_button(combined_extract_text, "Copy")
            with col_dl:
                st.download_button(
                    "Download .txt",
                    data=combined_extract_text,
                    file_name=filename_for(company_name, "extract", "txt"),
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_extract_primary",
                )
            st.caption(
                "Pipeline stopped after Stage 0 (extraction). Map + Reduce were skipped. "
                "Use this .txt as input to MultiDoc, paste into Gemini AI Studio, or pass to "
                "any other tool. To run the full pipeline, switch **Pipeline scope** above to "
                "*Full pipeline* and click Generate again."
            )
            with st.expander(
                f"Source files used (AR: {len(ar_names)}, transcripts: {len(tx_names)})",
                expanded=False,
            ):
                if ar_names:
                    st.markdown("**Annual reports:**")
                    for s in ar_names:
                        st.text(f"  • {s}")
                if tx_names:
                    st.markdown("**Quarterly transcripts:**")
                    for s in tx_names:
                        st.text(f"  • {s}")
            st.markdown(combined_extract_text)
        else:
            # ── Full pipeline — final document is THE deliverable ────────────
            doc = st.session_state["final_document"]
            col_title, col_copy, col_md, col_pdf = st.columns([2, 1, 1, 1])
            with col_title:
                actual = len(doc.split())
                st.subheader(f"Consolidated Document  ({actual:,} words)")
            with col_copy:
                copy_button(doc, "Copy")
            with col_md:
                st.download_button(
                    "Download .md",
                    data=doc,
                    file_name=filename_for(company_name, "synthnotes", "md"),
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_final_md",
                )
            with col_pdf:
                # PDF generation is best-effort; if deps are missing it degrades gracefully
                pdf_bytes = markdown_to_pdf_bytes(doc)
                if pdf_bytes:
                    st.download_button(
                        "Download .pdf",
                        data=pdf_bytes,
                        file_name=filename_for(company_name, "synthnotes", "pdf"),
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_final_pdf",
                    )
                else:
                    st.caption("PDF unavailable — install `markdown` and `xhtml2pdf` (see requirements.txt)")

            with st.expander(f"Source documents used ({len(sources)})", expanded=False):
                for s in sources:
                    st.text(f"  • {s}")

            # Combined extracted text — portable artefact you can paste into other tools
            if combined_extract_text:
                with st.expander("📋 Combined extracted text (portable — usable in other tools)", expanded=False):
                    st.caption(
                        "The combined extracted text from Stage 0 — what survived extraction from "
                        "all ARs and transcripts, with `==== From: filename ====` headers between "
                        "sources. Download to paste into MultiDoc, Gemini AI Studio, or any other "
                        "tool. This is the input that fed Map+Reduce in this run."
                    )
                    ext_words = len(combined_extract_text.split())
                    st.caption(f"~{ext_words:,} words across all extracted sources")
                    st.download_button(
                        "Download combined extract (.txt)",
                        data=combined_extract_text,
                        file_name=filename_for(company_name, "extract", "txt"),
                        mime="text/plain",
                        use_container_width=True,
                        key="dl_combined_extract",
                    )

            # Interim notes download — lets the user re-run synthesis later without re-paying for Map
            if interim_text:
                with st.expander("💾 Save interim notes (for re-running synthesis later)", expanded=False):
                    st.caption(
                        "The interim file holds the Map-stage output (per-section notes). "
                        "Download it now — next time, switch **Input mode** to "
                        "*Saved interim notes* and upload this file to skip the expensive Map "
                        "stage and re-run synthesis with a different prompt or length."
                    )
                    interim_section_count = max(0, interim_text.count(INTERIM_SECTION_SEPARATOR) - 1)
                    interim_word_count = len(interim_text.split())
                    st.caption(
                        f"~{interim_word_count:,} words across {interim_section_count} section(s) of intermediate notes"
                    )
                    st.download_button(
                        "Download interim notes (.txt)",
                        data=interim_text,
                        file_name=filename_for(company_name, "interim", "txt"),
                        mime="text/plain",
                        use_container_width=True,
                        key="dl_interim",
                    )

            st.markdown(doc)

    # ── Cost panel ─────────────────────────────────────────────────────────────
    render_usage_panel()


if __name__ == "__main__":
    main()
