"""
SynthNotes Combined — Extract → Notes → Synthesise, in one guided app.

This app merges the two apps you actually used together in your transmission
research, into a single pipeline with three stages:

    Stage 1 — EXTRACT    (from SynthNotes MultiDocLean)
        Strip boilerplate out of annual reports & investor presentations,
        keeping only the analyst-relevant narrative.

    Stage 2 — NOTES      (Map stage, from SynthNotes MultiDoc)
        Chunk the extracted content + transcripts and turn each chunk into
        structured notes following your instructions.

    Stage 3 — SYNTHESISE (Reduce stage, from SynthNotes MultiDoc)
        Weave all the notes into one coherent equity-research note.

You can run the whole thing in one click, OR start from any stage if you
already have the output from the previous one. Every stage's output is offered
as a download, so nothing is lost between steps.

Code for the extraction stage is lifted from SynthNotes-MultiDocLean; the map,
synthesis, export and cost-tracking engine is lifted from SynthNotes-MultiDoc.
Only the UI and the stage-wiring are new.
"""

import streamlit as st
from google import genai
from google.genai import types
import os, io, re, time, json, base64, html as html_module
from datetime import datetime
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MAX_OUTPUT_TOKENS  = 65536
MAX_FILES          = 500
MAX_FILE_SIZE_MB   = 10
PARALLEL_WORKERS   = 3

WORDS_PER_REDUCE_BATCH = 500_000
MAX_REDUCE_DEPTH       = 3

# ── Extraction (Stage 1) parameters — from MultiDocLean ────────────────────────
EXTRACTION_CHUNK_SIZE_AR         = 80_000
EXTRACTION_CHUNK_OVERLAP_AR      = 2_000
EXTRACTION_CHUNK_SIZE_TRANSCRIPT = 30_000
EXTRACTION_CHUNK_OVERLAP_TRANSCRIPT = 1_000
EXTRACTION_PARALLEL_WORKERS = 6
EXTRACTION_SKIP_MARKER = "[chunk skipped — no analyst-relevant content]"

# Live models only, ordered best-quality first — kept in step with SynthNotes-Pro.
# Removed Aug 2026: gemini-2.0-flash-lite and gemini-1.5-flash (shut down by Google,
# calls would 404), gemini-3-flash-preview (superseded by the stable gemini-3.7-flash),
# and gemini-3.5-flash ($1.50/$9.00 per 1M for lower quality than 3.7 Flash's
# $0.75/$3.75 — strictly dominated).
MODELS = {
    "Gemini 3.1 Pro (Max quality, preview)": "gemini-3.1-pro-preview",
    "Gemini 3.7 Flash (Best all-round)":     "gemini-3.7-flash",
    "Gemini 2.5 Pro (Stable, high quality)": "gemini-2.5-pro",
    "Gemini 2.5 Flash (Fast)":               "gemini-2.5-flash",
    "Gemini 3.5 Flash Lite (Cheap)":         "gemini-3.5-flash-lite",
    "Gemini 2.5 Flash Lite (Cheapest)":      "gemini-2.5-flash-lite",
}

# Defaults by stage. Stage 1 decides what the later stages ever get to see, so it is a
# content-dropping step, not a mechanical one — it gets a real model for the same reason
# Pro's refinement stage was moved off Flash Lite. Stage 3 writes the final note and
# gets the max-quality model.
DEFAULT_EXTRACT_MODEL = "Gemini 3.7 Flash (Best all-round)"
DEFAULT_MAP_MODEL     = "Gemini 3.7 Flash (Best all-round)"
DEFAULT_REDUCE_MODEL  = "Gemini 3.1 Pro (Max quality, preview)"


def default_model_index(display_name: str) -> int:
    """Selectbox index for a default, looked up by name so reordering MODELS is safe."""
    keys = list(MODELS.keys())
    return keys.index(display_name) if display_name in keys else 0

# Approximate pricing per 1M tokens (USD), <200K context. Verified Aug 2026.
MODEL_PRICING = {
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.7-flash":       (0.75,  3.75),  # promo rate to 31 Dec 2026, then 1.50/7.50
    "gemini-2.5-pro":         (1.25, 10.00),
    "gemini-2.5-flash":       (0.30,  2.50),
    "gemini-3.5-flash-lite":  (0.30,  2.50),
    "gemini-2.5-flash-lite":  (0.10,  0.40),
}

LENGTH_PRESETS = {
    "Short (~2000 words)":    2000,
    "Standard (~4000 words)": 4000,
    "Long (~6000 words)":     6000,
    "Maximum (~8000 words)":  8000,
    "Custom":                 None,
}

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

INTERIM_FILE_HEADER = "==== SynthNotes MultiDoc — Interim Notes ===="
INTERIM_SECTION_SEPARATOR = "==== SECTION ===="


def resolve_api_key() -> str:
    """Find the Gemini key from Streamlit secrets, env, or a pasted value."""
    key = ""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")  # Streamlit Cloud
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        key = st.session_state.get("_pasted_api_key", "")
    return key


# google-genai client. The legacy google-generativeai SDK was retired on 30 Nov 2025
# and cannot reach the Gemini 3.x models. The key arrives at runtime (secrets, env, or
# pasted into the sidebar), so the client is built here rather than at import.
_client = None


def configure_genai(key: str) -> None:
    global _client
    if key:
        try:
            _client = genai.Client(api_key=key)
        except Exception:
            _client = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL HELPERS  (from MultiDoc)
# ══════════════════════════════════════════════════════════════════════════════

class _StreamHandle:
    """Iterable wrapper around a google-genai streaming response.

    The raw stream is a generator, so the tracking attributes generate_with_retry
    attaches cannot be set on it directly the way the old SDK allowed. This holds them
    and captures usage_metadata as chunks go past, so stream_and_collect can still
    record cost once iteration finishes."""

    def __init__(self, stream, model_id: str, stage: str = ""):
        self._stream = stream
        self._tracked_model_id = model_id
        self._tracked_stage = stage
        self.usage_metadata = None

    def __iter__(self):
        for chunk in self._stream:
            usage = getattr(chunk, "usage_metadata", None)
            if usage is not None:
                self.usage_metadata = usage
            yield chunk


class _ModelHandle:
    """Adapter exposing the old `.generate_content(...)` signature on top of
    google-genai, so every existing call site keeps working unchanged."""

    def __init__(self, model_id: str):
        self.model_name = model_id

    def generate_content(self, prompt, stream: bool = False, generation_config=None):
        if _client is None:
            raise RuntimeError("No Gemini API key configured - cannot call the API.")
        contents = prompt if isinstance(prompt, list) else [prompt]
        config = types.GenerateContentConfig(**generation_config) if generation_config else None
        if stream:
            return _StreamHandle(_client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config), self.model_name)
        return _client.models.generate_content(
            model=self.model_name, contents=contents, config=config)


def get_model(display_name: str) -> "_ModelHandle":
    """Cache and return a model handle for the given UI label."""
    cache = st.session_state.setdefault("_model_cache", {})
    model_id = MODELS.get(display_name, "gemini-3.7-flash")
    if model_id not in cache:
        cache[model_id] = _ModelHandle(model_id)
    return cache[model_id]


def _record_usage(model_id: str, response, stage: str = "") -> None:
    """Append a usage entry to st.session_state['usage_log']. Silent on failure."""
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
        piece = getattr(chunk, "text", None)
        if piece:
            full_text += piece
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Extraction prompts (Stage 1) — from MultiDocLean ───────────────────────────
DEFAULT_EXTRACTION_AR_PROMPT = """You are extracting ONLY the analyst-relevant narrative portions from a chunk of an Indian annual report.

KEEP these sections — output them VERBATIM (do not paraphrase, summarise, or alter):
- Chairman's Letter / Managing Director's Letter / CEO's Letter
- Management Discussion and Analysis (MD&A) — all sub-sections
- Business Performance / Business Review / Operations Review
- Segment / Geography / Business Vertical Performance (narrative AND tabular form)
- Financial Highlights / Performance at a Glance / 5-year or 10-year summary tables
- The summary Profit & Loss and Balance Sheet, and for lenders the NII / NIM / PPOP /
  credit cost / GNPA / NNPA / RoA / RoE / AUM and loan-book-mix disclosures
- Segment information tables (segment revenue, segment results, segment assets)
- Key operating metrics tables (volumes, capacity, realisations, branch/store counts)
- Strategy / Strategic Direction / Future Outlook / Five-Year View
- Director's Report — narrative business commentary only (skip the routine governance text)
- Business Responsibility Report — only if it contains substantive business content

SKIP these sections entirely (output nothing for these):
- Notice of AGM
- Notes to Accounts (the detailed numbered notes), EXCEPT any note carrying segment
  information or a business-mix breakdown
- Significant Accounting Policies
- Schedules and annexures that are pure statutory detail
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
   The same applies to figures: a downstream stage builds a key-financials section and a
   quantitative reference from whatever survives here, so preserve summary financial
   tables, segment splits and operating metrics VERBATIM, with their units, their period
   labels (FY22, Q3FY24) and their footnotes. Never round, never convert, never redraw a
   table into prose.
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

# ── Map prompt (Stage 2) — from MultiDoc ───────────────────────────────────────
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

# ── Default synthesis prompt (Stage 3) — from MultiDoc ─────────────────────────
DEFAULT_USER_PROMPT = """ROLE
You are an experienced equity analyst explaining ONE company to a sharp colleague who
knows markets but not this company. Write the way you'd actually talk it through: in plain
English, landing the things that matter and leaving out the noise. Your reader is an
investment analyst, not an engineer — explain the business in business terms, never in
procurement or product-brochure language.

SOURCES
All documents in this project: ~5 years of earnings-call transcripts, the last few
investor presentations, and recent annual reports. Read them together. The transcripts —
especially the analyst Q&A — are where management explains the WHY, so they carry the
story. Annual reports supply the hard facts: the financial-highlights tables, segment and
loan-book splits, capacity and balance-sheet data that Sections 1 and 9 are built from.
Never reproduce annual-report narrative or outlook prose, which is generic boilerplate.

TWO ABSOLUTE RULES
1. FACT before INTERPRETATION, always visually separate. State what management said or what
   the numbers show as plain narrative. Then, where you add your own judgement, start a new
   paragraph beginning "My read:" — the ONLY place your inference may appear. A reader must
   be able to skip every "My read:" and still have a complete, accurate account of what the
   company and management actually said.
2. No outside data, no invented quotes, no estimated figures. Not in the documents = say so
   explicitly ("not disclosed in the documents"). Never fill a gap in a table with a guess,
   an interpolation, or a number carried over from another year. Quotes <=25 words, exact;
   otherwise paraphrase.

HOW TO WRITE — FORMAT
The old version of this note was unbroken prose and was hard to read. Mix the two forms:
- **Short paragraphs** (3-6 sentences) carry an ARGUMENT — anything with a "because" in it,
  any mechanism, any judgement. Reasoning belongs in prose.
- **Bullets** carry PARALLEL ITEMS — a set of drivers, segments, risks, guidance points,
  or figures that sit at the same level. Lists belong in bullets.
- The test: if the items need connecting words to make sense together, write a paragraph.
  If they'd read as "X, and also Y, and also Z", write bullets.
- Never run more than two consecutive paragraphs without a bold sub-heading or a bulleted
  list breaking them up. A wall of prose is a defect.
- **Bold sub-headings** (a few words) every 2-4 paragraphs, labelling what comes next.
- **Tables** belong in Sections 1 and 9 only. Do not put tables in the narrative sections.
- Numbers: round in the narrative (Sections 2-7) — "margins went from roughly 11% to 14%".
  Exact, as-reported, with units and period labels in Sections 1 and 9.

LENGTH AND COMPLETENESS
Sections 2-7 are the body of the note and must be written at full depth — the same depth
as if Sections 1, 8 and 9 did not exist. The new sections are ADDITIONS, not a budget to
be found by trimming the story. Nothing that belongs in the narrative may be dropped,
shortened, or deferred to the summary on the grounds that it appears there too. Section 8
repeats; it does not replace.

═══════════════════════════════════════════════════════════════════════════════
WRITE THE NOTE IN THIS ORDER
═══════════════════════════════════════════════════════════════════════════════

## 1. THE BUSINESS AT A GLANCE
The basics, before any story. A reader who knows nothing should finish this section
knowing what the company sells, to whom, and what its numbers look like. No narrative
arc here, no interpretation beyond one closing "My read:".

**What this business does** — 4-6 bullets, one line each: what it sells, who buys it,
how it makes money, where it operates, how big it is. Plain language.

**Revenue split** (or **Loan book split** for a lender) — a markdown table of the mix by
segment / product / geography / customer type, with the split for the most recent period
and, where the documents allow, the same split 3-5 years earlier so the shift is visible.
State the period labels and the unit in the header. Add one line under the table naming
what changed in the mix and by how much.

**Key financials** — a markdown table, years as columns, most recent last. Pick the metric
set that fits the business:
- **Lender (bank / NBFC / HFC / MFI):** NII, NIM (%), PPOP, Credit Cost (%), GNPA (%),
  NNPA (%), PAT, RoA (%), RoE (%), and AUM or loan book with its growth.
- **Everything else:** Revenue, EBITDA, EBITDA margin (%), PAT, PAT margin (%), and where
  the documents give them: order book, capacity, volumes, realisations, RoCE / RoE.
- **Insurer / AMC / exchange / other financial:** use the metric set the company itself
  reports (APE, VNB margin, AUM, yields, take-rate) rather than forcing either template.
Report only what the documents contain. Where a cell is not disclosed, write "n/a" — never
estimate. Below the table, 3-5 bullets naming the most striking movements in it.

**Charts** — for the two or three most important series in this section, emit a chart block
in exactly this format, on its own lines:

```synthnotes-chart
{"title": "Revenue and EBITDA margin", "type": "line", "x": ["FY21","FY22","FY23","FY24","FY25"], "unit": "Rs cr", "series": [{"name": "Revenue", "values": [1200, 1450, 1810, 2200, 2610]}]}
```

Rules for chart blocks: `type` is "line" or "bar". `x` is the period labels. Every series
must have exactly as many `values` as there are `x` labels — use `null` for a period the
documents do not disclose. Use only figures that appear in your table above. Emit 2-4
chart blocks, no more. If the documents do not support a clean series, emit no chart block
at all — a missing chart is fine, an invented one is not.

My read: what the shape of these numbers tells you before we get into the story.

## 2. THE STORY OF THE LAST 5 YEARS  (the heart of the note — most space)
Tell it as a narrative, not a ledger. Cover, and connect, these threads:
- GROWTH: how sales and EBITDA grew, and crucially WHERE the growth came from — which
  products/end-markets pulled it and WHY that demand appeared (a capex cycle, a policy
  push, exports, a competitor stumbling, share gains). Don't just name the driver —
  explain the mechanism behind it.
- MARGINS: what happened to margins and WHY — pricing power because the market was tight?
  richer mix? or squeezed by a specific raw material? Say which, and whether management
  framed the gains as durable or temporary.
- ORDER BOOK (or LOAN BOOK / AUM): how it grew and what it's signalling, since it leads
  sales — and whether management said anything about its QUALITY (margin, or asset
  quality), not just its size.
- CAPACITY (or BRANCH / DISTRIBUTION BUILD-OUT): did they add it, did it arrive in time,
  did demand absorb it, or did they expand into a hot market late?
Weave these into one story anchored on the few drivers that genuinely mattered. Use bold
sub-headings for each thread, prose for the mechanisms, bullets where you are listing
parallel drivers or a sequence of events.
My read: how much of this growth is structural versus a cyclical/peak moment, and which
parts I'd trust to persist.

## 3. WHY THIS COMPANY WINS  (competitive advantage)
What actually lets them win business and hold margin — technology, approvals and track
record, customer relationships, scale, being one of few who can do the hard work?
Explain it as why a customer picks them over the next supplier.
My read: whether that edge is durable or just today's tightness flattering everyone.

## 4. COMPETITION, AND HOW MANAGEMENT TALKS ABOUT IT
Who they compete with, and what management says when analysts push on new competition,
new capacity coming in, imports, or pricing pressure — do they sound relaxed or guarded,
and have they admitted any share loss or pricing slippage? Whether they engage the
question or deflect it is itself informative.
My read: whether the competitive threat is real and how honestly management is facing it.

## 5. WHAT MANAGEMENT EXPECTS NEXT  (their words, kept clearly as their words)
Pull together what management has guided or signalled on demand and sales growth, order-
book outlook, and margins — and the REASONS they give for each, since the reasoning
matters more than the number. Include their stated plans on capacity and capital
allocation. Bullets work well for the guidance points themselves; use prose for the
reasoning behind them. Keep this strictly "management says", never blended with your view.
My read: which expectations look well-supported versus optimistic, and what they're
quietly assuming.

## 6. WHAT TO REMEMBER
A short section: the handful of things a busy investor should actually carry away — the
load-bearing points that recurred and matter. Plain sentences, no grab-bag.

## 7. THE NOTE IN ONE PAGE
The whole story compressed to roughly 500-600 words — what this business is, what happened
over five years and why, what makes it win, what could break it, and what management
expects next. Written so someone who reads ONLY this page comes away with the argument
intact. Lead with a 3-4 sentence paragraph stating the story in full, then 6-10 bullets
carrying the supporting points. No new material may appear here that is not already
somewhere above.

## 8. ALL THE NUMBERS IN ONE PLACE
Every quantitative fact in the note, collected for quick reference — the reader should
never have to hunt back through the prose for a figure. Organise as markdown tables under
bold sub-headings, in this order where the material exists:
- **Financial summary** — the Section 1 table, repeated in full.
- **Segment / product / geography splits** — revenue or loan book by cut, across periods.
- **Operating metrics** — volumes, capacity, utilisation, realisations, branch or store
  counts, employee counts, order book, AUM.
- **Asset quality and returns** (lenders) — GNPA, NNPA, PCR, credit cost, restructured
  book, RoA, RoE, capital adequacy.
- **Balance sheet and cash flow** — debt, net debt, D/E, working capital, capex, OCF, FCF.
- **Guidance and targets** — every forward-looking number management has given, each with
  the period it refers to and the quarter it was said in.
- **Valuation and capital-return data** — only if the documents contain it.
Rules: every figure must carry its unit and its period label. Report as-reported, never
converted or rounded. Where a metric was disclosed in some periods and not others, show
the periods you have and "n/a" elsewhere. This section is a reference table, not prose —
no commentary, no "My read:".

TONE
Long enough to do justice to Section 2, short enough to read in one sitting. Depth on the
things that matter, silence on the things that don't. If a section is thin in the sources,
keep it short and say what's missing — never pad with generic industry talk.
"""


# ── Reduce prompts (Stage 3) — from MultiDoc ───────────────────────────────────
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

TOTAL: ~[sum of the NARRATIVE section budgets — must be close to {target_word_count}] words, plus ~1850 words of mandatory sections
CHRONOLOGY_NOTE: [one sentence about how sections are ordered, e.g. "Sections are in chronological order from Q1 2023 to Q4 2024" OR "Chronology could not be inferred; sections are organised by topic."]

### MANDATORY SECTIONS — these three are fixed and always present
The user's instructions specify an opening section and two closing sections. Reproduce
them as the first and last sections of your outline, with these headings and budgets:

- **First section**, heading `## 1. THE BUSINESS AT A GLANCE` — word budget ~600 words.
- **Second-to-last**, heading `## THE NOTE IN ONE PAGE` — word budget ~550 words.
- **Last section**, heading `## ALL THE NUMBERS IN ONE PLACE` — word budget ~700 words.

**These three budgets sit ON TOP of the target and do NOT count towards it.** They are
structural additions; they must not be funded by shrinking the narrative. The narrative
sections between them still have the full ~{target_word_count} words to share.

### RULES
1. Section count: the three mandatory sections above, PLUS typically **5–8 narrative
   sections** between them, scaled to content and length target.
2. The word budgets of the NARRATIVE sections (everything except the three mandatory ones)
   MUST sum to approximately **{target_word_count}** (±10%). Do not subtract the mandatory
   sections' budgets from that total — they are additional.
3. Each section should be coherent, self-contained, and cover distinct material (no overlap between sections).
4. Section headings should reflect the user's instructions in form and tone.
5. Do NOT write any prose body — only the outline structure.
6. The COVERAGE line for each section should be specific enough that a separate writer could write JUST that section knowing only its coverage description and the source notes.

### STRICT FORMAT REQUIREMENTS (a downstream parser depends on these)
- Each section heading MUST begin with `## ` (two hashes then a space). Not `###`, not `**bold**`, not numbered. Exactly `## `.
- Each section MUST include both lines below the heading:
  `- Coverage: …`
  `- Word budget: ~N words`  (with a number; the parser extracts the first integer)
- Do not omit either line for any section. If a section is short, still include both labels.
- Use the exact label words "Coverage" and "Word budget" (case-insensitive but spelled exactly).

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
4. Apply the user's formatting instructions within this section. In particular: mix short
   paragraphs with bullet lists — paragraphs for reasoning and mechanisms, bullets for
   parallel items — and never run more than two consecutive paragraphs without a bold
   sub-heading or a bulleted list breaking them up.
5. Tables and chart blocks belong ONLY in the sections whose assignment calls for them
   (the opening key-financials section and the closing numbers section). If your section
   is one of those, reproduce figures exactly as the notes give them, with units and
   period labels, and write "n/a" for anything not disclosed rather than estimating.
6. Begin your output with the heading line exactly: `## {section_heading}`
7. Do NOT include preamble like "This section covers…" — start with content immediately after the heading.
8. Do NOT include a conclusion that summarises other sections — the final document has its own flow.
   (The one-page summary section is the single exception: summarising the rest IS its assignment.)

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


# ══════════════════════════════════════════════════════════════════════════════
# 4. STAGE 1 — EXTRACT  (from MultiDocLean)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_one_chunk(chunk_text, chunk_n, total_chunks, filename, extraction_prompt, model) -> Optional[str]:
    prompt = EXTRACTION_WRAPPER_TEMPLATE.format(
        extraction_prompt=extraction_prompt.strip(),
        filename=filename, chunk_n=chunk_n, total_chunks=total_chunks, chunk_text=chunk_text,
    )
    try:
        resp = generate_with_retry(model, prompt, stage="Extraction")
        return resp.text
    except Exception as e:
        return f"[Extraction failed for chunk {chunk_n} of {filename}: {e}]"


def extract_pass(files, extraction_prompt, chunk_size, overlap, model, status_write, source_type_label):
    """Run extraction on a list of (filename, content). Returns [(filename, extracted)]."""
    if not files:
        return []
    tasks = []
    per_file_chunk_counts = []
    for file_idx, (filename, content) in enumerate(files):
        chunks = create_chunks_with_overlap(content, chunk_size, overlap)
        per_file_chunk_counts.append(len(chunks))
        for chunk_idx, chunk_text in enumerate(chunks):
            tasks.append((file_idx, chunk_idx, chunk_text, filename, len(chunks)))

    n_total = len(tasks)
    status_write(f"  {source_type_label}: {len(files)} file(s) → {n_total} chunk(s) "
                 f"(parallel × {EXTRACTION_PARALLEL_WORKERS}).")

    results = {}
    with ThreadPoolExecutor(max_workers=EXTRACTION_PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(_extract_one_chunk, chunk_text, chunk_idx + 1, total_in_file,
                            filename, extraction_prompt, model): (file_idx, chunk_idx, filename)
            for file_idx, chunk_idx, chunk_text, filename, total_in_file in tasks
        }
        done = 0
        for fut in as_completed(futures):
            file_idx, chunk_idx, filename = futures[fut]
            results[(file_idx, chunk_idx)] = fut.result()
            done += 1
            status_write(f"    • {source_type_label}: {done}/{n_total} chunk(s) done")

    out = []
    for file_idx, (filename, content) in enumerate(files):
        n_chunks = per_file_chunk_counts[file_idx]
        kept = []
        for chunk_idx in range(n_chunks):
            output = results.get((file_idx, chunk_idx))
            if not output or not output.strip():
                continue
            stripped = output.strip()
            if stripped == EXTRACTION_SKIP_MARKER or stripped.startswith("[chunk skipped"):
                continue
            kept.append(stripped)
        if not kept:
            status_write(f"  ⚠️  {filename}: extraction returned no retained content — skipping this file")
            continue
        combined = "\n\n".join(kept)
        combined = re.sub(r"\n*\[remainder skipped[^\]]*\]\s*$", "", combined).strip()
        orig_words = len(content.split())
        kept_words = len(combined.split())
        retention = (kept_words / orig_words * 100) if orig_words else 0
        status_write(f"  ✓ {filename}: {orig_words:,} → {kept_words:,} words ({retention:.0f}% retained)")
        out.append((filename, combined))
    return out


def serialize_combined_extract(extracted_ar, extracted_transcripts) -> str:
    lines = ["==== SynthNotes MultiDocLean — Combined Extracted Content ===="]
    total_kept = sum(len(c.split()) for _, c in (extracted_ar + extracted_transcripts))
    lines.append(f"{len(extracted_ar)} AR file(s) + {len(extracted_transcripts)} transcript file(s)  "
                 f"|  total {total_kept:,} words after extraction")
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. STAGE 2 — NOTES / MAP  (from MultiDoc)
# ══════════════════════════════════════════════════════════════════════════════

def process_chunk(chunk_text, chunk_n, total_chunks, filename, file_position, total_files, user_prompt, model) -> Optional[str]:
    prompt = MAP_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt.strip(), filename=filename, chunk_n=chunk_n,
        total_chunks=total_chunks, file_position=file_position, total_files=total_files,
        chunk_text=chunk_text,
    )
    try:
        resp = generate_with_retry(model, prompt, stage="Map (per chunk)")
        return resp.text
    except Exception as e:
        return f"_[Section {chunk_n} of {filename} failed: {e}]_"


def run_map_stage(files, user_prompt, target_word_count, map_model, status_write):
    """Chunk each file and run the map prompt on every chunk, parallelised.
    Returns (notes_list, filenames)."""
    chunk_size, overlap = compute_chunk_params(target_word_count)
    tasks = []
    per_file_counts = []
    for fi, (fname, content) in enumerate(files):
        chunks = create_chunks_with_overlap(content, chunk_size, overlap)
        per_file_counts.append(len(chunks))
        for ci, ch in enumerate(chunks):
            tasks.append((fi, ci, ch, fname, len(chunks)))

    total = len(tasks)
    status_write(f"  Notes/Map: {len(files)} file(s) → {total} chunk(s) "
                 f"(chunk {chunk_size:,}/{overlap:,} words, parallel × {PARALLEL_WORKERS}).")

    results = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(process_chunk, ch, ci + 1, tot, fname, fi + 1, len(files),
                            user_prompt, map_model): (fi, ci)
            for (fi, ci, ch, fname, tot) in tasks
        }
        done = 0
        for fut in as_completed(futures):
            fi, ci = futures[fut]
            results[(fi, ci)] = fut.result()
            done += 1
            status_write(f"    • Notes/Map: {done}/{total} chunk(s) done")

    notes = []
    for fi, (fname, content) in enumerate(files):
        for ci in range(per_file_counts[fi]):
            r = results.get((fi, ci))
            if r and r.strip():
                notes.append(r.strip())
    filenames = [f for f, _ in files]
    return notes, filenames


def serialize_interim(notes_list, filenames) -> str:
    header_lines = [INTERIM_FILE_HEADER, f"Generated from {len(filenames)} source file(s)", "Source filenames:"]
    for f in filenames:
        header_lines.append(f"- {f}")
    header_lines.append("")
    header = "\n".join(header_lines)
    body = ("\n\n" + INTERIM_SECTION_SEPARATOR + "\n").join(notes_list)
    return header + "\n" + INTERIM_SECTION_SEPARATOR + "\n" + body + "\n"


def parse_interim(text) -> Tuple[List[str], List[str]]:
    filenames = []
    header_match = re.search(r"Source filenames:\s*\n((?:-\s.+\n?)+)", text)
    if header_match:
        for line in header_match.group(1).splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                filenames.append(stripped[2:].strip())
    if INTERIM_SECTION_SEPARATOR in text:
        parts = text.split(INTERIM_SECTION_SEPARATOR)
        notes = [p.strip() for p in parts[1:] if p.strip()]
    else:
        notes = [text.strip()]
    return notes, filenames or ["(filenames not recorded in interim file)"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. STAGE 3 — SYNTHESISE / REDUCE  (from MultiDoc)
# ══════════════════════════════════════════════════════════════════════════════

def _final_reduce(notes_list, filenames, user_prompt, target_word_count, reduce_model, status_write) -> str:
    combined = "\n\n".join(notes_list)
    prompt = REDUCE_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt.strip(), target_word_count=target_word_count,
        num_files=len(filenames), filename_list="\n".join(f"- {f}" for f in filenames),
        combined_notes=combined,
    )
    ph = st.empty()
    resp = generate_with_retry(reduce_model, prompt, stream=True,
                               generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS},
                               stage="Final reduce (synthesis)")
    text, _ = stream_and_collect(resp, ph)
    return text


def _generate_outline(combined_notes, filenames, user_prompt, target_word_count, model) -> str:
    prompt = OUTLINE_PROMPT.format(
        user_prompt=user_prompt.strip(), target_word_count=target_word_count,
        num_files=len(filenames), filename_list="\n".join(f"- {f}" for f in filenames),
        combined_notes=combined_notes,
    )
    resp = generate_with_retry(model, prompt, stage="Plan (outline)")
    return resp.text


_HEADING_PATTERNS = (
    re.compile(r"^#{2,4}\s+(\S.*?)\s*:?\s*$"),
    re.compile(r"^\*\*([^*]+?)\*\*\s*:?\s*$"),
    re.compile(r"^\d+\.\s+(\S.*?)\s*:?\s*$"),
)
_COVERAGE_RE = re.compile(r"^[-*]?\s*(?:Coverage|Covers|Description|Content|What\s+it\s+covers)\s*:\s*(.+)$", re.IGNORECASE)
_BUDGET_RE = re.compile(r"^[-*]?\s*(?:Word\s*budget|Words|Length|Target\s*words?|Budget|Approx\.?\s*words)\s*:\s*(.+)$", re.IGNORECASE)


# The three structural sections carry fixed budgets that sit ON TOP of the user's target
# word count. The target governs the NARRATIVE only, so adding these sections can never
# shrink the story — which was the whole point of adding them.
MANDATORY_SECTION_BUDGETS = (
    (("BUSINESS AT A GLANCE", "AT A GLANCE"), 600),
    (("NOTE IN ONE PAGE", "IN ONE PAGE"), 550),
    (("NUMBERS IN ONE PLACE", "ALL THE NUMBERS"), 700),
)
MANDATORY_BUDGET_TOTAL = sum(budget for _keys, budget in MANDATORY_SECTION_BUDGETS)


def _mandatory_budget_for(heading: str) -> Optional[int]:
    """Fixed budget if this heading is one of the structural sections, else None."""
    upper = (heading or "").upper()
    for keys, budget in MANDATORY_SECTION_BUDGETS:
        if any(key in upper for key in keys):
            return budget
    return None


def _enforce_section_budgets(sections: List[dict], target_word_count: int) -> List[dict]:
    """Pin the structural sections and guarantee the narrative keeps the full target.

    Runs after outline parsing, so it corrects the planner rather than trusting it:
    structural sections get their fixed budgets, and if the planner funded them by
    trimming the narrative, the narrative budgets are scaled back up to the target."""
    narrative = []
    for section in sections:
        fixed = _mandatory_budget_for(section.get("heading", ""))
        section["mandatory"] = fixed is not None
        if fixed is not None:
            section["budget"] = fixed
        else:
            narrative.append(section)
    if not narrative:
        return sections

    # Narrative sections the planner left without a parseable budget share what is left.
    missing = [x for x in narrative if x["budget"] <= 0]
    if missing:
        assigned = sum(x["budget"] for x in narrative if x["budget"] > 0)
        remaining = max(target_word_count - assigned, 0)
        per_missing = remaining // len(missing) if remaining else target_word_count // len(narrative)
        for x in missing:
            x["budget"] = max(per_missing, 150)

    # If the planner squeezed the narrative to fit the structural sections inside the
    # target, scale it back out. Only ever scales UP - an over-budget outline is the
    # planner deciding the material warrants it, and is left alone.
    narrative_total = sum(x["budget"] for x in narrative)
    if 0 < narrative_total < target_word_count * 0.9:
        factor = target_word_count / float(narrative_total)
        for x in narrative:
            x["budget"] = int(round(x["budget"] * factor))
    return sections


def _parse_outline(outline_text) -> List[dict]:
    sections = []
    current = None
    _META_PREFIXES = ("TOTAL", "CHRONOLOGY_NOTE", "DOCUMENT TITLE")

    def flush():
        nonlocal current
        if current and current.get("heading"):
            sections.append(current)
        current = None

    for raw_line in outline_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        heading_text = None
        for pat in _HEADING_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            candidate = m.group(1).strip()
            if candidate.upper().startswith(_META_PREFIXES):
                continue
            heading_text = candidate
            break
        if heading_text:
            flush()
            current = {"heading": heading_text, "coverage": "", "budget": 0}
            continue
        if current is None:
            continue
        cov_m = _COVERAGE_RE.match(line)
        if cov_m:
            current["coverage"] = cov_m.group(1).strip()
            continue
        bud_m = _BUDGET_RE.match(line)
        if bud_m:
            num = re.search(r"(\d+)", bud_m.group(1))
            if num:
                current["budget"] = int(num.group(1))
            continue
    flush()
    return [s for s in sections if s["heading"]]


def _extract_outline_metadata(outline_text) -> Tuple[Optional[str], Optional[str]]:
    title = None
    title_match = re.search(r"^#\s+(.+?)$", outline_text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
        if title.startswith("#"):
            title = None
    chrono = None
    chrono_match = re.search(r"CHRONOLOGY_NOTE\s*:\s*(.+?)$", outline_text, re.MULTILINE)
    if chrono_match:
        chrono = chrono_match.group(1).strip()
    return title, chrono


def _write_section(section, section_n, total_sections, outline_text, combined_notes, user_prompt, model) -> str:
    prompt = SECTION_PROMPT.format(
        user_prompt=user_prompt.strip(), section_heading=section["heading"],
        section_coverage=section.get("coverage", "(no specific coverage stated)"),
        section_word_budget=section.get("budget", 500), section_n=section_n,
        total_sections=total_sections, outline_text=outline_text, combined_notes=combined_notes,
    )
    resp = generate_with_retry(model, prompt, stage="Write (section)")
    return resp.text


def plan_then_write_final(notes_list, filenames, user_prompt, target_word_count, model, status_write) -> str:
    combined_notes = "\n\n".join(notes_list)
    status_write(f"📋 PLAN stage — generating outline for ~{target_word_count}-word document…")
    outline_text = _generate_outline(combined_notes, filenames, user_prompt, target_word_count, model)
    sections = _parse_outline(outline_text)
    if not sections:
        snippet = outline_text.strip().replace("\n", " ⏎ ")[:500]
        if len(outline_text.strip()) > 500:
            snippet += "…"
        status_write(f"⚠️  Outline parsing found 0 sections — falling back to single-pass synthesis. "
                     f"Model output started with: \"{snippet}\"")
        # This path writes the whole document in one call, structural sections included,
        # so it needs the combined budget - otherwise the narrative gets squeezed to fit.
        return _final_reduce(notes_list, filenames, user_prompt,
                             target_word_count + MANDATORY_BUDGET_TOTAL, model, status_write)

    sections = _enforce_section_budgets(sections, target_word_count)

    title, chronology_note = _extract_outline_metadata(outline_text)
    narrative_budget = sum(s["budget"] for s in sections if not s.get("mandatory"))
    structural_budget = sum(s["budget"] for s in sections if s.get("mandatory"))
    total_budget = narrative_budget + structural_budget
    status_write(
        f"📋 Outline: **{len(sections)} sections** — narrative {narrative_budget:,} words "
        f"(target {target_word_count:,}) + key-financials/summary/numbers sections "
        f"{structural_budget:,} words = {total_budget:,} total"
        + (f", title: '{title}'" if title else "")
    )

    status_write(f"✏️  WRITE stage — generating {len(sections)} sections in parallel…")
    results = [None] * len(sections)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(_write_section, section, i + 1, len(sections),
                            outline_text, combined_notes, user_prompt, model): i
            for i, section in enumerate(sections)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            wc = len(results[i].split()) if results[i] else 0
            s = sections[i]
            status_write(f"  • Section {done}/{len(sections)}: '{s['heading']}' "
                         f"({wc:,} words, budget ~{s['budget']:,})")

    body_parts = [r.strip() for r in results if r and r.strip()]
    document_parts = []
    if title:
        document_parts.append(f"# {title}")
    if chronology_note:
        document_parts.append(f"*{chronology_note}*")
    document_parts.extend(body_parts)
    stitched = "\n\n".join(document_parts)
    actual_words = len(stitched.split())
    status_write(f"✓ Synthesis complete — **{actual_words:,} words** "
                 f"({actual_words / target_word_count * 100:.0f}% of {target_word_count:,}-word target)")
    return stitched


def _intermediate_reduce(notes_batch, user_prompt, target_words, reduce_model, depth) -> str:
    combined = "\n\n".join(notes_batch)
    prompt = INTERMEDIATE_REDUCE_PROMPT.format(
        user_prompt=user_prompt.strip(), target_word_count=target_words, combined_notes=combined,
    )
    resp = generate_with_retry(reduce_model, prompt, stage=f"Intermediate reduce (depth {depth})")
    return resp.text


def _group_into_batches(notes_list, max_words_per_batch) -> List[List[str]]:
    batches, current, current_words = [], [], 0
    for note in notes_list:
        wc = len(note.split())
        if current_words + wc > max_words_per_batch and current:
            batches.append(current)
            current, current_words = [note], wc
        else:
            current.append(note)
            current_words += wc
    if current:
        batches.append(current)
    return batches


def hierarchical_reduce(notes_list, filenames, user_prompt, target_word_count, reduce_model, status_write, depth=0) -> str:
    total_words = sum(len(n.split()) for n in notes_list)
    fits_in_one_call = total_words <= WORDS_PER_REDUCE_BATCH or len(notes_list) <= 1
    if fits_in_one_call:
        if depth == 0:
            status_write(f"Notes fit in one synthesis pass (~{total_words:,} words). Starting plan-then-write…")
        else:
            status_write(f"[Depth {depth}] Notes fit in one pass (~{total_words:,} words). Starting plan-then-write…")
        return plan_then_write_final(notes_list, filenames, user_prompt, target_word_count, reduce_model, status_write)
    if depth >= MAX_REDUCE_DEPTH:
        raise ValueError(f"Input remains too large after {MAX_REDUCE_DEPTH} levels of hierarchical reduce "
                         f"(still {total_words:,} words). Try a smaller target or fewer files.")
    batches = _group_into_batches(notes_list, WORDS_PER_REDUCE_BATCH)
    status_write(f"[Depth {depth}] {total_words:,} words exceeds {WORDS_PER_REDUCE_BATCH:,}-word budget — "
                 f"splitting into {len(batches)} batch(es) and compressing each…")
    intermediate_target = 5000
    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(_intermediate_reduce, batch, user_prompt, intermediate_target, reduce_model, depth): i
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
    return hierarchical_reduce(intermediates, filenames, user_prompt, target_word_count, reduce_model, status_write, depth + 1)


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPORT + COST  (from MultiDoc)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
# The synthesis prompt asks for time-series charts as fenced ```synthnotes-chart blocks
# holding JSON. Everything here is best-effort and additive: a block that will not parse,
# or that fails validation, is dropped without comment. The figures are also present in
# the markdown table the chart was built from, so a missing chart loses no information.

CHART_BLOCK_RE = re.compile(r"```synthnotes-chart\s*\n(.*?)\n?```", re.DOTALL)
CHART_PLACEHOLDER = "@@SYNTHNOTES_CHART_%d@@"
CHART_PLACEHOLDER_RE = re.compile(r"@@SYNTHNOTES_CHART_(\d+)@@")


def _valid_chart_spec(spec) -> bool:
    """A spec is usable only if every series lines up with the x labels."""
    if not isinstance(spec, dict):
        return False
    x = spec.get("x")
    series = spec.get("series")
    if not isinstance(x, list) or not x or not isinstance(series, list) or not series:
        return False
    for entry in series:
        if not isinstance(entry, dict):
            return False
        values = entry.get("values")
        if not isinstance(values, list) or len(values) != len(x):
            return False
        for v in values:
            if v is not None and not isinstance(v, (int, float)):
                return False
    return True


def parse_chart_blocks(md_text: str):
    """Split chart blocks out of the markdown.

    Returns (markdown_with_placeholders, [spec, ...]). Invalid blocks are removed
    entirely rather than left in the document as raw JSON."""
    specs = []

    def _swap(match):
        try:
            spec = json.loads(match.group(1).strip())
        except Exception:
            return ""
        if not _valid_chart_spec(spec):
            return ""
        specs.append(spec)
        return "\n" + (CHART_PLACEHOLDER % (len(specs) - 1)) + "\n"

    return CHART_BLOCK_RE.sub(_swap, md_text), specs


def markdown_for_download(md_text: str) -> str:
    """Plain-text-friendly markdown: chart blocks become a one-line caption.

    Nothing is lost — the prompt requires chart values to come from the table printed
    immediately above the block."""
    def _caption(match):
        try:
            spec = json.loads(match.group(1).strip())
        except Exception:
            return ""
        # Same validation as parse_chart_blocks, so the text version never advertises a
        # chart that the rendered versions dropped.
        if not _valid_chart_spec(spec):
            return ""
        return "*Chart: %s*" % spec.get("title", "chart")

    return CHART_BLOCK_RE.sub(_caption, md_text)


def render_chart_png(spec) -> Optional[bytes]:
    """Render one chart spec to PNG bytes. Returns None if matplotlib is unavailable
    or anything at all goes wrong — callers treat charts as optional."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        x = [str(label) for label in spec["x"]]
        series = spec["series"]
        kind = str(spec.get("type", "line")).lower()
        positions = list(range(len(x)))

        fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
        if kind == "bar":
            width = 0.8 / max(len(series), 1)
            for i, entry in enumerate(series):
                values = [float("nan") if v is None else v for v in entry["values"]]
                offsets = [p - 0.4 + width * (i + 0.5) for p in positions]
                ax.bar(offsets, values, width=width, label=str(entry.get("name", "series")))
        else:
            for entry in series:
                values = [float("nan") if v is None else v for v in entry["values"]]
                ax.plot(positions, values, marker="o", linewidth=2,
                        label=str(entry.get("name", "series")))
        ax.set_xticks(positions)
        ax.set_xticklabels(x)
        if spec.get("title"):
            ax.set_title(str(spec["title"]), fontsize=11, fontweight="bold")
        if spec.get("unit"):
            ax.set_ylabel(str(spec["unit"]), fontsize=9)
        if len(series) > 1:
            ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def render_markdown_with_charts(md_text: str) -> None:
    """Render the note in Streamlit, drawing any chart blocks inline where they sit."""
    body, specs = parse_chart_blocks(md_text)
    for part in CHART_PLACEHOLDER_RE.split(body):
        if part.isdigit() and int(part) < len(specs):
            png = render_chart_png(specs[int(part)])
            if png:
                st.image(png, use_container_width=True)
        elif part.strip():
            st.markdown(part)


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-DOWNLOAD  (ported from SynthNotes-Pro)
# ══════════════════════════════════════════════════════════════════════════════

def auto_download_files(files) -> None:
    """Trigger browser downloads for (filename, content, mime) tuples.

    Content may be str or bytes. Downloads are staggered by 600ms so browsers do not
    block them. The first multi-file download per site prompts the user for permission;
    it is silent after that."""
    if not files:
        return
    blocks = []
    for i, (filename, content, mime) in enumerate(files):
        if isinstance(content, (bytes, bytearray)):
            payload = json.dumps(base64.b64encode(bytes(content)).decode("ascii"))
            make_blob = (
                "  var bin = atob(%s);\n"
                "  var arr = new Uint8Array(bin.length);\n"
                "  for (var j = 0; j < bin.length; j++) { arr[j] = bin.charCodeAt(j); }\n"
                "  var blob = new Blob([arr], {type: %s});\n" % (payload, json.dumps(mime))
            )
        else:
            make_blob = "  var blob = new Blob([%s], {type: %s});\n" % (
                json.dumps(content), json.dumps(mime))
        blocks.append(
            "setTimeout(function() {\n"
            + make_blob
            + "  var url = URL.createObjectURL(blob);\n"
            "  var a = document.createElement('a');\n"
            "  a.href = url;\n"
            "  a.download = %s;\n" % json.dumps(filename)
            + "  document.body.appendChild(a);\n"
            "  a.click();\n"
            "  setTimeout(function() { URL.revokeObjectURL(url); document.body.removeChild(a); }, 200);\n"
            "}, %d);" % (i * 600)
        )
    components.html("<script>" + "\n".join(blocks) + "</script>", height=0)


def _consume_pending_auto_download() -> None:
    """Fire any downloads staged by the last successful run, exactly once."""
    pending = st.session_state.pop("pending_auto_download", None)
    if pending:
        auto_download_files(pending)
        st.success(
            "✓ Auto-downloaded to your downloads folder: "
            + " · ".join("`%s`" % f[0] for f in pending)
            + "  *(the first multi-file download per site may ask your browser for permission)*"
        )


def markdown_to_pdf_bytes(md_text, title="SynthNotes Combined Output") -> Optional[bytes]:
    try:
        import markdown as md_lib
        from xhtml2pdf import pisa
        import io
    except ImportError:
        return None
    body_md, chart_specs = parse_chart_blocks(md_text)
    html_body = md_lib.markdown(body_md, extensions=["tables", "fenced_code", "nl2br"])
    # Swap each placeholder for the rendered chart. A chart that fails to render is
    # replaced with nothing, leaving the surrounding table untouched.
    for i, spec in enumerate(chart_specs):
        png = render_chart_png(spec)
        replacement = ""
        if png:
            replacement = '<img src="data:image/png;base64,%s" style="width:16cm" />' % (
                base64.b64encode(png).decode("ascii"))
        html_body = html_body.replace(CHART_PLACEHOLDER % i, replacement)
    safe_title = html_module.escape(title)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{safe_title}</title>
<style>
  @page {{ size: A4; margin: 1.5cm; }}
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4; color: #1a1a2e; }}
  h1 {{ font-size: 18pt; border-bottom: 1px solid #ccc; padding-bottom: 6px; margin-top: 0; }}
  h2 {{ font-size: 14pt; margin-top: 18px; }} h3 {{ font-size: 12pt; margin-top: 14px; }}
  h4 {{ font-size: 11pt; margin-top: 12px; }} p {{ margin: 6px 0; }}
  ul, ol {{ margin: 6px 0; padding-left: 22px; }} li {{ margin: 3px 0; }}
  blockquote {{ border-left: 3px solid #ccc; padding-left: 10px; margin-left: 0; color: #555; font-style: italic; }}
  table {{ border-collapse: collapse; margin: 10px 0; }} td, th {{ border: 1px solid #ddd; padding: 5px 8px; font-size: 10pt; }}
  th {{ background: #f0f2f6; font-weight: bold; }}
  img {{ margin: 10px 0; }}
</style></head><body>{html_body}</body></html>"""
    import io
    pdf_buf = io.BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_buf)
    if pisa_status.err:
        return None
    pdf_buf.seek(0)
    return pdf_buf.read()


def _sanitize_filename_component(s, fallback="untitled") -> str:
    if s is None:
        return fallback
    s = s.strip()
    if not s:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return safe or fallback


def filename_for(company_name, kind, ext) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    company = _sanitize_filename_component(company_name)
    kind_safe = _sanitize_filename_component(kind, fallback="output")
    return f"{date_str}_{company}_{kind_safe}.{ext.lstrip('.')}"


def compute_cost(input_tokens, output_tokens, model_id) -> float:
    pricing = MODEL_PRICING.get(model_id)
    if not pricing:
        return 0.0
    in_price, out_price = pricing
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def render_usage_panel():
    log = st.session_state.get("usage_log", [])
    if not log:
        return
    by_stage = {}
    total_in, total_out, total_cost = 0, 0, 0.0
    for entry in log:
        s = entry["stage"] or "other"
        model = entry["model"]
        cost = compute_cost(entry["input_tokens"], entry["output_tokens"], model)
        slot = by_stage.setdefault(s, {"input": 0, "output": 0, "cost": 0.0, "models": set()})
        slot["input"] += entry["input_tokens"]
        slot["output"] += entry["output_tokens"]
        slot["cost"] += cost
        slot["models"].add(model)
        total_in += entry["input_tokens"]
        total_out += entry["output_tokens"]
        total_cost += cost
    with st.expander(f"💰 Usage & cost — ~${total_cost:.4f} this session", expanded=False):
        st.caption("Approximate, based on the built-in pricing table — check your Google Cloud billing for exact figures.")
        lines = ["| Stage | Model(s) | Input tokens | Output tokens | Cost (USD) |", "|---|---|---:|---:|---:|"]
        for stage, vals in by_stage.items():
            models = ", ".join(sorted(vals["models"]))
            lines.append(f"| {stage} | `{models}` | {vals['input']:,} | {vals['output']:,} | ${vals['cost']:.4f} |")
        lines.append(f"| **Total** | — | **{total_in:,}** | **{total_out:,}** | **${total_cost:.4f}** |")
        st.markdown("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# 8. UI
# ══════════════════════════════════════════════════════════════════════════════

def read_uploaded(uploaded_file) -> Tuple[str, str]:
    """Return (filename, text) for an uploaded .txt/.md file."""
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    return uploaded_file.name, text


def main():
    st.set_page_config(page_title="SynthNotes Combined", page_icon="📝", layout="wide")

    # Leave room under the last sidebar widget so dropdown popups never open past
    # the bottom of the viewport with nowhere to scroll.
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
            padding-bottom: 14rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📝 SynthNotes Combined")
    st.caption("Extract → Notes → Synthesise. The two apps you used for transmission research, "
               "joined into one pipeline. Run it end-to-end, or jump in at any stage.")

    api_key = resolve_api_key()

    # ── Sidebar: key + models + length ────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        if not api_key:
            st.warning("No Gemini API key found.")
            pasted = st.text_input("Paste your Gemini API key", type="password",
                                   help="Get one free at aistudio.google.com/apikey. "
                                        "On Streamlit Cloud, add GEMINI_API_KEY under Settings → Secrets instead.")
            if pasted:
                st.session_state["_pasted_api_key"] = pasted
                api_key = pasted
        else:
            st.success("Gemini API key loaded ✓")

        st.divider()
        st.subheader("Models")
        st.caption("Cheap models are fine for the mechanical stages; the quality model matters only for the final write-up.")
        extract_model_name = st.selectbox("Stage 1 — Extraction model", list(MODELS.keys()),
                                          index=default_model_index(DEFAULT_EXTRACT_MODEL),
                                          help="Decides which passages the later stages ever see, so anything it "
                                               "drops is gone for good. 3.7 Flash costs more than Flash Lite per "
                                               "annual report, and is worth it.")
        map_model_name = st.selectbox("Stage 2 — Notes model", list(MODELS.keys()),
                                      index=default_model_index(DEFAULT_MAP_MODEL),
                                      help="Turns each chunk into notes. 3.7 Flash is the balance point.")
        reduce_model_name = st.selectbox("Stage 3 — Synthesis model", list(MODELS.keys()),
                                        index=default_model_index(DEFAULT_REDUCE_MODEL),
                                        help="Writes the final note — the stage where model quality shows most. "
                                             "3.1 Pro is the strongest available; switch to 2.5 Pro if you want a "
                                             "stable (non-preview) model.")

        st.divider()
        st.subheader("Final note length")
        st.caption("This is the length of the **narrative** sections. The key-financials, "
                   "one-page-summary and all-the-numbers sections add roughly "
                   f"{MANDATORY_BUDGET_TOTAL:,} words on top, so they never eat into the story.")
        # Radio rather than a selectbox: this sits low in the sidebar, and a dropdown
        # popup here opens below the fold. All five options fit inline.
        length_choice = st.radio("Target length", list(LENGTH_PRESETS.keys()), index=1)
        if LENGTH_PRESETS[length_choice] is None:
            target_word_count = st.number_input("Custom target (words)", min_value=500, max_value=20000, value=4000, step=500)
        else:
            target_word_count = LENGTH_PRESETS[length_choice]

        company_name = st.text_input("Company name (for file names)", value="Company",
                                     help="Used only to name the downloaded files, e.g. 20260722_Company_final.md")

    if api_key:
        configure_genai(api_key)

    # ── Start-from selector with plain-English explanation ────────────────────
    st.subheader("① Where do you want to start?")
    st.markdown(
        "This app has **three stages**. Pick where to begin depending on what you already have:\n\n"
        "- **Start fresh** — you have raw annual reports and transcripts. Runs all three stages.\n"
        "- **I already have the extract** — you ran Stage 1 before and saved the *extracted content*. Skips extraction.\n"
        "- **I already have the interim notes** — you have the *SynthNotes interim notes* file. Jumps straight to the final write-up."
    )
    start_from = st.radio(
        "Start from:",
        ["Start fresh (raw annual reports + transcripts)",
         "I already have the extracted content",
         "I already have the interim notes"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Uploaders depending on entry point ────────────────────────────────────
    ar_files, tx_files, extracted_files, interim_text = [], [], [], None
    pre_extract_transcripts = False

    if start_from.startswith("Start fresh"):
        st.subheader("② Upload your source documents")
        st.markdown("**Stage 1 (Extract)** will trim the boilerplate out of the annual reports so the later stages "
                    "only see the parts that matter. Upload plain-text (`.txt`) versions of your documents.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Annual reports & investor presentations**")
            st.caption("These get boilerplate-stripped in Stage 1.")
            ar_up = st.file_uploader("Annual reports / presentations (.txt)", type=["txt", "md"],
                                     accept_multiple_files=True, key="ar_up", label_visibility="collapsed")
            if ar_up:
                ar_files = [read_uploaded(f) for f in ar_up]
        with c2:
            st.markdown("**Earnings-call transcripts**")
            st.caption("Usually clean already — passed straight to notes by default.")
            tx_up = st.file_uploader("Transcripts (.txt)", type=["txt", "md"],
                                     accept_multiple_files=True, key="tx_up", label_visibility="collapsed")
            if tx_up:
                tx_files = [read_uploaded(f) for f in tx_up]
        pre_extract_transcripts = st.checkbox(
            "Also boilerplate-strip the transcripts in Stage 1 (optional; costs a little more)",
            value=False,
            help="Off = feed raw transcripts to the notes stage (what you did in your transmission work). "
                 "On = clean them first too.")

    elif start_from.startswith("I already have the extracted"):
        st.subheader("② Upload your extracted content")
        st.markdown("Upload the **extracted content** you saved earlier — either the combined extract file "
                    "or the individual 'shortened' documents. The app skips Stage 1 and starts at the notes stage.")
        ex_up = st.file_uploader("Extracted content (.txt)", type=["txt", "md"],
                                 accept_multiple_files=True, key="ex_up")
        if ex_up:
            extracted_files = [read_uploaded(f) for f in ex_up]

    else:  # interim
        st.subheader("② Upload your interim notes")
        st.markdown("Upload the **interim notes** file (starts with `==== SynthNotes MultiDoc — Interim Notes ====`). "
                    "The app skips straight to the final synthesis — no tokens spent re-reading the sources.")
        in_up = st.file_uploader("Interim notes (.txt)", type=["txt", "md"], accept_multiple_files=False, key="in_up")
        if in_up:
            _, interim_text = read_uploaded(in_up)

    # ── Synthesis instructions ────────────────────────────────────────────────
    st.divider()
    st.subheader("③ Instructions for the final note")
    st.caption("This is the prompt the synthesis stage follows. The default is your equity-research template — "
               "edit it if you want a different structure or focus.")
    with st.expander("View / edit the synthesis instructions", expanded=False):
        user_prompt = st.text_area("Synthesis instructions", value=DEFAULT_USER_PROMPT, height=320,
                                   label_visibility="collapsed")

    # ── Run ───────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("④ Run")
    st.markdown("One click runs every remaining stage. Each stage's output appears below as a **download** as soon as it's ready.")

    run = st.button("▶ Run pipeline", type="primary", use_container_width=True)

    if run:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar first.")
            st.stop()

        # Validate inputs for the chosen entry point
        if start_from.startswith("Start fresh") and not (ar_files or tx_files):
            st.error("Upload at least one annual report or transcript.")
            st.stop()
        if start_from.startswith("I already have the extracted") and not extracted_files:
            st.error("Upload at least one extracted-content file.")
            st.stop()
        if start_from.startswith("I already have the interim") and not interim_text:
            st.error("Upload your interim notes file.")
            st.stop()

        st.session_state.pop("usage_log", None)  # reset cost counter for this run

        # Progress lines render inside whichever st.status() block is active.
        status_write = st.write

        extract_model = get_model(extract_model_name)
        map_model = get_model(map_model_name)
        reduce_model = get_model(reduce_model_name)

        extract_text = None
        interim_notes_text = None
        final_doc = None

        try:
            # ── STAGE 1: EXTRACT ─────────────────────────────────────────────
            if start_from.startswith("Start fresh"):
                with st.status("Stage 1 — Extracting relevant content…", expanded=True) as s:
                    extracted_ar = extract_pass(ar_files, DEFAULT_EXTRACTION_AR_PROMPT,
                                                EXTRACTION_CHUNK_SIZE_AR, EXTRACTION_CHUNK_OVERLAP_AR,
                                                extract_model, status_write, "AR") if ar_files else []
                    if pre_extract_transcripts and tx_files:
                        extracted_tx = extract_pass(tx_files, DEFAULT_EXTRACTION_TRANSCRIPT_PROMPT,
                                                    EXTRACTION_CHUNK_SIZE_TRANSCRIPT, EXTRACTION_CHUNK_OVERLAP_TRANSCRIPT,
                                                    extract_model, status_write, "Transcript")
                        map_files = extracted_ar + extracted_tx
                        extract_text = serialize_combined_extract(extracted_ar, extracted_tx)
                    else:
                        # Default: extracted ARs + RAW transcripts (your real workflow)
                        map_files = extracted_ar + tx_files
                        extract_text = serialize_combined_extract(extracted_ar, [])
                    s.update(label="Stage 1 — Extraction done ✓", state="complete")
            elif start_from.startswith("I already have the extracted"):
                map_files = extracted_files
            else:
                map_files = None  # interim path skips map

            # ── STAGE 2: NOTES / MAP ─────────────────────────────────────────
            if start_from.startswith("I already have the interim"):
                notes_list, filenames = parse_interim(interim_text)
                status_write(f"Loaded interim notes: {len(notes_list)} section(s) from {len(filenames)} source file(s).")
            else:
                if not map_files:
                    st.error("Nothing to process after extraction — check your inputs.")
                    st.stop()
                with st.status("Stage 2 — Turning content into notes…", expanded=True) as s:
                    notes_list, filenames = run_map_stage(map_files, user_prompt, target_word_count,
                                                          map_model, status_write)
                    interim_notes_text = serialize_interim(notes_list, filenames)
                    s.update(label="Stage 2 — Notes done ✓", state="complete")

            # ── STAGE 3: SYNTHESISE / REDUCE ─────────────────────────────────
            with st.status("Stage 3 — Writing the final note…", expanded=True) as s:
                final_doc = hierarchical_reduce(notes_list, filenames, user_prompt, target_word_count,
                                                reduce_model, status_write)
                s.update(label="Stage 3 — Final note done ✓", state="complete")

            # Persist artifacts so the download buttons survive reruns
            st.session_state["out_extract"] = extract_text
            st.session_state["out_interim"] = interim_notes_text
            st.session_state["out_final"] = final_doc
            st.session_state["out_company"] = company_name

            # Stage auto-download — fires once on the next render, so a lost session
            # does not cost the run. The PDF carries the charts; the .md carries chart
            # captions with the underlying tables intact.
            _pending = []
            if extract_text:
                _pending.append((filename_for(company_name, "extract", "txt"),
                                 extract_text, "text/plain"))
            if interim_notes_text:
                _pending.append((filename_for(company_name, "interim", "txt"),
                                 interim_notes_text, "text/plain"))
            _pending.append((filename_for(company_name, "final", "md"),
                             markdown_for_download(final_doc), "text/markdown"))
            _final_pdf = markdown_to_pdf_bytes(final_doc)
            if _final_pdf:
                _pending.append((filename_for(company_name, "final", "pdf"),
                                 _final_pdf, "application/pdf"))
            st.session_state["pending_auto_download"] = _pending

        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()

    # ── Results (persist across reruns) ───────────────────────────────────────
    if st.session_state.get("out_final"):
        st.divider()
        _consume_pending_auto_download()
        st.subheader("⑤ Your outputs")
        st.markdown("Each stage's output is here. Download the interim ones too — they let you re-run a later stage "
                    "later without paying to redo the earlier ones.")
        company = st.session_state.get("out_company", "Company")

        cols = st.columns(4)
        if st.session_state.get("out_extract"):
            with cols[0]:
                st.download_button("⬇ Stage 1 · Extract (.txt)", data=st.session_state["out_extract"],
                                   file_name=filename_for(company, "extract", "txt"), mime="text/plain",
                                   use_container_width=True)
        if st.session_state.get("out_interim"):
            with cols[1]:
                st.download_button("⬇ Stage 2 · Interim notes (.txt)", data=st.session_state["out_interim"],
                                   file_name=filename_for(company, "interim", "txt"), mime="text/plain",
                                   use_container_width=True)
        with cols[2]:
            st.download_button("⬇ Final note (.md)", data=markdown_for_download(st.session_state["out_final"]),
                               file_name=filename_for(company, "final", "md"), mime="text/markdown",
                               use_container_width=True)
        with cols[3]:
            pdf_bytes = markdown_to_pdf_bytes(st.session_state["out_final"])
            if pdf_bytes:
                st.download_button("⬇ Final note (.pdf)", data=pdf_bytes,
                                   file_name=filename_for(company, "final", "pdf"), mime="application/pdf",
                                   use_container_width=True)
            else:
                st.caption("PDF needs `markdown` + `xhtml2pdf`")

        render_usage_panel()
        st.markdown("### Final note preview")
        render_markdown_with_charts(st.session_state["out_final"])


if __name__ == "__main__":
    main()
