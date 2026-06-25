"""
SynthNotes FactbaseNote — two-pass company research note generator.

Workflow:
  1. User uploads two buckets of .txt files:
       - Annual reports + investor presentations  (Pass 1 source)
       - Quarterly call transcripts               (Pass 2 source)
  2. User reviews / edits two prompts (defaults shipped):
       - Pass 1 (FACT-BASE extraction)
       - Pass 2 (ANALYSIS NOTE reasoning, using factbase as context)
  3. Pipeline runs sequentially:
       Pass 1 Map → Pass 1 Reduce (writes Sections A–F) → FACTBASE
       Pass 2 Map (with factbase) → Pass 2 Reduce (writes Sections 1–10) → ANALYSIS NOTE
       Merge (concatenation, no LLM): FACTBASE + ANALYSIS NOTE → final document
  4. Length is controlled by total word target + Pass 1/Pass 2 split + hardcoded
     within-pass section weights. Each section is written as a separate parallel
     LLM call with its own word budget — produces reliable length compliance.

Borrows chunking, retry, streaming, cost-tracking, and PDF helpers from
SynthNotes-Pro / SynthNotes-MultiDoc; self-contained — no imports from siblings.
"""

import streamlit as st
import google.generativeai as genai
import os, re, time, json, html as html_module
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import streamlit.components.v1 as components


# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY", "")
if _api_key:
    genai.configure(api_key=_api_key)

MAX_OUTPUT_TOKENS  = 65536
MAX_FILES_PER_BUCKET = 50         # Reasonable cap; can be raised
MAX_FILE_SIZE_MB   = 10
PARALLEL_WORKERS   = 3

# Different chunk sizes per source type — AR/IP are denser and benefit from
# larger context windows; transcripts are conversational and chunk smaller cleanly.
CHUNK_SIZE_AR_IP    = 5000
CHUNK_OVERLAP_AR_IP = 500
CHUNK_SIZE_TRANSCRIPT    = 4000
CHUNK_OVERLAP_TRANSCRIPT = 400

MODELS = {
    "Gemini 2.5 Flash (Fast)":       "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite (Cheap)": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro (Best)":         "gemini-2.5-pro",
    "Gemini 3.0 Flash":              "gemini-3-flash-preview",
    "Gemini 3.5 Flash":              "gemini-3.5-flash",
    "Gemini 2.0 Flash":              "gemini-2.0-flash-lite",
    "Gemini 1.5 Flash":              "gemini-1.5-flash",
}

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

# Length presets for the FINAL combined document (factbase + analysis note).
LENGTH_PRESETS = {
    "Short (~4000 words)":    4000,
    "Standard (~6000 words)": 6000,
    "Default (~8000 words)":  8000,
    "Long (~10000 words)":   10000,
    "Custom":                 None,
}

DEFAULT_PASS_SPLIT = 30           # Pass 1 share, as %. Pass 2 = 100 - this.


# ── 2. SECTION DEFINITIONS (fixed structure; weights tunable in code) ──────────
# Each pass has a prescribed set of sections from the user's prompt design.
# Weights within each pass distribute that pass's word budget across sections.
# Modifying these constants is the only way to adjust within-pass weights — no UI knob.

PASS_1_SECTIONS: List[Dict] = [
    {
        "key": "A",
        "heading": "A. BUSINESS DEFINITION",
        "scope": "One-line factual description of what the company makes; revenue split by segment (latest year and ~3-5y prior with the years shown); customer-type split (utility/private capex/EPC/OEM/export) if disclosed; domestic vs export geography split, latest and prior.",
        "weight": 0.10,
    },
    {
        "key": "B",
        "heading": "B. TECHNICAL POSITIONING",
        "scope": "Technical/product positioning by product line — only fill rows relevant to this company. Transformers (kV classes, dry vs oil, MVA capacity, utilisation), conductors/cables (mix, metal exposure, pass-through), switchgear/GIS (voltage class, indigenisation %), reactive/FACTS (products, tie-ups, import dependence), towers (tonnage, export %), turbines (MW range, new-build vs aftermarket), batteries/electronics (chemistry, end-markets, import content).",
        "weight": 0.30,
    },
    {
        "key": "C",
        "heading": "C. SCALE & CAPACITY",
        "scope": "Capacity by line; expansions announced with stated commissioning dates; utilisation rates if given; capex spent and planned with sources.",
        "weight": 0.15,
    },
    {
        "key": "D",
        "heading": "D. FINANCIAL SPINE",
        "scope": "Last 5y where available — revenue, EBITDA, EBITDA margin, PAT (yearly). Net debt, working-capital days (receivable/inventory/payable), RoCE if stated, dividend/buyback, contingent liabilities. Related-party transactions of note, promoter pledge, auditor qualifications, management/board changes. Facts only, no judgement.",
        "weight": 0.20,
    },
    {
        "key": "E",
        "heading": "E. PRESENTATION DIFF OVER TIME",
        "scope": "Across the ~4 presentation vintages, list metrics/slides that APPEARED, DISAPPEARED, or changed definition (e.g. stopped showing order-book-by-margin; dropped a guidance number; reworded the strategy slide). Each with the two vintages being compared. This is fact about disclosure behaviour, not interpretation.",
        "weight": 0.15,
    },
    {
        "key": "F",
        "heading": "F. EXTRACTION GAPS",
        "scope": "List of fields that were 'not in sources' — so the next pass (transcript reasoning) knows what to fill from transcripts.",
        "weight": 0.10,
    },
]

PASS_2_SECTIONS: List[Dict] = [
    {
        "key": "1",
        "heading": "1. GROWTH DRIVERS, YEAR BY YEAR",
        "scope": "FACTS: Year-by-year table (last ~5y) — revenue growth (stated/derived) | what management ATTRIBUTED it to (paraphrased, tagged) | segment/geography they named as driver | any one-off/large order. Keep strictly to management's stated cause; do not yet classify structural vs cyclical here. INTERPRETATION (my read): your split of structural vs cyclical/one-time, and which attributions look durable vs fragile.",
        "weight": 0.14,
    },
    {
        "key": "2",
        "heading": "2. BIDDING DISCIPLINE & ORDER STRATEGY",
        "scope": "FACTS: Tagged bullets of what management actually said about order selection, walking away, margin thresholds, order-book size/inflow/execution period. INTERPRETATION (my read): genuine discipline or fill-the-factory; whether it changed as the cycle got hot.",
        "weight": 0.10,
    },
    {
        "key": "3",
        "heading": "3. COMPETITIVE INTENSITY & MOAT",
        "scope": "FACTS: Two tagged lists — (a) what mgmt said about their OWN advantage / why customers choose them; (b) what mgmt said about competition (new entrants, peer capacity, imports, pricing) and explicitly whether unprompted or analyst-pushed, direct or deflected. INTERPRETATION (my read): durable moat vs current tightness; competition genuinely rising; share gain/loss likely.",
        "weight": 0.12,
    },
    {
        "key": "4",
        "heading": "4. MARGIN DRIVERS",
        "scope": "FACTS: Tagged bullets of every reason management gave for margin moves — pricing/shortage, mix, operating leverage, specific commodity, employee cost, forex, pass-through ability and lag. Separate tailwinds from headwinds. INTERPRETATION (my read): how much of current margin is shortage-driven vs structural, and sustainability — cross-checked against F-FIN margin trajectory.",
        "weight": 0.14,
    },
    {
        "key": "5",
        "heading": "5. GROWTH CONSTRAINTS & MANPOWER",
        "scope": "Weight the LAST 4 QUARTERS. FACTS: Every constraint management named (capacity, specific input, working capital, customer/RoW delay, manpower); exactly what they said on manpower/attrition/hiring; remedy and timeline given. INTERPRETATION (my read): which constraint is genuinely binding; whether remedies are credible/on-track.",
        "weight": 0.10,
    },
    {
        "key": "6",
        "heading": "6. ANALYST QUESTION LOG — LAST ~8 QUARTERS",
        "scope": "Here-and-now; highest priority section. FACTS ONLY (no interpretation until the end): chronological table — Quarter [Q_FY__] | analyst question theme | management response (paraphrase; ≤25-word quote only where wording matters) | Direct/Partial/Evasive. Then INTERPRETATION (my read): which concerns recur, which got dodged, how the line of questioning shifted across the 8 quarters.",
        "weight": 0.18,
    },
    {
        "key": "7",
        "heading": "7. MANAGEMENT QUALITY & CYCLICALITY AWARENESS",
        "scope": "FACTS: (a) tagged quotes/paraphrases showing how they talk about the down-cycle in GOOD quarters (warn vs extrapolate); (b) capacity-expansion actions stated, with order-backing if mentioned; (c) SAID-vs-HAPPENED table: past forward statement [Q_FY__] | outcome | source — marked Delivered/Partly/Missed/Too-early (this is factual reconciliation, allowed here). INTERPRETATION (my read): cyclicality-awareness 1-5 with two strongest supporting facts; capital-allocation read referencing F-BS / F-GOV.",
        "weight": 0.10,
    },
    {
        "key": "8",
        "heading": "8. DISSONANCE SUMMARY",
        "scope": "FACTS: List each contradiction as transcript claim [Q_FY__] vs factbase F-xx, stated plainly with both sides. INTERPRETATION (my read): ranked by importance to the investment view.",
        "weight": 0.08,
    },
    {
        "key": "9",
        "heading": "9. OPEN QUESTIONS FOR THE FORWARD PASS",
        "scope": "Plain list. 3-6 specific things the backward sources don't resolve. No speculation. No INTERPRETATION block needed for this section.",
        "weight": 0.03,
    },
    {
        "key": "10",
        "heading": "10. SCORECARD ROW",
        "scope": "ONE pipe-delimited line — this is explicit synthesis, lives outside the facts/interpretation split. Columns: Company | segments (top 3) | end-customer skew | domestic/export % | 5y rev CAGR | 5y EBITDA CAGR | margin trend | primary growth driver (≤4 words) | structural vs cyclical | moat type (≤4 words) | competition rising? (Y/N) | binding constraint (≤4 words) | bidding discipline (H/M/L) | guidance reliability (H/M/L) | cyclicality awareness (1-5) | top dissonance flag (≤6 words). No INTERPRETATION block.",
        "weight": 0.01,
    },
]


# ── 3. DEFAULT PROMPTS (the user can edit these in the UI; defaults shipped here) ──

DEFAULT_PASS_1_PROMPT = """ROLE
You are a financial-data extractor. Build a structured FACT-BASE for ONE company using
ONLY the annual reports and investor presentations in this project. You are NOT writing
analysis or narrative in this pass — you extract verifiable facts and positioning signals
that a later reasoning pass will use.

SOURCES FOR THIS PASS

- Annual reports: FACT SOURCE ONLY — segment splits, capacity, kV/product detail,
employee numbers, balance-sheet items, related-party items, auditor matters.
- Investor presentations: facts (order book, segment/geography splits) AND the
positioning signal of what management chooses to show or drop across vintages.
- DO NOT use transcripts in this pass. DO NOT use any outside/training data.

HARD RULES

1. Every fact is tagged with a stable ID and a source: [FY__ AR p__] or [<date> IP slide__].
ID format: F-<CATEGORY>-<nn>, e.g. F-SEG-01, F-CAP-02, F-BS-03, F-IPDIFF-01.
2. STATED vs DERIVED: derived figures labelled "(derived)" with inputs shown.
3. If a field is not in the documents, write "not in sources" — never infer or fill.
4. No narrative, no adjectives, no outlook language. Facts only. If a sentence reads like
an annual-report MD&A line, it does not belong in this pass.
5. Reproduce annual-report prose? Never. Extract the number/fact, not the paragraph.

OUTPUT — a fact-base in these blocks. Keep each fact one line, tagged.

A. BUSINESS DEFINITION
- F-BIZ: one-line factual description of what they make.
- F-SEG: revenue split by segment, % for latest year AND ~3-5y prior (show the years).
- F-CUST: customer-type split if disclosed (utility / private capex / EPC / OEM / export), with %.
- F-GEO: domestic vs export split, latest and prior, with the years.

B. TECHNICAL POSITIONING  (use the relevant component rows; mark derived where inferred)
- Transformers: kV classes made (dist <33 / power 66-220 / EHV 400 / 765 / HVDC); dry vs oil; installed MVA capacity; utilisation if given. Tag each F-TECH-nn.
- Conductors/cables: product mix (ACSR / HTLS / EHV cable / OPGW); metal exposure; stated pass-through mechanism.
- Switchgear/GIS: voltage class; GIS vs AIS; indigenisation %.
- Reactive/FACTS: products; technology tie-ups; import dependence.
- Towers: tonnage capacity; galvanising; export %.
- Turbines/generators: MW range; new-build vs aftermarket split; application mix.
- Batteries/electronics: chemistry; application end-markets; import content.
(Only fill the rows relevant to this company.)

C. SCALE & CAPACITY
- F-CAP: capacity by line, expansions announced (with stated commissioning dates), utilisation. Capex spent/planned with sources.

D. FINANCIAL SPINE  (from AR, last 5y where available)
- F-FIN: revenue, EBITDA, EBITDA margin, PAT, for each year. (derived for CAGRs/margins)
- F-BS: net debt, working-capital days (receivable/inventory/payable if derivable), RoCE if stated, dividend/buyback, contingent liabilities.
- F-GOV: related-party transactions of note, promoter pledge, auditor qualifications, management/board changes. Facts only, no judgement.

E. PRESENTATION DIFF OVER TIME  (the positioning signal)
- F-IPDIFF: across the ~4 presentation vintages, list metrics/slides that APPEARED, DISAPPEARED, or changed definition (e.g. stopped showing order-book-by-margin; dropped a guidance number; reworded the strategy slide). Each with the two vintages being compared. This is fact about disclosure behaviour, not interpretation.

F. EXTRACTION GAPS
- List the fields that were "not in sources" — so the reasoning pass knows what the transcripts need to fill.

STYLE
Terse, tabular where natural, every line tagged and sourced. No prose paragraphs.
This document will be pasted verbatim into the next pass as reference context.
"""


DEFAULT_PASS_2_PROMPT = """ROLE
You are an equity analyst writing the REASONING note on ONE company. Your PRIMARY source
is the earnings-call transcripts (~5 years) in this project — especially the analyst Q&A.
You are also given a FACT-BASE produced from the annual reports and presentations; use it
as reference and check the transcript narrative against it.

[PASTE company_factbase.md HERE]

SOURCE DISCIPLINE
- Transcripts = PRIMARY for all WHY questions. Tag transcript claims [Q_FY__].
- FACT-BASE above = reference for facts; cite its IDs (e.g. F-SEG-02) when you use/test one.
- No outside/training data. Not in transcripts and not in the factbase = "not in sources".
- Quotes ≤25 words, exact, attributed. Prefer paraphrase. Never invent a quote.

THE GOLDEN RULE OF THIS PASS — FACTS BEFORE INTERPRETATION
In every section you produce TWO clearly separated parts:
① FACTS — what management actually said / what the documents show. Presented as a table
or tight tagged bullet list (NOT prose paragraphs). Each line is a paraphrase or a
short exact quote, with a source tag [Q_FY__]. No inference, no adjectives, no "this
suggests". If management said it, it goes here, attributed. If you concluded it, it
does NOT go here.
② INTERPRETATION (my read) — a clearly labelled block beneath the facts. This is the ONLY
place your inference, judgement, or conclusions may appear. Begin it literally with
"INTERPRETATION (my read):". Keep it to a short paragraph. If facts are thin, say the
interpretation is low-confidence.
A reader must be able to read ONLY the ① FACTS across the whole note and have a complete,
accurate picture of what management said, untouched by your opinion. If you ever find
yourself writing a conclusion inside a FACTS block, move it down to INTERPRETATION.

FORMAT each section like this:

## <Section name>

FACTS
| … scannable table or tagged bullets … |
INTERPRETATION (my read): <one short paragraph, clearly your inference>

OUTPUT — these sections, in order:
1. GROWTH DRIVERS, YEAR BY YEAR
2. BIDDING DISCIPLINE & ORDER STRATEGY
3. COMPETITIVE INTENSITY & MOAT
4. MARGIN DRIVERS
5. GROWTH CONSTRAINTS & MANPOWER (weight the LAST 4 QUARTERS)
6. ANALYST QUESTION LOG — LAST ~8 QUARTERS (FACTS ONLY in this section; INTERPRETATION at the end)
7. MANAGEMENT QUALITY & CYCLICALITY AWARENESS (cyclicality-awareness 1-5)
8. DISSONANCE SUMMARY (transcript claim vs factbase F-xx)
9. OPEN QUESTIONS FOR THE FORWARD PASS (plain list, 3-6 items)
10. SCORECARD ROW (pipe-delimited single line; explicit synthesis, outside facts/interpretation split)

STYLE
Within each section: ① FACTS scannable (tables / tagged bullets, never prose) ②
INTERPRETATION one short labelled paragraph. Thin sources → short + say what's missing.
Never let interpretation leak into a FACTS block. Never pad with generic industry talk.
"""


# ── 4. PROMPT TEMPLATES (wrap user prompts for Map / Section-write stages) ─────

MAP_WRAPPER_TEMPLATE = """{user_pass_prompt}

---

IMPORTANT — YOU ARE SEEING ONLY ONE CHUNK OF THE INPUT
This is **chunk {chunk_n} of {total_chunks}** from source file: **{filename}**.
You may be seeing only part of a single document, and other source documents in this pass are being processed in parallel.

YOUR SPECIFIC TASK FOR THIS CHUNK
- Do NOT produce the full structured output (do NOT emit Sections A–F / 1–10 in their final form for this chunk alone).
- Instead, EXTRACT every substantive item from this chunk that's relevant to the user's prompt above — facts, quotes, data points, named entities, dates, source tags.
- For each extracted item, tag it with the section letter (A/B/C/D/E/F for Pass 1) or number (1–10 for Pass 2) it most belongs to, so a later assembly step can route it.
- Preserve the user's source-tag format (e.g. `[FY__ AR p__]`, `[<date> IP slide__]`, `[Q_FY__]`).
- Do NOT invent facts or content not present in the source chunk.
- Do NOT include meta-commentary about the chunk.

OUTPUT FORMAT for this chunk
For each extractable item, output ONE line like:
`[Section <letter or number>] <the extracted fact, quote, or data point with its source tag>`

SOURCE CHUNK
{chunk_text}
"""


PASS_1_SECTION_PROMPT = """You are writing ONE SECTION of a structured FACT-BASE document. The factbase is being assembled from facts extracted across multiple annual-report and investor-presentation chunks.

### USER'S FULL PASS 1 PROMPT (for HARD RULES, STYLE, ID conventions, OUTPUT specifications)
{user_pass1_prompt}

### YOUR ASSIGNED SECTION
- **Heading**: {section_heading}
- **Scope**: {section_scope}
- **Target word count**: approximately **{section_word_budget} words**

### CRITICAL — DO NOT COVER OTHER SECTIONS' MATERIAL
Other sections (A, B, C, D, E, F — see your prompt above) are being written by separate parallel calls. Stay strictly within YOUR assigned section's scope.

### ID ASSIGNMENT
This is the FIRST place IDs are finalised. Number facts within your section starting from 01 — e.g., for Section B use `F-TECH-01`, `F-TECH-02`, …; for Section D use `F-FIN-01`, `F-BS-01`, etc. Follow the user's ID prefix conventions from the Pass 1 prompt above.

### EXTRACTED PER-CHUNK FACTS (tagged by section — pull items relevant to YOUR section)
{combined_map_output}

### INSTRUCTIONS
1. Read all the extracted facts above.
2. Select facts tagged `[Section {section_key}]` (or which clearly belong to your section).
3. Consolidate, deduplicate, and produce your section's content following the user's Pass 1 prompt for THIS section.
4. Begin with the section heading exactly: `{section_heading}`
5. Assign final unique IDs per the user's conventions.
6. Stay near **{section_word_budget} words**. If your draft is much shorter than budget, pull more facts from the per-chunk extracts (you may be under-using available material). If much longer, you may be including material assigned to other sections.
7. Follow the user's HARD RULES and STYLE rules from the Pass 1 prompt exactly.
8. No preamble. No conclusion. Just the section.

Write your section now."""


PASS_2_SECTION_PROMPT = """You are writing ONE SECTION of a structured ANALYSIS NOTE, assembled from per-quarter transcript notes.

### USER'S FULL PASS 2 PROMPT (with FACT-BASE injected — has SOURCE DISCIPLINE, GOLDEN RULE, FORMAT, STYLE)
{user_pass2_prompt_with_factbase}

### YOUR ASSIGNED SECTION
- **Heading**: {section_heading}
- **Scope**: {section_scope}
- **Target word count**: approximately **{section_word_budget} words**

### CRITICAL — DO NOT COVER OTHER SECTIONS' MATERIAL
Other sections (1 through 10 — see your prompt above) are being written by separate parallel calls. Stay strictly within YOUR assigned section.

### GOLDEN RULE — FACTS BEFORE INTERPRETATION
Per the user's Pass 2 prompt, each section MUST have:
① **FACTS** — tagged bullets or table (no inference, no adjectives, no "this suggests")
② **INTERPRETATION (my read):** — one short paragraph clearly labelled, ONLY place inference appears
A reader should be able to read only the ① FACTS across the whole note for a complete factual picture.

Sections 9 and 10 are exceptions:
- Section 9 (Open Questions): plain list only, no INTERPRETATION block
- Section 10 (Scorecard Row): single pipe-delimited line, no INTERPRETATION block

### EXTRACTED PER-CHUNK ANALYSIS (tagged by section)
{combined_map_output}

### INSTRUCTIONS
1. Read all the extracted material above.
2. Select content tagged `[Section {section_key}]` (or which clearly belongs to your section).
3. Consolidate per the user's Pass 2 prompt specifications for THIS section's format and content.
4. Begin with the section heading exactly: `{section_heading}`
5. Apply the FACTS / INTERPRETATION format from the user's prompt (where applicable).
6. Section 6: produce a chronological table — Quarter [Q_FY__] | analyst question theme | management response paraphrase | Direct/Partial/Evasive. INTERPRETATION at the end of the table.
7. Section 8 (Dissonance): cite specific factbase IDs (F-xx) in each contradiction line.
8. Section 10 (Scorecard): ONE pipe-delimited line only. Keep it terse.
9. Stay near **{section_word_budget} words**. If your draft is much shorter than budget, pull more from the per-chunk extracts.
10. Cite [Q_FY__] tags on all transcript claims; cite F-xx tags when referencing factbase items.
11. No preamble. No conclusion. Just the section.

Write your section now."""


# ── 5. UTILITIES ───────────────────────────────────────────────────────────────

def get_model(display_name: str) -> genai.GenerativeModel:
    cache = st.session_state.setdefault("_model_cache", {})
    model_id = MODELS.get(display_name, "gemini-2.5-flash")
    if model_id not in cache:
        cache[model_id] = genai.GenerativeModel(model_id)
    return cache[model_id]


def _record_usage(model_id: str, response, stage: str = "") -> None:
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


def generate_with_retry(model, prompt, max_retries: int = 3, stream: bool = False,
                        generation_config=None, stage: str = ""):
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


# ── Section-routing helper ─────────────────────────────────────────────────────
# Each Map-stage output line is tagged `[Section X] ...`. Grouping by tag at
# Reduce time means each section writer only receives its own facts — cutting
# Reduce-stage input tokens by ~5-9× vs the original "pass everything to everyone"
# behaviour, with no quality impact when Map tagging is well-behaved.

# Matches:  [Section A] ...  •  [Section 1] ...  •  [Section 2.A] ...  •  case-insensitive
_SECTION_TAG_RE = re.compile(r"^\s*\[Section\s+([A-Za-z0-9.]+)\]\s*(.*)$", re.IGNORECASE)


def route_by_section(map_output: List[str]) -> Tuple[Dict[str, List[str]], List[str], float]:
    """Parse all Map output lines and group by their [Section X] tag.
    Returns:
        (by_section, unrouted_lines, unrouted_fraction)
        - by_section: dict keyed by section letter/number → list of tagged lines
        - unrouted_lines: lines without a recognisable section tag (sent as
          a catch-all pool to every section writer)
        - unrouted_fraction: fraction of total lines that were unrouted
          (status panel surfaces this so you can sanity-check Map tagging)
    """
    by_section: Dict[str, List[str]] = {}
    unrouted: List[str] = []
    total_lines = 0
    for chunk_output in map_output:
        for raw_line in chunk_output.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            total_lines += 1
            m = _SECTION_TAG_RE.match(line)
            if m:
                key = m.group(1).upper()
                by_section.setdefault(key, []).append(line)
            else:
                unrouted.append(line)
    unrouted_fraction = (len(unrouted) / total_lines) if total_lines else 0.0
    return by_section, unrouted, unrouted_fraction


# ── Pre-flight cost estimation ─────────────────────────────────────────────────
# Rough estimate shown to the user BEFORE running, so they can sanity-check
# expected spend at current input size and model selection. Real cost typically
# lands within ±30% of this. Output-token estimates assume the Map stage extracts
# ~2,000 words per chunk and Reduce produces ~the configured word budget.

_TOKENS_PER_WORD = 1.4  # tagged factual content tokenises slightly above plain prose


def estimate_pipeline_cost(
    ar_ip_words: int, transcript_words: int,
    pass1_word_budget: int, pass2_word_budget: int,
    map_model_id: str, reduce_model_id: str,
) -> Dict:
    """Estimate cost in USD for each pipeline stage at current settings."""
    # Pass 1 Map
    p1_step = CHUNK_SIZE_AR_IP - CHUNK_OVERLAP_AR_IP
    p1_chunks = max(1, (ar_ip_words + p1_step - 1) // p1_step) if ar_ip_words else 0
    p1_prompt_overhead_words = 1500
    p1_map_in_tokens  = int(p1_chunks * (CHUNK_SIZE_AR_IP + p1_prompt_overhead_words) * _TOKENS_PER_WORD)
    p1_map_out_tokens = int(p1_chunks * 2000 * _TOKENS_PER_WORD)
    p1_map_cost = compute_cost(p1_map_in_tokens, p1_map_out_tokens, map_model_id)

    # Pass 1 Reduce (post-routing: each section receives ~1/6 of Map output)
    p1_map_total_output_tokens = p1_map_out_tokens
    p1_reduce_in_tokens  = int(p1_map_total_output_tokens + 6 * p1_prompt_overhead_words * _TOKENS_PER_WORD)
    p1_reduce_out_tokens = int(pass1_word_budget * _TOKENS_PER_WORD)
    p1_reduce_cost = compute_cost(p1_reduce_in_tokens, p1_reduce_out_tokens, reduce_model_id)

    # Pass 2 Map (factbase carried into every chunk)
    p2_step = CHUNK_SIZE_TRANSCRIPT - CHUNK_OVERLAP_TRANSCRIPT
    p2_chunks = max(1, (transcript_words + p2_step - 1) // p2_step) if transcript_words else 0
    p2_factbase_overhead_words = pass1_word_budget  # factbase carried per chunk
    p2_map_in_tokens  = int(p2_chunks * (CHUNK_SIZE_TRANSCRIPT + p1_prompt_overhead_words + p2_factbase_overhead_words) * _TOKENS_PER_WORD)
    p2_map_out_tokens = int(p2_chunks * 2000 * _TOKENS_PER_WORD)
    p2_map_cost = compute_cost(p2_map_in_tokens, p2_map_out_tokens, map_model_id)

    # Pass 2 Reduce (post-routing: each section receives ~1/10 of Map output;
    # factbase carried into each of 10 section calls)
    p2_reduce_in_tokens  = int(p2_map_out_tokens + 10 * (pass1_word_budget + p1_prompt_overhead_words) * _TOKENS_PER_WORD)
    p2_reduce_out_tokens = int(pass2_word_budget * _TOKENS_PER_WORD)
    p2_reduce_cost = compute_cost(p2_reduce_in_tokens, p2_reduce_out_tokens, reduce_model_id)

    return {
        "pass1_map_cost":    p1_map_cost,
        "pass1_reduce_cost": p1_reduce_cost,
        "pass2_map_cost":    p2_map_cost,
        "pass2_reduce_cost": p2_reduce_cost,
        "total_cost":        p1_map_cost + p1_reduce_cost + p2_map_cost + p2_reduce_cost,
        "pass1_chunks":      p1_chunks,
        "pass2_chunks":      p2_chunks,
    }


# ── 6. MAP STAGE (Pass 1 and Pass 2) ───────────────────────────────────────────

def _map_one_chunk(
    user_pass_prompt: str, chunk_text: str, chunk_n: int, total_chunks: int,
    filename: str, model, stage_label: str,
) -> Optional[str]:
    """Apply a user's pass-prompt to one chunk via the MAP wrapper template."""
    prompt = MAP_WRAPPER_TEMPLATE.format(
        user_pass_prompt=user_pass_prompt.strip(),
        chunk_n=chunk_n,
        total_chunks=total_chunks,
        filename=filename,
        chunk_text=chunk_text,
    )
    try:
        resp = generate_with_retry(model, prompt, stage=stage_label)
        return resp.text
    except Exception as e:
        return f"_[Section {chunk_n} of {filename} failed: {e}]_"


def map_pass(
    files: List[Tuple[str, str]], user_pass_prompt: str,
    chunk_size: int, overlap: int, model,
    stage_label: str, status_write,
) -> List[str]:
    """Run the Map stage for a pass: chunk all files, process all chunks in
    parallel, return the list of per-chunk outputs in original order."""
    tasks: List[Tuple[int, int, int, str, str]] = []
    for file_idx, (filename, content) in enumerate(files, start=1):
        chunks = create_chunks_with_overlap(content, chunk_size, overlap)
        for i, chunk_text in enumerate(chunks, start=1):
            tasks.append((file_idx, i, len(chunks), filename, chunk_text))

    status_write(
        f"  {stage_label} stage: {len(tasks)} section(s) across {len(files)} file(s), "
        f"parallel × {PARALLEL_WORKERS} workers"
    )

    results: List[Optional[str]] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _map_one_chunk, user_pass_prompt, chunk_text, chunk_n, total_chunks,
                filename, model, stage_label,
            ): idx
            for idx, (file_position, chunk_n, total_chunks, filename, chunk_text) in enumerate(tasks)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            status_write(f"    • {stage_label}: {done}/{len(tasks)} chunks complete")

    return [r for r in results if r and r.strip()]


# ── 7. SECTION-WRITING STAGE (Pass 1 and Pass 2 Reduce) ────────────────────────

def _write_pass1_section(
    section: Dict, user_pass1_prompt: str, combined_map_output: str, model,
) -> str:
    """Write ONE section (A–F) of the factbase, with assigned word budget."""
    prompt = PASS_1_SECTION_PROMPT.format(
        user_pass1_prompt=user_pass1_prompt.strip(),
        section_heading=section["heading"],
        section_scope=section["scope"],
        section_key=section["key"],
        section_word_budget=section["word_budget"],
        combined_map_output=combined_map_output,
    )
    resp = generate_with_retry(model, prompt, stage="Pass 1 section write")
    return resp.text


def _write_pass2_section(
    section: Dict, user_pass2_prompt_with_factbase: str,
    combined_map_output: str, model,
) -> str:
    """Write ONE section (1–10) of the analysis note, with assigned word budget."""
    prompt = PASS_2_SECTION_PROMPT.format(
        user_pass2_prompt_with_factbase=user_pass2_prompt_with_factbase.strip(),
        section_heading=section["heading"],
        section_scope=section["scope"],
        section_key=section["key"],
        section_word_budget=section["word_budget"],
        combined_map_output=combined_map_output,
    )
    resp = generate_with_retry(model, prompt, stage="Pass 2 section write")
    return resp.text


def reduce_pass1(
    map_output: List[str], user_pass1_prompt: str, pass1_word_budget: int,
    model, status_write,
) -> str:
    """Assemble FACTBASE by writing each section A–F in parallel with its budget.

    Uses section-routing: each section writer receives ONLY the Map output lines
    tagged for its section, plus a catch-all pool of unrouted lines (lines that
    didn't carry a `[Section X]` tag). Cuts Reduce input tokens by ~5-6× vs the
    previous "all-to-all" approach.
    """
    by_section, unrouted, unrouted_fraction = route_by_section(map_output)
    status_write(
        f"  Pass 1 Reduce: routed {sum(len(v) for v in by_section.values())} tagged lines into "
        f"{len(by_section)} section bucket(s); {len(unrouted)} unrouted lines treated as catch-all "
        f"({unrouted_fraction:.0%} of total)"
    )

    sections = []
    for s in PASS_1_SECTIONS:
        sections.append({**s, "word_budget": int(pass1_word_budget * s["weight"])})

    status_write(f"  Pass 1 Reduce: writing {len(sections)} sections (A–F) in parallel")

    def _section_input(sec_key: str) -> str:
        """Lines tagged for this section + the catch-all unrouted pool."""
        section_lines = by_section.get(sec_key.upper(), [])
        return "\n".join(section_lines + unrouted)

    results: List[Optional[str]] = [None] * len(sections)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _write_pass1_section, section, user_pass1_prompt,
                _section_input(section["key"]), model,
            ): i
            for i, section in enumerate(sections)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            sec = sections[i]
            wc = len(results[i].split()) if results[i] else 0
            section_input_words = len(_section_input(sec["key"]).split())
            status_write(
                f"    • Pass 1 §{sec['key']}: {done}/{len(sections)} '{sec['heading']}' "
                f"({wc:,} words written from {section_input_words:,}-word input, budget ~{sec['word_budget']:,})"
            )

    return "\n\n".join(r.strip() for r in results if r and r.strip())


def reduce_pass2(
    map_output: List[str], user_pass2_prompt_with_factbase: str,
    pass2_word_budget: int, model, status_write,
) -> str:
    """Assemble ANALYSIS NOTE by writing each section 1–10 in parallel with its budget.

    Uses the same section-routing pattern as Pass 1 Reduce. Cuts Reduce input
    tokens by ~9-10× vs the previous "all-to-all" approach.
    """
    by_section, unrouted, unrouted_fraction = route_by_section(map_output)
    status_write(
        f"  Pass 2 Reduce: routed {sum(len(v) for v in by_section.values())} tagged lines into "
        f"{len(by_section)} section bucket(s); {len(unrouted)} unrouted lines treated as catch-all "
        f"({unrouted_fraction:.0%} of total)"
    )

    sections = []
    for s in PASS_2_SECTIONS:
        sections.append({**s, "word_budget": int(pass2_word_budget * s["weight"])})

    status_write(f"  Pass 2 Reduce: writing {len(sections)} sections (1–10) in parallel")

    def _section_input(sec_key: str) -> str:
        section_lines = by_section.get(sec_key.upper(), [])
        return "\n".join(section_lines + unrouted)

    results: List[Optional[str]] = [None] * len(sections)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _write_pass2_section, section, user_pass2_prompt_with_factbase,
                _section_input(section["key"]), model,
            ): i
            for i, section in enumerate(sections)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            sec = sections[i]
            wc = len(results[i].split()) if results[i] else 0
            section_input_words = len(_section_input(sec["key"]).split())
            status_write(
                f"    • Pass 2 §{sec['key']}: {done}/{len(sections)} '{sec['heading']}' "
                f"({wc:,} words written from {section_input_words:,}-word input, budget ~{sec['word_budget']:,})"
            )

    return "\n\n".join(r.strip() for r in results if r and r.strip())


def inject_factbase(pass2_prompt: str, factbase: str) -> str:
    """Substitute the factbase into the user's Pass 2 prompt at the placeholder
    location, or prepend it as a labelled block if the placeholder is missing."""
    placeholder = "[PASTE company_factbase.md HERE]"
    if placeholder in pass2_prompt:
        return pass2_prompt.replace(placeholder, factbase)
    return (
        "### COMPANY FACT-BASE (reference for this pass)\n"
        f"{factbase}\n\n"
        "---\n\n"
        f"{pass2_prompt}"
    )


# ── 8. INTERIM SAVE/LOAD ───────────────────────────────────────────────────────
# Per the user's request: save FACTBASE + Pass 2 Map output (NOT Pass 1 Map).
# Resume mode loads this file and skips both Map stages; goes straight to Pass 2 Reduce.

INTERIM_HEADER       = "==== SynthNotes FactbaseNote — Interim Notes ===="
INTERIM_FACTBASE_TAG = "==== FACTBASE ===="
INTERIM_PASS2_MAP_TAG = "==== PASS_2_MAP ===="
INTERIM_PASS2_CHUNK_SEP = "==== CHUNK ===="


def serialize_interim(
    factbase: str, pass2_map_output: List[str], filenames_ar_ip: List[str],
    filenames_transcripts: List[str],
) -> str:
    """Serialise factbase + Pass 2 Map output to a single .txt file."""
    lines = [
        INTERIM_HEADER,
        f"AR/IP source files ({len(filenames_ar_ip)}):",
    ]
    for f in filenames_ar_ip:
        lines.append(f"- {f}")
    lines.append(f"Transcript source files ({len(filenames_transcripts)}):")
    for f in filenames_transcripts:
        lines.append(f"- {f}")
    lines.append("")
    lines.append(INTERIM_FACTBASE_TAG)
    lines.append(factbase)
    lines.append("")
    lines.append(INTERIM_PASS2_MAP_TAG)
    lines.append(("\n\n" + INTERIM_PASS2_CHUNK_SEP + "\n").join(pass2_map_output))
    return "\n".join(lines) + "\n"


def parse_interim(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """Parse a saved interim file. Returns (factbase, pass2_map_output, ar_ip_filenames, transcript_filenames)."""
    # Filenames from header
    ar_ip_files: List[str] = []
    transcript_files: List[str] = []

    ar_match = re.search(r"AR/IP source files \(\d+\):\s*\n((?:-\s.+\n?)+)", text)
    if ar_match:
        for line in ar_match.group(1).splitlines():
            s = line.strip()
            if s.startswith("- "):
                ar_ip_files.append(s[2:].strip())

    tx_match = re.search(r"Transcript source files \(\d+\):\s*\n((?:-\s.+\n?)+)", text)
    if tx_match:
        for line in tx_match.group(1).splitlines():
            s = line.strip()
            if s.startswith("- "):
                transcript_files.append(s[2:].strip())

    # Factbase: between FACTBASE tag and PASS_2_MAP tag
    factbase = ""
    fb_start = text.find(INTERIM_FACTBASE_TAG)
    p2_start = text.find(INTERIM_PASS2_MAP_TAG)
    if fb_start != -1 and p2_start != -1 and p2_start > fb_start:
        factbase = text[fb_start + len(INTERIM_FACTBASE_TAG): p2_start].strip()

    # Pass 2 Map output: after PASS_2_MAP tag, split on CHUNK separators
    pass2_map: List[str] = []
    if p2_start != -1:
        rest = text[p2_start + len(INTERIM_PASS2_MAP_TAG):].strip()
        if INTERIM_PASS2_CHUNK_SEP in rest:
            pass2_map = [c.strip() for c in rest.split(INTERIM_PASS2_CHUNK_SEP) if c.strip()]
        elif rest:
            pass2_map = [rest]

    return factbase, pass2_map, ar_ip_files, transcript_files


# ── 9. PDF EXPORT ──────────────────────────────────────────────────────────────

def markdown_to_pdf_bytes(md_text: str, title: str = "SynthNotes FactbaseNote") -> Optional[bytes]:
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
  p  {{ margin: 6px 0; }}
  ul, ol {{ margin: 6px 0; padding-left: 22px; }}
  li {{ margin: 3px 0; }}
  code {{ background: #f4f4f4; padding: 1px 4px; font-family: 'Courier New', monospace; font-size: 9.5pt; }}
  pre {{ background: #f4f4f4; padding: 8px; font-family: 'Courier New', monospace; font-size: 9.5pt; }}
  blockquote {{ border-left: 3px solid #ccc; padding-left: 10px; margin-left: 0; color: #555; font-style: italic; }}
  table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
  td, th {{ border: 1px solid #ddd; padding: 5px 8px; font-size: 9.5pt; }}
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


# ── 10. UI ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SynthNotes FactbaseNote", layout="wide", page_icon="📑")
    st.title("📑 SynthNotes FactbaseNote")
    st.caption(
        "Two-pass company research-note generator. Pass 1: extract a tagged FACT-BASE "
        "from annual reports + investor presentations. Pass 2: write the ANALYSIS NOTE "
        "from quarterly transcripts, using the factbase as reference. Output is the "
        "concatenation of both."
    )
    api_key_check()

    # ── Sidebar: model settings ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Model Settings")
        map_model_name = st.selectbox(
            "Map model (per-chunk extraction)",
            list(MODELS.keys()), index=1,  # Flash-Lite — 3× cheaper than Flash, fine for tagged extraction
            key="map_model",
            help=(
                "Used for Pass 1 and Pass 2 Map stages. Default is **Flash-Lite** "
                "(cheapest); upgrade to Flash if extraction quality looks thin. "
                "Map is mechanical work — extracting tagged facts — not reasoning, "
                "so the cheapest tier usually suffices."
            ),
        )
        reduce_model_name = st.selectbox(
            "Reduce model (section writing)",
            list(MODELS.keys()), index=2,
            key="reduce_model",
            help="Used to write each section of FACTBASE and ANALYSIS NOTE. Higher quality recommended (Pro by default).",
        )

    # ── Mode toggle ────────────────────────────────────────────────────────────
    st.markdown("### 1. Input mode")
    mode = st.radio(
        "Start from",
        ["Source .txt files (full pipeline)", "Saved interim notes (skip to Pass 2 Reduce)"],
        key="input_mode",
        label_visibility="collapsed",
        help=(
            "**Source files** — full pipeline: Pass 1 Map → Pass 1 Reduce → Pass 2 Map → Pass 2 Reduce → merge.\n\n"
            "**Saved interim notes** — load a previously-saved interim .txt (contains FACTBASE + Pass 2 Map output). "
            "Skips Pass 1 entirely and Pass 2 Map. Only Pass 2 Reduce + merge runs. Cheap and fast for iterating "
            "on Pass 2 section writing."
        ),
    )
    is_interim_mode = mode.startswith("Saved interim")

    # ── Inputs (conditional on mode) ───────────────────────────────────────────
    uploaded_ar_ip: List = []
    uploaded_transcripts: List = []
    interim_uploaded = None

    if is_interim_mode:
        st.markdown("### 2. Interim notes file")
        interim_uploaded = st.file_uploader(
            "Upload a saved interim notes .txt (from a previous run)",
            type=["txt"], accept_multiple_files=False,
            key="interim_uploader",
            help=(
                "The interim file contains the FACTBASE + the Pass 2 Map output from a previous run. "
                "Loading it lets you re-run Pass 2 Reduce with edited prompt or different length."
            ),
        )
        if interim_uploaded:
            size_kb = interim_uploaded.size / 1024
            st.info(f"✓ Interim file loaded: **{interim_uploaded.name}** ({size_kb:.1f} KB)")
    else:
        st.markdown("### 2. Annual Reports + Investor Presentations")
        uploaded_ar_ip = st.file_uploader(
            f"Upload AR + IP .txt files (up to {MAX_FILES_PER_BUCKET})",
            type=["txt"], accept_multiple_files=True,
            key="ar_ip_uploader",
            help="Pass 1 source. Annual reports and investor presentations only — no transcripts here.",
        )
        if uploaded_ar_ip:
            words = sum(len(f.getvalue().decode("utf-8", errors="replace").split()) for f in uploaded_ar_ip)
            st.caption(f"✓ {len(uploaded_ar_ip)} AR/IP file(s), ~{words:,} words total")

        st.markdown("### 3. Quarterly Call Transcripts")
        uploaded_transcripts = st.file_uploader(
            f"Upload transcript .txt files (up to {MAX_FILES_PER_BUCKET})",
            type=["txt"], accept_multiple_files=True,
            key="transcript_uploader",
            help="Pass 2 source. Earnings-call / management-meeting transcripts.",
        )
        if uploaded_transcripts:
            words = sum(len(f.getvalue().decode("utf-8", errors="replace").split()) for f in uploaded_transcripts)
            st.caption(f"✓ {len(uploaded_transcripts)} transcript file(s), ~{words:,} words total")

    # ── Pass 1 prompt (editable) ───────────────────────────────────────────────
    next_section_num = "4" if not is_interim_mode else "3"
    st.markdown(f"### {next_section_num}. Pass 1 prompt — FACT-BASE extraction (editable)")
    with st.expander("View / edit Pass 1 prompt (default shipped)", expanded=False):
        pass1_prompt = st.text_area(
            "Pass 1 prompt",
            value=DEFAULT_PASS_1_PROMPT,
            height=400,
            key="pass1_prompt",
            label_visibility="collapsed",
            help="The full Pass 1 prompt used during AR/IP processing. Edit if you want different fact categories or rules.",
        )
    if is_interim_mode:
        st.caption("*(Pass 1 prompt not used in interim mode — Pass 1 is skipped.)*")

    # ── Pass 2 prompt (editable) ───────────────────────────────────────────────
    next_section_num = str(int(next_section_num) + 1)
    st.markdown(f"### {next_section_num}. Pass 2 prompt — ANALYSIS NOTE reasoning (editable)")
    with st.expander("View / edit Pass 2 prompt (default shipped)", expanded=False):
        pass2_prompt = st.text_area(
            "Pass 2 prompt",
            value=DEFAULT_PASS_2_PROMPT,
            height=400,
            key="pass2_prompt",
            label_visibility="collapsed",
            help=(
                "The full Pass 2 prompt used for transcript processing. The placeholder "
                "`[PASTE company_factbase.md HERE]` will be auto-replaced with the factbase. "
                "If you remove it, the factbase is prepended as a labelled block."
            ),
        )

    # ── Length & split ─────────────────────────────────────────────────────────
    next_section_num = str(int(next_section_num) + 1)
    st.markdown(f"### {next_section_num}. Output length")
    length_label = st.radio(
        "Total document length (FACTBASE + ANALYSIS NOTE combined)",
        list(LENGTH_PRESETS.keys()),
        index=2,  # "Default (~8000 words)"
        horizontal=True,
        key="length_preset",
    )
    total_words = LENGTH_PRESETS[length_label]
    if total_words is None:
        total_words = st.number_input(
            "Custom total word count",
            min_value=1000, max_value=15000, value=8000, step=500,
            key="custom_wc",
        )

    pass1_share_pct = st.slider(
        "Pass 1 share (rest goes to Pass 2)",
        min_value=20, max_value=50, value=DEFAULT_PASS_SPLIT, step=5,
        key="pass1_share",
        help=(
            f"Default {DEFAULT_PASS_SPLIT}%: factbase is meant to be terse and tabular; analysis note carries the depth. "
            "Raise toward 40–50% if your factbase always comes out under-developed."
        ),
    )
    pass1_word_budget = int(total_words * pass1_share_pct / 100)
    pass2_word_budget = total_words - pass1_word_budget
    st.caption(
        f"📐 Pass 1 (FACTBASE): ~**{pass1_word_budget:,} words** ({pass1_share_pct}%)  •  "
        f"Pass 2 (ANALYSIS NOTE): ~**{pass2_word_budget:,} words** ({100 - pass1_share_pct}%)  •  "
        f"Total target ~{total_words:,} words"
    )

    # ── Pipeline scope (only in source-files mode) ─────────────────────────────
    stop_after = "Full pipeline (FACTBASE + ANALYSIS NOTE merged)"
    if not is_interim_mode:
        next_section_num = str(int(next_section_num) + 1)
        st.markdown(f"### {next_section_num}. Pipeline scope")
        stop_after = st.radio(
            "How far to run",
            [
                "Full pipeline (FACTBASE + ANALYSIS NOTE merged)",
                "Stop after Pass 1 (FACTBASE only)",
                "Stop after Pass 2 Map (FACTBASE + interim file, skip section writing)",
            ],
            index=0,
            key="stop_after",
            label_visibility="collapsed",
            help=(
                "**Full pipeline (default)** — Pass 1 Map → Pass 1 Reduce → Pass 2 Map → "
                "Pass 2 Reduce → merge.\n\n"
                "**Stop after Pass 1** — produces only the FACTBASE. Useful when you want to "
                "review or hand-edit the factbase before spending compute on Pass 2.\n\n"
                "**Stop after Pass 2 Map** — produces the FACTBASE and the Pass 2 Map output, "
                "saving them as an interim .txt file you can resume from later. Useful for "
                "auditing what was extracted from transcripts before committing to the 10 "
                "section-writing calls. The cheapest way to produce an interim file for iteration."
            ),
        )

    stop_after_pass1     = stop_after.startswith("Stop after Pass 1")
    stop_after_pass2_map = stop_after.startswith("Stop after Pass 2 Map")

    # ── Pre-flight cost estimate ───────────────────────────────────────────────
    # Approximate cost at current settings, shown before the user commits.
    # Source-files mode only — interim mode has different (much smaller) cost.
    if not is_interim_mode and (uploaded_ar_ip or uploaded_transcripts):
        ar_ip_words = sum(
            len(f.getvalue().decode("utf-8", errors="replace").split()) for f in (uploaded_ar_ip or [])
        )
        tx_words = sum(
            len(f.getvalue().decode("utf-8", errors="replace").split()) for f in (uploaded_transcripts or [])
        )
        map_model_id = MODELS.get(map_model_name, "gemini-2.5-flash")
        reduce_model_id = MODELS.get(reduce_model_name, "gemini-2.5-pro")

        est = estimate_pipeline_cost(
            ar_ip_words, tx_words, pass1_word_budget, pass2_word_budget,
            map_model_id, reduce_model_id,
        )

        # Scope cost to what will actually run given the stop_after selection
        if stop_after_pass1:
            run_cost = est["pass1_map_cost"] + est["pass1_reduce_cost"]
            run_desc = "Pass 1 Map + Pass 1 Reduce"
        elif stop_after_pass2_map:
            run_cost = est["pass1_map_cost"] + est["pass1_reduce_cost"] + est["pass2_map_cost"]
            run_desc = "Pass 1 (full) + Pass 2 Map"
        else:
            run_cost = est["total_cost"]
            run_desc = "Full pipeline (all 4 stages)"

        with st.expander(
            f"💰 Estimated cost for this run: **~${run_cost:.2f}**  ({run_desc})",
            expanded=False,
        ):
            st.caption(
                "Rough estimate based on input word count, expected chunks, and current model "
                "selection. Actual cost typically lands within ±30% of this. The biggest "
                "uncertainty is per-chunk output length — verbose extraction inflates Map cost. "
                "Section-routing in Reduce keeps the Reduce stages tight regardless of input size."
            )
            cost_lines = [
                "| Stage | Model | Est. cost (USD) |",
                "|---|---|---:|",
                f"| Pass 1 Map ({est['pass1_chunks']:,} chunks) | `{map_model_id}` | ${est['pass1_map_cost']:.4f} |",
                f"| Pass 1 Reduce (6 sections, routed) | `{reduce_model_id}` | ${est['pass1_reduce_cost']:.4f} |",
                f"| Pass 2 Map ({est['pass2_chunks']:,} chunks) | `{map_model_id}` | ${est['pass2_map_cost']:.4f} |",
                f"| Pass 2 Reduce (10 sections, routed) | `{reduce_model_id}` | ${est['pass2_reduce_cost']:.4f} |",
                f"| **Total (full pipeline)** | — | **${est['total_cost']:.4f}** |",
            ]
            st.markdown("\n".join(cost_lines))
            st.caption(
                f"Input scale: AR/IP {ar_ip_words:,} words → {est['pass1_chunks']:,} Pass 1 chunks; "
                f"Transcripts {tx_words:,} words → {est['pass2_chunks']:,} Pass 2 chunks."
            )

    # ── Generate ───────────────────────────────────────────────────────────────
    st.divider()
    button_label = "Generate research note"
    if stop_after_pass1:
        button_label = "Generate FACTBASE only"
    elif stop_after_pass2_map:
        button_label = "Generate FACTBASE + Pass 2 Map (save interim)"
    if st.button(button_label, type="primary", use_container_width=True):
        # Validation
        if is_interim_mode:
            if not interim_uploaded:
                st.error("Please upload a saved interim notes file.")
                st.stop()
            if not pass2_prompt.strip():
                st.error("Please provide a Pass 2 prompt.")
                st.stop()
        else:
            if not uploaded_ar_ip:
                st.error("Please upload at least one AR/IP .txt file (Pass 1 source).")
                st.stop()
            if not uploaded_transcripts:
                st.error("Please upload at least one transcript .txt file (Pass 2 source).")
                st.stop()
            if not pass1_prompt.strip() or not pass2_prompt.strip():
                st.error("Please provide both Pass 1 and Pass 2 prompts.")
                st.stop()
            if len(uploaded_ar_ip) > MAX_FILES_PER_BUCKET or len(uploaded_transcripts) > MAX_FILES_PER_BUCKET:
                st.error(f"Too many files in one bucket — limit is {MAX_FILES_PER_BUCKET}.")
                st.stop()

        # Decode files
        def _decode_files(files_uploaded):
            out = []
            for f in files_uploaded:
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

        ar_ip_files: List[Tuple[str, str]] = []
        transcript_files: List[Tuple[str, str]] = []
        factbase = ""
        pass2_map_output: List[str] = []

        if is_interim_mode:
            try:
                interim_text = interim_uploaded.getvalue().decode("utf-8", errors="replace")
                factbase, pass2_map_output, ar_ip_names, tx_names = parse_interim(interim_text)
                if not factbase or not pass2_map_output:
                    st.error("Could not parse the interim file — missing FACTBASE or PASS_2_MAP sections.")
                    st.stop()
                ar_ip_files = [(n, "") for n in ar_ip_names]
                transcript_files = [(n, "") for n in tx_names]
            except Exception as e:
                st.error(f"Could not load interim file: {e}")
                st.stop()
        else:
            ar_ip_files = _decode_files(uploaded_ar_ip)
            transcript_files = _decode_files(uploaded_transcripts)
            if not ar_ip_files or not transcript_files:
                st.error("Need at least one usable file in each bucket. Check filtering above.")
                st.stop()

        map_model    = get_model(map_model_name)
        reduce_model = get_model(reduce_model_name)
        st.session_state["usage_log"] = []  # fresh per run

        ar_ip_names      = [f[0] for f in ar_ip_files]
        transcript_names = [f[0] for f in transcript_files]

        # Track which stages actually ran so the output renderer can branch.
        # Set to one of: "pass1", "pass2_map", "full"
        stopped_after = "full"

        with st.status("Processing…", expanded=True) as status:
            try:
                # ── STAGE A: PASS 1 (only in source-files mode) ──────────────
                if not is_interim_mode:
                    st.write(f"**STAGE A — PASS 1 (FACT-BASE)** — {len(ar_ip_files)} AR/IP file(s)")
                    pass1_map_output = map_pass(
                        ar_ip_files, pass1_prompt,
                        CHUNK_SIZE_AR_IP, CHUNK_OVERLAP_AR_IP,
                        map_model, "Pass 1 Map", st.write,
                    )
                    if not pass1_map_output:
                        raise ValueError("Pass 1 Map produced no notes.")
                    st.write(
                        f"  ✓ Pass 1 Map complete — {len(pass1_map_output)} chunk extracts, "
                        f"~{sum(len(n.split()) for n in pass1_map_output):,} words"
                    )

                    factbase = reduce_pass1(
                        pass1_map_output, pass1_prompt, pass1_word_budget,
                        reduce_model, st.write,
                    )
                    if not factbase.strip():
                        raise ValueError("Pass 1 Reduce produced empty factbase.")
                    st.write(
                        f"  ✓ FACTBASE complete — **{len(factbase.split()):,} words** "
                        f"({len(factbase.split()) / pass1_word_budget * 100:.0f}% of {pass1_word_budget:,} budget)"
                    )
                else:
                    st.write(
                        f"⏩ Pass 1 skipped — factbase loaded from interim "
                        f"({len(factbase.split()):,} words)"
                    )

                # ── EARLY EXIT 1: stop after Pass 1 ─────────────────────────
                if stop_after_pass1:
                    stopped_after = "pass1"
                    # Persist results — no interim file (Pass 2 Map didn't run)
                    st.session_state["final_document"]       = "# FACT-BASE\n\n" + factbase.strip()
                    st.session_state["factbase"]             = factbase
                    st.session_state["analysis_note"]        = ""
                    st.session_state["interim_notes_text"]   = ""
                    st.session_state["ar_ip_filenames"]      = ar_ip_names
                    st.session_state["transcript_filenames"] = []
                    st.session_state["total_target"]         = pass1_word_budget
                    st.session_state["stopped_after"]        = stopped_after
                    status.update(label="Done — stopped after Pass 1 (FACTBASE only)", state="complete")
                    st.write(f"**✓ FACTBASE: {len(factbase.split()):,} words.** Pass 2 was skipped.")
                else:
                    # ── STAGE B (part 1): PASS 2 MAP (factbase as context) ──
                    if not is_interim_mode:
                        st.write(f"**STAGE B — PASS 2 (ANALYSIS NOTE)** — {len(transcript_files)} transcript file(s)")
                        pass2_prompt_with_factbase = inject_factbase(pass2_prompt, factbase)
                        pass2_map_output = map_pass(
                            transcript_files, pass2_prompt_with_factbase,
                            CHUNK_SIZE_TRANSCRIPT, CHUNK_OVERLAP_TRANSCRIPT,
                            map_model, "Pass 2 Map", st.write,
                        )
                        if not pass2_map_output:
                            raise ValueError("Pass 2 Map produced no notes.")
                        st.write(
                            f"  ✓ Pass 2 Map complete — {len(pass2_map_output)} chunk extracts, "
                            f"~{sum(len(n.split()) for n in pass2_map_output):,} words"
                        )
                    else:
                        st.write(
                            f"⏩ Pass 2 Map skipped — {len(pass2_map_output)} extracts loaded from interim"
                        )

                    # Save interim NOW (before Pass 2 Reduce) so user has it
                    st.session_state["interim_notes_text"] = serialize_interim(
                        factbase, pass2_map_output, ar_ip_names, transcript_names,
                    )

                    # ── EARLY EXIT 2: stop after Pass 2 Map ─────────────────
                    if stop_after_pass2_map:
                        stopped_after = "pass2_map"
                        st.session_state["final_document"]       = "# FACT-BASE\n\n" + factbase.strip()
                        st.session_state["factbase"]             = factbase
                        st.session_state["analysis_note"]        = ""
                        st.session_state["ar_ip_filenames"]      = ar_ip_names
                        st.session_state["transcript_filenames"] = transcript_names
                        st.session_state["total_target"]         = pass1_word_budget
                        st.session_state["stopped_after"]        = stopped_after
                        status.update(
                            label="Done — stopped after Pass 2 Map (interim ready)",
                            state="complete",
                        )
                        st.write(
                            f"**✓ FACTBASE: {len(factbase.split()):,} words. "
                            f"Pass 2 Map: {len(pass2_map_output)} extracts.** "
                            "Interim file ready to download below."
                        )
                    else:
                        # ── STAGE B (part 2): PASS 2 REDUCE ─────────────────
                        pass2_prompt_with_factbase = inject_factbase(pass2_prompt, factbase)
                        analysis_note = reduce_pass2(
                            pass2_map_output, pass2_prompt_with_factbase, pass2_word_budget,
                            reduce_model, st.write,
                        )
                        if not analysis_note.strip():
                            raise ValueError("Pass 2 Reduce produced empty analysis note.")
                        st.write(
                            f"  ✓ ANALYSIS NOTE complete — **{len(analysis_note.split()):,} words** "
                            f"({len(analysis_note.split()) / pass2_word_budget * 100:.0f}% of {pass2_word_budget:,} budget)"
                        )

                        # ── STAGE C: MERGE (mechanical) ─────────────────────
                        final_doc = (
                            "# FACT-BASE\n\n"
                            + factbase.strip()
                            + "\n\n---\n\n"
                            + "# ANALYSIS NOTE\n\n"
                            + analysis_note.strip()
                        )
                        st.session_state["final_document"]       = final_doc
                        st.session_state["factbase"]             = factbase
                        st.session_state["analysis_note"]        = analysis_note
                        st.session_state["ar_ip_filenames"]      = ar_ip_names
                        st.session_state["transcript_filenames"] = transcript_names
                        st.session_state["total_target"]         = total_words
                        st.session_state["stopped_after"]        = "full"

                        actual = len(final_doc.split())
                        pct = actual / total_words * 100
                        status.update(label="Done!", state="complete")
                        st.write(f"**✓ Final document: {actual:,} words ({pct:.0f}% of {total_words:,}-word target)**")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    if "final_document" in st.session_state:
        st.divider()
        doc            = st.session_state["final_document"]
        interim_text   = st.session_state.get("interim_notes_text", "")
        ar_ip_names    = st.session_state.get("ar_ip_filenames", [])
        tx_names       = st.session_state.get("transcript_filenames", [])
        total_target   = st.session_state.get("total_target", 0)
        stopped_after  = st.session_state.get("stopped_after", "full")

        # Adapt the title + caption + filenames to what was actually generated
        if stopped_after == "pass1":
            doc_title       = "FACT-BASE (Pass 1 only)"
            doc_subtitle    = "Pass 2 was skipped. The interim file is not available because Pass 2 Map did not run."
            md_filename     = "synthnotes_factbasenote_factbase.md"
            pdf_filename    = "synthnotes_factbasenote_factbase.pdf"
        elif stopped_after == "pass2_map":
            doc_title       = "FACT-BASE (Pass 2 Map saved to interim — section writing skipped)"
            doc_subtitle    = "Pass 2 section writing was skipped. The interim file below contains the factbase + Pass 2 Map output — load it via *Saved interim notes* mode to run only Pass 2 Reduce."
            md_filename     = "synthnotes_factbasenote_factbase.md"
            pdf_filename    = "synthnotes_factbasenote_factbase.pdf"
        else:
            doc_title       = "Research Note"
            doc_subtitle    = None
            md_filename     = "synthnotes_factbasenote.md"
            pdf_filename    = "synthnotes_factbasenote.pdf"

        col_title, col_copy, col_md, col_pdf = st.columns([2, 1, 1, 1])
        with col_title:
            actual = len(doc.split())
            st.subheader(f"{doc_title}  ({actual:,} words)")
            if doc_subtitle:
                st.caption(doc_subtitle)
        with col_copy:
            copy_button(doc, "Copy")
        with col_md:
            st.download_button(
                "Download .md",
                data=doc,
                file_name=md_filename,
                mime="text/markdown",
                use_container_width=True,
                key="dl_md",
            )
        with col_pdf:
            pdf_bytes = markdown_to_pdf_bytes(doc)
            if pdf_bytes:
                st.download_button(
                    "Download .pdf",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf",
                )
            else:
                st.caption("PDF unavailable — install `markdown` + `xhtml2pdf`")

        with st.expander(
            f"Source files used (AR/IP: {len(ar_ip_names)}, transcripts: {len(tx_names)})",
            expanded=False,
        ):
            st.markdown("**Annual reports + Investor presentations:**")
            for s in ar_ip_names:
                st.text(f"  • {s}")
            st.markdown("**Quarterly transcripts:**")
            for s in tx_names:
                st.text(f"  • {s}")

        if interim_text:
            with st.expander("💾 Save interim notes (FACTBASE + Pass 2 Map output)", expanded=False):
                st.caption(
                    "The interim file holds the FACTBASE and the Pass 2 Map output. "
                    "Download now — next time, switch **Input mode** to *Saved interim notes* "
                    "and upload this file to skip Pass 1 entirely and Pass 2 Map. Only Pass 2 "
                    "Reduce + merge will run. Useful for iterating on Pass 2 section writing "
                    "without re-paying Pass 1 + Pass 2 Map costs."
                )
                interim_chunk_count = max(0, interim_text.count(INTERIM_PASS2_CHUNK_SEP))
                st.caption(
                    f"Factbase + {interim_chunk_count} Pass 2 Map chunk(s) embedded · "
                    f"~{len(interim_text.split()):,} words total"
                )
                st.download_button(
                    "Download interim notes (.txt)",
                    data=interim_text,
                    file_name="synthnotes_factbasenote_interim.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_interim",
                )

        st.markdown(doc)

    render_usage_panel()


if __name__ == "__main__":
    main()
