"""
SynthNotes MultiDoc — multi-file note synthesis with user-provided prompt.

Workflow:
  1. User uploads N .txt files (up to MAX_FILES)
  2. User provides a custom prompt describing the notes they want
  3. User selects target output length
  4. Map stage: each file is chunked; each chunk is processed with the user prompt
     (cheap model, parallelised across all chunks of all files)
  5. Reduce stage: all per-chunk notes are synthesised into one document, with
     chronology inferred from content cues, hitting the user's target length
     (quality model, single call leveraging Gemini's large context)

Borrows the chunking, retry, streaming, and cost-tracking logic from SynthNotes-Pro;
this is a separate self-contained app (no imports from sibling apps).
"""

import streamlit as st
import google.generativeai as genai
import os, re, time, json, html as html_module
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


# ── 4. CORE PROCESSING (map-reduce) ────────────────────────────────────────────

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

INTERIM_FILE_HEADER = "==== SynthNotes MultiDoc — Interim Notes ===="
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

def markdown_to_pdf_bytes(md_text: str, title: str = "SynthNotes MultiDoc Output") -> Optional[bytes]:
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
    st.set_page_config(page_title="SynthNotes MultiDoc", layout="wide", page_icon="📚")
    st.title("📚 SynthNotes MultiDoc")
    st.caption(
        "Multi-file note synthesis. Upload many .txt files, provide your own prompt, "
        "get one consolidated document with chronology inferred from content. "
        "Borrows the chunking and retry logic from SynthNotes Pro."
    )
    api_key_check()

    # ── Sidebar: model settings ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Model Settings")
        map_model_name = st.selectbox(
            "Per-section model (Map stage)",
            list(MODELS.keys()), index=0,
            key="map_model",
            help=(
                "Used to extract notes from each chunk. Cheap, fast models work well here "
                "since the work is mechanical extraction, not synthesis."
            ),
        )
        reduce_model_name = st.selectbox(
            "Synthesis model (Reduce stage)",
            list(MODELS.keys()), index=2,
            key="reduce_model",
            help=(
                "Used to combine all per-chunk notes into the final document and infer "
                "chronology. Higher-quality model recommended (2.5 Pro is the default)."
            ),
        )

    # ── Mode toggle ────────────────────────────────────────────────────────────
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
    uploaded: List = []
    interim_uploaded = None
    combine_files = False

    if not is_interim_mode:
        st.markdown("### 2. Source files")
        uploaded = st.file_uploader(
            f"Upload up to {MAX_FILES} text files (.txt)",
            type=["txt"], accept_multiple_files=True,
            help="Files are processed in parallel. Each file is chunked individually then combined.",
            key="source_uploader",
        )
        if uploaded:
            total_words = 0
            with st.expander(f"✓ {len(uploaded)} file(s) loaded — view list", expanded=False):
                for f in uploaded:
                    try:
                        sample = f.getvalue().decode("utf-8", errors="replace")
                        wc = len(sample.split())
                        total_words += wc
                        st.text(f"  • {f.name}  ({f.size/1024:.1f} KB, ~{wc:,} words)")
                    except Exception:
                        st.text(f"  • {f.name}  ({f.size/1024:.1f} KB)")
            st.caption(f"Total input: ~{total_words:,} words across {len(uploaded)} file(s).")

        combine_files = st.checkbox(
            "Concatenate all files into one before processing",
            value=False,
            help=(
                "**Off (default)** — process each file separately, then synthesise. Best when "
                "files have distinct identities (e.g. quarterly call transcripts, separate interviews).\n\n"
                "**On** — join all files into one large text blob, then chunk and process. Better "
                "when files are arbitrary fragments of one larger source."
            ),
        )
    else:
        st.markdown("### 2. Interim notes file")
        interim_uploaded = st.file_uploader(
            "Upload a saved interim notes .txt (from a previous run)",
            type=["txt"], accept_multiple_files=False,
            key="interim_uploader",
            help=(
                "The interim file is what you downloaded from a previous output as "
                "**'Download interim notes (.txt)'**. Loading it skips the expensive Map stage "
                "and lets you re-run synthesis with a different prompt or length."
            ),
        )
        if interim_uploaded:
            size_kb = interim_uploaded.size / 1024
            st.info(f"✓ Interim file loaded: **{interim_uploaded.name}** ({size_kb:.1f} KB)")

    # ── User prompt ────────────────────────────────────────────────────────────
    st.markdown("### 3. Your prompt")
    st.caption(
        "Describe the notes you want. This prompt is used in **both** stages — it tells the "
        "map stage what to extract from each section, and it tells the reduce stage what the "
        "final document should look like."
    )
    user_prompt = st.text_area(
        "Prompt",
        height=200,
        key="user_prompt",
        placeholder=(
            "e.g. 'Generate a research note tracking how management commentary on margins "
            "has evolved across these earnings calls. For each quarter: stated guidance, actual "
            "margin reported, and forward-looking commitments. Flag any inconsistencies between "
            "quarters. Group by quarter chronologically.'"
        ),
        label_visibility="collapsed",
    )

    # ── Length & chunk size ────────────────────────────────────────────────────
    st.markdown("### 4. Output length")
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

    # ── Process ────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("Generate consolidated document", type="primary", use_container_width=True):
        if not user_prompt.strip():
            st.error("Please provide a prompt.")
            st.stop()

        # ── Branch on mode: gather (all_notes, filenames) ─────────────────────
        all_notes: List[str] = []
        filenames: List[str] = []
        files: List[Tuple[str, str]] = []

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
            if not uploaded:
                st.error("Please upload at least one .txt file.")
                st.stop()
            if len(uploaded) > MAX_FILES:
                st.error(f"Too many files — limit is {MAX_FILES}.")
                st.stop()

            # Read all file contents (UTF-8 with BOM fallback)
            for f in uploaded:
                size_mb = f.size / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"`{f.name}`: file too large ({size_mb:.1f} MB; limit {MAX_FILE_SIZE_MB} MB).")
                    st.stop()
                data = f.getvalue()
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        content = data.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        try:
                            content = data.decode("latin-1")
                        except Exception as e:
                            st.error(f"`{f.name}`: could not decode ({e}).")
                            st.stop()
                if not content.strip():
                    st.warning(f"`{f.name}` is empty — skipping.")
                    continue
                files.append((f.name, content))

            if not files:
                st.error("No usable file content after filtering. Please upload non-empty .txt files.")
                st.stop()
            filenames = [f[0] for f in files]

        map_model    = get_model(map_model_name)
        reduce_model = get_model(reduce_model_name)

        # Fresh usage log per run
        st.session_state["usage_log"] = []

        with st.status("Processing…", expanded=True) as status:
            try:
                # ── Map stage (only in source-files mode) ─────────────────────
                if not is_interim_mode:
                    if combine_files:
                        st.write(f"📎 Concatenating {len(files)} file(s) into one combined input…")
                        combined_content = "\n\n".join(
                            f"=== {name} ===\n{content}" for name, content in files
                        )
                        files = [("(combined input)", combined_content)]
                        filenames = ["(combined input)"]

                    # Build a flat task list across all chunks of all files
                    tasks: List[Tuple[int, int, int, str, str]] = []
                    for file_idx, (filename, content) in enumerate(files, start=1):
                        chunks = create_chunks_with_overlap(content, chunk_size, overlap)
                        for i, chunk_text in enumerate(chunks, start=1):
                            tasks.append((file_idx, i, len(chunks), filename, chunk_text))

                    st.write(
                        f"📊 Map stage: **{len(tasks)} section(s)** across **{len(files)} file(s)**. "
                        f"Processing in parallel with {PARALLEL_WORKERS} workers."
                    )

                    # Process all chunks in parallel, preserve original order
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
                            st.write(f"  • Map: {done}/{len(tasks)} sections complete")

                    all_notes = [r for r in results if r and r.strip()]
                    if not all_notes:
                        raise ValueError("Map stage produced no notes — check input files and prompt.")

                    map_words = sum(len(n.split()) for n in all_notes)
                    st.write(
                        f"✓ Map stage complete — {len(all_notes)} section notes, "
                        f"~{map_words:,} words of intermediate content"
                    )
                else:
                    interim_words = sum(len(n.split()) for n in all_notes)
                    st.write(
                        f"⏩ Skipping Map stage — using {len(all_notes)} pre-extracted section notes "
                        f"(~{interim_words:,} words) from interim file"
                    )

                # ── Save interim notes to session state (for download) ────────
                st.session_state["interim_notes_text"] = serialize_interim(all_notes, filenames)

                # ── Reduce stage (hierarchical — handles arbitrary input size) ─
                final_doc = hierarchical_reduce(
                    all_notes, filenames, user_prompt, word_count,
                    reduce_model, st.write, depth=0,
                )
                if not final_doc.strip():
                    raise ValueError("Synthesis returned empty output.")

                st.session_state["final_document"]   = final_doc
                st.session_state["source_filenames"] = filenames
                st.session_state["target_words"]     = word_count

                actual_words = len(final_doc.split())
                pct_of_target = actual_words / word_count * 100
                status.update(label="Done!", state="complete")
                st.write(
                    f"✓ Final document: **{actual_words:,} words** "
                    f"({pct_of_target:.0f}% of {word_count:,}-word target)"
                )

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    if "final_document" in st.session_state:
        st.divider()
        doc          = st.session_state["final_document"]
        sources      = st.session_state.get("source_filenames", [])
        target       = st.session_state.get("target_words", 0)
        interim_text = st.session_state.get("interim_notes_text", "")

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
                file_name="synthnotes_multidoc_output.md",
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
                    file_name="synthnotes_multidoc_output.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_final_pdf",
                )
            else:
                st.caption("PDF unavailable — install `markdown` and `xhtml2pdf` (see requirements.txt)")

        with st.expander(f"Source documents used ({len(sources)})", expanded=False):
            for s in sources:
                st.text(f"  • {s}")

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
                    file_name="synthnotes_multidoc_interim.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_interim",
                )

        st.markdown(doc)

    # ── Cost panel ─────────────────────────────────────────────────────────────
    render_usage_panel()


if __name__ == "__main__":
    main()
