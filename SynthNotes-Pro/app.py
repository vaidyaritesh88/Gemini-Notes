import streamlit as st
import google.generativeai as genai
import os, io, re, time, tempfile, json, html as html_module, subprocess, glob
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import PyPDF2
import streamlit.components.v1 as components

# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY", "")
if _api_key:
    genai.configure(api_key=_api_key)

CHUNK_WORD_SIZE    = 4000
CHUNK_WORD_OVERLAP = 400
INTEL_CHUNK_SIZE   = 4000   # intelligence extraction chunk size (notes are input, already dense)
INTEL_OVERLAP      = 400
MAX_OUTPUT_TOKENS  = 65536
MAX_PDF_MB         = 25
MAX_AUDIO_MB       = 200

MODELS = {
    "Gemini 2.5 Flash (Fast)":       "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite (Cheap)": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro (Best)":         "gemini-2.5-pro",
    "Gemini 3.0 Flash":              "gemini-3-flash-preview",
    "Gemini 2.0 Flash":              "gemini-2.0-flash-lite",
    "Gemini 1.5 Flash":              "gemini-1.5-flash",
}

MEETING_TYPES = ["Expert Meeting", "Management Meeting", "Internal Discussion"]

SUMMARY_PRESETS = {
    "Short (~500 words)":      500,
    "Standard (~1000 words)": 1000,
    "Detailed (~1500 words)": 1500,
    "Full (~2500 words)":     2500,
    "Custom":                 None,
}

REFINEMENT_INSTRUCTIONS = {
    "Expert Meeting":      "Pay special attention to industry jargon, technical terms, company names, and domain-specific terminology. Preserve all proper nouns exactly.",
    "Management Meeting":  "Pay special attention to names of attendees, action item owners, project names, deadlines, and organizational terminology.",
    "Internal Discussion": "Pay special attention to participant names, project/product names, technical terms, and any referenced documents or systems.",
}

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_prompts_pro.json")


# ── 2. NOTES PROMPTS ───────────────────────────────────────────────────────────
# These are identical to the Lite app — proven, high-quality capture prompts.
# The Pro difference comes entirely from the intelligence extraction layer below.

EXPERT_MEETING_DETAILED_PROMPT = """### **PRIMARY DIRECTIVE: MAXIMUM DETAIL & STRICT COMPLETENESS**
Your goal is to produce the most thorough, granular notes possible. Remove conversational filler ("um," "you know," repetition) but **nothing substantive should be omitted.** Every factual claim, example, explanation, aside, and data point in the transcript must appear in your notes. When in doubt, INCLUDE it. Err heavily on the side of over-inclusion. Longer, more detailed notes are always preferred over concise ones.

### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles, affiliations, years of experience, roles). Use simple, direct language.
- **DO NOT:** Summarize or include introductions about consulting firms.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body STRICTLY in Question/Answer format.

**(2.A) Questions:**
-   Identify the core question being asked and rephrase it clearly in **bold**. Do NOT copy the question verbatim from the transcript — clean up filler, false starts, and rambling phrasing into a clear, well-formed question that preserves the original intent.
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or any similar label. The bold question text stands alone.
-   If the questioner provides context, framing, or a multi-part question, capture the full scope — do not reduce a multi-part question to a single line.
-   **LONG QUESTIONS / PREAMBLE:** Sometimes a question is long because the interviewer provides substantial framing, background, or context before asking — this preamble is part of the question and must be preserved as part of the bold question text. Do NOT treat the preamble as part of the answer.
-   **SPACING:** Leave exactly one blank line between the end of one answer and the start of the next bold question, so each Q&A pair is visually separated.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question (no blank line between the bold question and its first bullet).
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   Use **multiple bullet points** per answer — do NOT collapse a detailed response into a single bullet.
-   **ZERO SKIPPING RULE:** If the expert said it with substance, it must appear in your notes. Do NOT skip examples, anecdotes, specific sentences, or supporting details even if they seem minor or repetitive. Every distinct point gets its own bullet. If an answer contains 8 substantive points, you must produce at least 8 bullets — never condense them into 3-4.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, specific entities mentioned, time periods, market sizes, growth rates, company names, product names, and geographies.
-   **PRIORITY #2: CAPTURE ALL NUANCE & REASONING.** Do not over-summarize or reduce complex answers to surface-level statements. You must retain the following:
    -   **Sentiment & Tone:** Note if the expert is confident, uncertain, speculative, cautious, or enthusiastic (e.g., "The expert was highly confident that...," "He cautioned that...").
    -   **Qualifiers & Conditions:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "except in," "only when," "roughly," "approximately," "a potential risk is...").
    -   **Key Examples & Analogies:** If the expert uses a specific example, anecdote, case study, or analogy to illustrate a point, capture it in full, even if it spans multiple sentences — these are often the most valuable parts of an expert call.
    -   **Cause & Effect:** Retain any reasoning chains provided (e.g., "...because of regulatory changes," "...which led to a 15% decline in...").
    -   **Comparisons & Contrasts:** If the expert compares companies, products, approaches, or time periods, capture both sides of the comparison with the specific details for each.
    -   **Tangential but relevant points:** If the expert volunteers additional context, background, or related information beyond the direct question, include it — do NOT discard it as off-topic.
-   **PRIORITY #3: PRESERVE MULTI-STEP EXPLANATIONS.** If an answer involves a sequence of steps, a timeline, or a logical chain, preserve the full sequence rather than summarizing the conclusion only."""

EXPERT_MEETING_CONCISE_PROMPT = """### **PRIMARY DIRECTIVE: EFFICIENT & NUANCED**
Your goal is to be **efficient**, not just brief. Efficiency means removing conversational filler ("um," "you know," repetition) but **preserving all substantive information**. Your output should be concise yet information-dense.

### **NOTES STRUCTURE**

**(1.) Opening overview or Expert background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or expert intro, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, dates, numbers, titles).
- **DO NOT:** Summarize.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body in Question/Answer format.

**(2.A) Questions:**
-   Identify the core question being asked and rephrase it clearly in **bold**. Do NOT copy verbatim from the transcript — clean up filler and rambling into a clear, well-formed question.
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or any similar label.
-   **LONG QUESTIONS / PREAMBLE:** Preserve substantive framing as part of the bold question text.
-   **SPACING:** Leave exactly one blank line between the end of one answer and the start of the next bold question.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question.
-   Each bullet must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** Numbers, percentages, company names, metrics, specific entities.
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Sentiment, qualifiers, key examples, cause & effect chains."""

MANAGEMENT_MEETING_PROMPT = """### **NOTES STRUCTURE: MANAGEMENT MEETING**

Structure the notes to capture decisions, action items, and key discussion points.

**(1.) Meeting Overview (Conditional):**
- If the transcript begins with an agenda or introductions, capture attendees, date, and agenda items as bullet points.

**(2.) Discussion Topics:**
Structure the body by topic/agenda item using **bold headings**.

For each topic:
- **Key Points:** Bullet-point the main arguments, data, and perspectives shared.
- **Decisions Made:** Clearly state any decisions reached, who made them, and the rationale.
- **Action Items:** List each action item with the responsible person and any stated deadline.
- **Open Questions:** Note unresolved issues or items deferred for follow-up.

**PRIORITY #1: CAPTURE ALL DECISIONS AND ACTION ITEMS.** These are the most critical outputs.
**PRIORITY #2: CAPTURE ALL DATA.** Names, numbers, dates, metrics, and specific references.
**PRIORITY #3: PRESERVE CONTEXT.** Include the reasoning behind decisions and any dissenting views."""

INTERNAL_DISCUSSION_PROMPT = """### **NOTES STRUCTURE: INTERNAL DISCUSSION**

Structure the notes to capture the flow of ideas, key arguments, and conclusions. This is NOT a Q&A format — focus on capturing the substance of the discussion as it evolved.

**(1.) Discussion Context (Conditional):**
- If the discussion has a stated purpose or background, capture it as bullet points at the top.

**(2.) Discussion Flow:**
Structure the body by topic or theme using **bold headings**.

For each topic:
- Capture each participant's key contributions and perspectives as bullet points.
- Note areas of agreement and disagreement.
- Highlight any data, examples, or evidence cited.
- Flag any concerns, risks, or caveats raised.

**(3.) Conclusions & Next Steps:**
- Summarize any conclusions reached.
- List follow-up items or next steps with owners if identified.

**PRIORITY #1: CAPTURE ALL PERSPECTIVES.** Include different viewpoints even if they disagree.
**PRIORITY #2: CAPTURE ALL DATA.** Names, numbers, references, and specific examples.
**PRIORITY #3: PRESERVE REASONING.** Include the "why" behind opinions and conclusions."""

PROMPT_INITIAL = """You are a High-Fidelity Factual Extraction Engine. Your task is to analyze a meeting transcript chunk and generate detailed, factual notes.
Your primary directive is **100% completeness and accuracy**. Process the transcript sequentially and generate notes following the structure below.
---
{base_instructions}
---
**MEETING TRANSCRIPT CHUNK:**
{chunk_text}
"""

PROMPT_CONTINUATION = """You are a High-Fidelity Factual Extraction Engine continuing a note-taking task from a long transcript.

### **CONTEXT FROM PREVIOUS PROCESSING**
Below is a summary of the notes generated from the previous transcript chunk. Use this to understand the flow of the conversation.
{context_package}

### **CONTINUATION INSTRUCTIONS**
1.  **PROCESS THE ENTIRE CHUNK:** Your task is to process the **entire** new transcript chunk provided below. Every substantive point, example, data point, and nuanced opinion in this chunk MUST appear in your output.
2.  **HANDLE OVERLAP:** The beginning of this new chunk overlaps with the end of the previous one. Process it naturally. Your output will be automatically de-duplicated later.
3.  **MAINTAIN FORMAT:** Continue to use the exact same formatting as established in the base instructions.
4.  **NO META-COMMENTARY:** NEVER produce statements about the transcript itself, such as "the transcript does not contain an answer," "no relevant information in this section," etc. Always extract and document whatever substantive content exists.
5.  **MID-CHUNK STARTS:** If the chunk starts in the middle of a response, begin your notes by capturing that content under the most relevant heading from context. Do not skip or discard partial content.
6.  **MAINTAIN OUTPUT VOLUME:** This chunk contains the same amount of content as the first chunk. Your output for this chunk MUST be equally detailed and equally long. Do NOT taper off, summarize, or become briefer.

---
{base_instructions}
---

**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}
"""


# ── 3. INTELLIGENCE EXTRACTION PROMPTS ─────────────────────────────────────────
# The intelligence layer classifies and organises what was actually said.
# STRICT RULE: Every point must trace back to something stated in the notes.
# No inferences, no "so what", no AI-added interpretations — purely factual classification.

INTEL_EXPERT_PROMPT = """You are a research analyst extracting structured intelligence from expert consultation notes.

**CRITICAL RULE: Only include what the expert actually said. Do not infer, interpret, or add your own analysis. Every point must be traceable to a specific statement in the notes.**

---

## EXPERT BACKGROUND
[Only if the notes contain bio/background information about the expert. Skip this section entirely if no background is present in the notes.]

## CORE THESIS
The expert's 2–3 overarching views as stated in the call — their headline positions on the main subject. Use the expert's own framing where possible.

## KEY INSIGHTS
The most important, substantive things the expert said — prioritising non-obvious points over general context. Each bullet should be a specific, complete statement. Aim for 6–10 bullets.
- Do NOT include background context or widely-known facts unless the expert made a specific claim about them.
- Do NOT collapse multiple distinct points into one bullet.

## HARD DATA & FACTS
Every specific figure, statistic, or named reference the expert provided:
- Numbers, percentages, growth rates, market sizes, price points, ratios
- Named companies, products, geographies with specific context
- Dates, timelines, durations
Format: one bullet per data point with just enough context to understand it.

## DIRECT QUOTES
2–4 verbatim sentences from the expert that best capture their views. Choose the most specific and quotable lines.
Format: *"[exact quote]"* — [brief topic label]

## EXPRESSED RISKS & UNCERTAINTIES
Things the expert themselves flagged as risks, concerns, downside scenarios, or areas of uncertainty. Note their stated confidence level where apparent (e.g., "the expert cautioned that...", "he was uncertain about...").
Only include risks the expert actually raised — do not add risks of your own.

## STATED NON-CONSENSUS VIEWS
Views the expert themselves described as contrarian, surprising, or different from conventional wisdom — or views that are clearly at odds with a commonly-held position as stated in the notes.
Leave this section blank if the notes contain no such views.

## QUESTIONS & UNCERTAINTIES THE EXPERT RAISED
Open questions, unresolved issues, or areas the expert themselves said need further investigation or monitoring. Only include things the expert explicitly flagged — not questions you think are interesting.

---
MEETING NOTES:
{notes}
"""

INTEL_MANAGEMENT_PROMPT = """You are extracting structured intelligence from management meeting notes.

**CRITICAL RULE: Only include what was actually said or decided in the meeting. Do not infer, interpret, or add your own analysis.**

---

## MEETING CONTEXT
1–2 sentences stating the purpose of this meeting and its overall outcome, as described in the notes.

## DECISIONS MADE
Each decision reached in the meeting. Be specific — not "discussed the budget" but "approved Q3 budget at $2.4M."
Format: **[Decision]** — Rationale: [the stated reason, if given in the notes]
If no rationale was stated, omit that part.

## ACTION ITEMS
Each action item as recorded in the notes.
Format: **[Owner]** | [Specific action] | Due: [date or timeframe stated, or "Not stated"]

## KEY DEBATES
Topics that were contested in the meeting. For each: what was the question, what positions were expressed, and how it was resolved (or note if it was not resolved).

## UNRESOLVED ISSUES
Decisions deferred, open questions, or items explicitly flagged for follow-up in the meeting.

## KEY DATA & FACTS REFERENCED
Important numbers, metrics, targets, or facts cited during the meeting that informed discussion or decisions.

---
MEETING NOTES:
{notes}
"""

INTEL_INTERNAL_PROMPT = """You are extracting structured intelligence from internal discussion notes.

**CRITICAL RULE: Only include what was actually said in the discussion. Do not infer, interpret, or add your own analysis.**

---

## DISCUSSION GOAL
What this discussion was trying to achieve or decide, as stated in the notes.

## VIEWS & POSITIONS EXPRESSED
The main arguments, positions, or perspectives raised by participants. Capture the substance — what were people actually saying and why? Attribute views to individuals only where the notes do so.

## WHERE THERE WAS AGREEMENT
Points where the group converged or reached shared understanding, as recorded in the notes.

## WHERE THERE WAS DISAGREEMENT
Points of contention as they appear in the notes. What was the nature of the disagreement?

## CONCLUSIONS REACHED
What was actually concluded or agreed upon, as stated in the notes. Be precise about what was settled vs. what remains open.

## NEXT STEPS
Actions or decisions to happen next, as stated in the notes. Include owners and timelines if the notes record them.

---
MEETING NOTES:
{notes}
"""

INTEL_SYNTHESIS_PROMPT = """You are combining intelligence extracts from multiple sections of the same {meeting_type} notes into one unified brief.

**CRITICAL RULE: Only include what was in the original notes. Do not add analysis or inferences.**

Merge the section extracts by:
1. Consolidating duplicate or overlapping points — keep the most complete, specific version
2. Preserving all unique content across sections
3. Maintaining the same section structure as the input
4. For Expert Meeting: keeping the best 2–4 direct quotes total
5. Not adding any information not present in the source extracts

Output the final unified intelligence brief in the same structured format.

---
SECTION EXTRACTS:
{extracts}
"""


# ── 4. SUMMARY PROMPTS ─────────────────────────────────────────────────────────
# Two-tier output: a brief (~100-150 words) followed by a structured detailed summary.
# STRICT RULE: The summary must only contain what is in the intelligence brief.
# No inferences, no implications, no AI-added analysis.

SUMMARY_EXPERT_PROMPT = """You are writing a factual summary of an expert consultation call for a professional reader.

**CRITICAL RULE: Only include information that appears in the intelligence brief below. Do not add your own analysis, inferences, or interpretations. If something is not in the brief, do not include it.**

Target: approximately {word_count} words for the Detailed Summary. The Brief should be ~150 words.
{focus_block}
---

## BRIEF
[~150 words. A concise factual overview: who the expert is (if background is in the brief), the main subject of the call, and the expert's core position. Flowing prose, no headers or bullets. Written so it stands alone as a quick reference.]

---

## DETAILED SUMMARY

### Expert Background
[Only if the intelligence brief contains background information. Skip entirely if not present.]

### Core Views
[The expert's main stated positions. 4–6 bullets. Start each bullet with the substance. Use the expert's own framing where captured in the brief.]

### Key Evidence & Data
[Specific facts, numbers, company references, and examples from the brief. One bullet per distinct data point. Include the context that was stated — not context you are adding.]

### Risks & Uncertainties
[Only risks and uncertainties the expert themselves raised. Preserve their stated level of confidence. Do not add risks they did not mention.]

### Non-Consensus Views
[Only views the expert described as contrarian or that the brief identifies as non-consensus. Skip this section if the brief contains none.]

### Direct Quotes
[The verbatim quotes from the brief. Do not paraphrase — use exact wording.]

---
INTELLIGENCE BRIEF:
{intelligence}
"""

SUMMARY_MANAGEMENT_PROMPT = """You are writing a factual summary of a management meeting.

**CRITICAL RULE: Only include information from the intelligence brief below. Do not add your own analysis or inferences.**

Target: approximately {word_count} words for the Detailed Summary. The Brief should be ~100 words.
{focus_block}
---

## BRIEF
[~100 words. States the purpose of the meeting, the key decisions made, and the most critical action items — all from the brief. Prose format, no headers.]

---

## DETAILED SUMMARY

### Meeting Purpose & Context
[As stated in the intelligence brief.]

### Decisions Made
[Each decision from the brief, numbered. Include the stated rationale where the brief records one.]

### Action Items

| Owner | Action | Deadline |
|-------|--------|----------|
[Populated from the action items in the brief. "Not stated" for any deadlines not recorded.]

### Key Debates
[As recorded in the brief — what was contested, what positions were held, how it was resolved.]

### Open Issues & Next Steps
[Deferred decisions and next steps as stated in the brief.]

---
INTELLIGENCE BRIEF:
{intelligence}
"""

SUMMARY_INTERNAL_PROMPT = """You are writing a factual summary of an internal team discussion.

**CRITICAL RULE: Only include information from the intelligence brief below. Do not add your own analysis or inferences.**

Target: approximately {word_count} words for the Detailed Summary. The Brief should be ~100 words.
{focus_block}
---

## BRIEF
[~100 words. States what the discussion was about, the key conclusion, and the most important next step — all from the brief. Prose format, no headers.]

---

## DETAILED SUMMARY

### Context & Goal
[As stated in the intelligence brief.]

### Key Arguments & Positions
[The views expressed as recorded in the brief. Do not add attribution not present in the brief.]

### Where There Was Agreement
[As stated in the brief.]

### Where There Was Disagreement
[As stated in the brief.]

### Conclusions & Next Steps
[What was concluded and what happens next, as recorded in the brief.]

---
INTELLIGENCE BRIEF:
{intelligence}
"""

SUMMARY_REFINEMENT_PROMPT = """You are revising a meeting summary based on a user instruction.

**CRITICAL RULE: Only use information from the intelligence brief. Do not add information not present there. Do not add inferences or interpretations.**

CURRENT SUMMARY:
{current_summary}

---
INTELLIGENCE BRIEF (source of truth — do not add anything not present here):
{intelligence}

---
USER INSTRUCTION:
{instruction}

Apply the instruction and return the complete revised summary. Maintain the two-tier structure (## BRIEF and ## DETAILED SUMMARY). Only modify what the instruction asks for. Preserve all factual accuracy."""


# ── 4b. ANALYSIS PROMPT ────────────────────────────────────────────────────────
# The analysis page is the ONLY place where inferences are permitted.
# Everything here is clearly labelled as AI analysis, not meeting facts.

ANALYSIS_PROMPT = """You are an analyst helping a professional think through the implications of a meeting.

**CONTEXT:**
- The Intelligence Brief below contains only facts from the meeting — what was actually said.
- Your job is to analyse those facts and answer the user's question with your own reasoning.
- Clearly separate facts you are drawing on from the inferences you are making.

**FORMAT YOUR RESPONSE AS FOLLOWS:**
1. **Facts I am drawing on** (brief — pull the specific points from the brief that are relevant to the question)
2. **Analysis & Inferences** (your reasoning — clearly marked as your interpretation, not meeting facts)
3. **Caveats** (what would need to be true for your analysis to hold, or what information is missing)

**IMPORTANT:**
- Do not present inferences as facts.
- Use phrases like "This suggests...", "A possible interpretation is...", "One reading of this is..." to signal inference.
- If the brief does not contain enough information to answer the question well, say so clearly.

---
INTELLIGENCE BRIEF (facts from the meeting — your source material):
{intelligence}

---
QUESTION / ANALYSIS REQUEST:
{question}
"""

# Preset questions per meeting type — gives users a useful starting point
ANALYSIS_PRESETS = {
    "Expert Meeting": [
        "What are the key risks to the expert's thesis that they may be underweighting?",
        "What assumptions is the expert making that are not explicitly stated?",
        "How does this expert's view differ from a consensus or bullish/bearish market view?",
        "What follow-up questions would stress-test the expert's position?",
        "Based on what the expert said, what would need to change for their view to be wrong?",
    ],
    "Management Meeting": [
        "What risks or second-order effects of the decisions made may not have been considered?",
        "Are there any tensions between the decisions made and the open issues left unresolved?",
        "What assumptions are embedded in the action items that could cause them to fail?",
        "Which unresolved issues are most likely to resurface and cause problems?",
    ],
    "Internal Discussion": [
        "What underlying tensions in the disagreements might not have been fully aired?",
        "What assumptions does the group seem to be making that are not explicitly challenged?",
        "Are the conclusions reached well-supported by the arguments made in the discussion?",
        "What is the group not talking about that might be relevant?",
    ],
}


# ── 5. SAVED PROMPTS HELPERS ──────────────────────────────────────────────────

def load_saved_prompts() -> dict:
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("default", None)
                data.setdefault("prompts", {})
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"default": None, "prompts": {}}


def write_saved_prompts(data: dict):
    try:
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        st.error(f"Could not save prompts file: {e}")


# ── 6. CORE UTILITIES ──────────────────────────────────────────────────────────

def get_model(display_name: str) -> genai.GenerativeModel:
    cache = st.session_state.setdefault("_model_cache", {})
    model_id = MODELS.get(display_name, "gemini-2.5-flash")
    if model_id not in cache:
        cache[model_id] = genai.GenerativeModel(model_id)
    return cache[model_id]


def generate_with_retry(model, prompt, max_retries: int = 3, stream: bool = False, generation_config=None):
    kwargs = {"stream": stream}
    if generation_config:
        kwargs["generation_config"] = generation_config
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt, **kwargs)
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


def create_context_from_notes(notes_text: str, chunk_number: int) -> str:
    if not notes_text.strip():
        return ""
    headings = re.findall(r"(\*\*.*?\*\*)", notes_text)
    if not headings:
        return ""
    recent = headings[-3:] if len(headings) >= 3 else headings
    parts = [
        f"**Chunk #{chunk_number} Context Summary:**",
        f"- Sections processed so far: {len(headings)}",
        f"- Recent topics: {', '.join(h.strip('*') for h in recent[-2:])}",
        f"- Last section: {headings[-1]}",
    ]
    match = re.search(re.escape(headings[-1]) + r"(.*?)(?=\*\*|$)", notes_text, re.DOTALL)
    if match:
        parts.append(f"- Last section content:\n{match.group(1).strip()[:300]}...")
    return "\n".join(parts)


def cleanup_stitched_notes(text: str) -> str:
    if not text or not text.strip():
        return text
    artifact_patterns = [
        r'^[\-\*]*\s*(?:The|This)\s+(?:transcript|section|chunk|portion)\s+(?:does not|doesn\'t|appears to)\s+.*$',
        r'^[\-\*]*\s*(?:No relevant|No additional|No further|No substantive)\s+(?:information|content|data|details).*$',
        r'^[\-\*]*\s*\[(?:No content|Empty|Continues|Continuation)\].*$',
    ]
    lines = [l for l in text.split("\n")
             if not any(re.match(p, l.strip(), re.IGNORECASE) for p in artifact_patterns)]
    result, last_heading = [], None
    for line in lines:
        m = re.match(r'^(\*\*[^*\n]+\*\*)\s*$', line.strip())
        if m:
            h = m.group(1).strip()
            if h == last_heading:
                continue
            last_heading = h
        elif line.strip():
            last_heading = None
        result.append(line)
    text = "\n".join(result)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(l.rstrip() for l in text.split("\n")).strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        raise ValueError("PDF is encrypted — please decrypt it first.")
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    if not pages:
        raise ValueError("No text could be extracted from this PDF.")
    return "\n".join(pages)


# ── 7. AUDIO TRANSCRIPTION ─────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, model, status_write, context: str = "", file_ext: str = ".audio") -> str:
    transcription_instruction = "Transcribe this audio accurately, preserving the speaker's words as closely as possible."
    if context.strip():
        transcription_instruction += (
            f"\n\nContext to help with accurate transcription "
            f"(use this to correctly identify domain-specific terms, names, and abbreviations):\n{context.strip()}"
        )
    local_paths, cloud_names, transcripts = [], [], []
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
        f.write(audio_bytes)
        input_path = f.name
    local_paths.append(input_path)
    try:
        chunk_pattern = input_path + "_chunk_%03d.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-f", "segment", "-segment_time", "300",
             "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", chunk_pattern],
            capture_output=True, timeout=300,
        )
        chunk_files = sorted(glob.glob(input_path + "_chunk_*.wav"))
        local_paths.extend(chunk_files)
        if not chunk_files:
            converted_path = input_path + "_full.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", converted_path],
                capture_output=True, timeout=120,
            )
            local_paths.append(converted_path)
            chunk_files = [converted_path]
        for i, chunk_path in enumerate(chunk_files):
            status_write(f"Transcribing audio chunk {i+1} of {len(chunk_files)}…")
            cloud = genai.upload_file(path=chunk_path)
            cloud_names.append(cloud.name)
            while cloud.state.name == "PROCESSING":
                time.sleep(2)
                cloud = genai.get_file(cloud.name)
            if cloud.state.name != "ACTIVE":
                raise RuntimeError(f"Audio chunk {i+1} failed to process in the cloud.")
            resp = generate_with_retry(model, [transcription_instruction, cloud])
            transcripts.append(resp.text)
    finally:
        for p in local_paths:
            try: os.remove(p)
            except Exception: pass
        for n in cloud_names:
            try: genai.delete_file(n)
            except Exception: pass
    return "\n\n".join(transcripts).strip()


# ── 8. TRANSCRIPT REFINEMENT ───────────────────────────────────────────────────

def refine_transcript(raw: str, meeting_type: str, speakers: str, model, status_write) -> str:
    lang_instr = (
        "IMPORTANT: Your entire output MUST be in English. "
        "If the transcript contains Hindi, Hinglish, or any other non-English language, "
        "translate all content into clear, natural English while preserving the original meaning."
    )
    extra = REFINEMENT_INSTRUCTIONS.get(meeting_type, "")
    speaker_info = f"Participants: {speakers}." if speakers.strip() else ""
    words = raw.split()
    if len(words) <= CHUNK_WORD_SIZE:
        status_write("Refining transcript (single chunk)…")
        prompt = (
            f"Refine the following transcript. Correct spelling, grammar, and punctuation. "
            f"Label speakers clearly if possible. {speaker_info} {extra}\n{lang_instr}\n\n"
            f"TRANSCRIPT:\n{raw}"
        )
        return generate_with_retry(model, prompt).text
    chunks = create_chunks_with_overlap(raw, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
    status_write(f"Refining transcript ({len(chunks)} chunks in parallel)…")
    prompts = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            prompts.append(
                f"You are refining a transcript. Correct spelling, grammar, and punctuation. "
                f"Label speakers clearly if possible. {speaker_info} {extra}\n{lang_instr}\n\n"
                f"TRANSCRIPT CHUNK TO REFINE:\n{chunk}"
            )
        else:
            ctx_tail = " ".join(chunks[i - 1].split()[-CHUNK_WORD_OVERLAP:])
            prompts.append(
                f"You are continuing to refine a long transcript. "
                f"Below is the tail of the previous section for context. {speaker_info} {extra}\n{lang_instr}\n"
                f"---\nCONTEXT (previous chunk tail):\n...{ctx_tail}\n---\n"
                f"NEW CHUNK TO REFINE:\n{chunk}"
            )
    results = [None] * len(chunks)
    def _refine_one(idx, prompt):
        return idx, generate_with_retry(model, prompt).text
    with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as executor:
        futures = {executor.submit(_refine_one, i, p): i for i, p in enumerate(prompts)}
        done = 0
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text
            done += 1
            status_write(f"Refinement: {done}/{len(chunks)} chunks done…")
    return "\n\n".join(r for r in results if r)


# ── 9. NOTES GENERATION ────────────────────────────────────────────────────────

def _build_base_prompt(meeting_type: str, detail_level: str, extra_context: str) -> str:
    if meeting_type == "Expert Meeting":
        base = EXPERT_MEETING_DETAILED_PROMPT if detail_level == "Detailed" else EXPERT_MEETING_CONCISE_PROMPT
    elif meeting_type == "Management Meeting":
        base = MANAGEMENT_MEETING_PROMPT
    else:
        base = INTERNAL_DISCUSSION_PROMPT
    if extra_context.strip():
        base += f"\n\n**ADDITIONAL CONTEXT PROVIDED:**\n{extra_context.strip()}"
    return base


def generate_notes(transcript: str, meeting_type: str, detail_level: str, extra_context: str, model, status_write) -> str:
    base = _build_base_prompt(meeting_type, detail_level, extra_context)
    words = transcript.split()
    if len(words) <= CHUNK_WORD_SIZE:
        status_write("Generating notes (single chunk)…")
        prompt = f"{base}\n\n**MEETING TRANSCRIPT:**\n{transcript}"
        ph = st.empty()
        resp = generate_with_retry(model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
        notes, _ = stream_and_collect(resp, ph)
        return notes
    chunks = create_chunks_with_overlap(transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
    all_chunk_notes = []
    context_package = ""
    for i, chunk in enumerate(chunks):
        status_write(f"Generating notes: chunk {i+1} of {len(chunks)}…")
        template = PROMPT_INITIAL if i == 0 else PROMPT_CONTINUATION
        prompt = template.format(
            base_instructions=base,
            chunk_text=chunk,
            context_package=context_package,
        )
        ph = st.empty()
        resp = generate_with_retry(model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
        chunk_notes, _ = stream_and_collect(resp, ph)
        all_chunk_notes.append(chunk_notes)
        context_package = create_context_from_notes("\n\n".join(all_chunk_notes), i + 1)
    HEADING_RE = r"(?m)^(\*\*[^*\n]+\*\*)\s*$"
    final = all_chunk_notes[0]
    for i in range(1, len(all_chunk_notes)):
        prev, curr = all_chunk_notes[i - 1], all_chunk_notes[i]
        last_headings = list(re.finditer(HEADING_RE, prev))
        if not last_headings:
            final += "\n\n" + curr
            continue
        last_h = last_headings[-1].group(1)
        stitch = re.search(r"(?m)^" + re.escape(last_h) + r"\s*$", curr)
        if stitch:
            after = curr[stitch.start() + len(last_h):]
            next_h = re.search(HEADING_RE, after)
            final += "\n\n" + (after[next_h.start():] if next_h else after)
        else:
            final += "\n\n" + curr
    return cleanup_stitched_notes(final)


# ── 10. INTELLIGENCE EXTRACTION ────────────────────────────────────────────────

def extract_intelligence(notes: str, meeting_type: str, model, status_write) -> str:
    """
    Run the intelligence extraction pass on notes.
    Classifies and organises what was said — no inferences added.
    Long notes are chunked in parallel, then synthesised into one brief.
    """
    prompt_map = {
        "Expert Meeting":      INTEL_EXPERT_PROMPT,
        "Management Meeting":  INTEL_MANAGEMENT_PROMPT,
        "Internal Discussion": INTEL_INTERNAL_PROMPT,
    }
    base_prompt = prompt_map.get(meeting_type, INTEL_EXPERT_PROMPT)
    chunks = create_chunks_with_overlap(notes, INTEL_CHUNK_SIZE, INTEL_OVERLAP)

    # Single-chunk fast path
    if len(chunks) == 1:
        status_write("Extracting intelligence from notes…")
        return generate_with_retry(model, base_prompt.format(notes=notes.strip())).text

    # Multi-chunk: extract in parallel, then synthesise
    status_write(f"Extracting intelligence from {len(chunks)} note sections in parallel…")
    extracts: list[str] = [""] * len(chunks)

    def _extract_one(idx: int, chunk_text: str) -> tuple[int, str]:
        return idx, generate_with_retry(model, base_prompt.format(notes=chunk_text)).text

    with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as executor:
        futures = {executor.submit(_extract_one, i, c): i for i, c in enumerate(chunks)}
        done = 0
        for fut in as_completed(futures):
            idx, text = fut.result()
            extracts[idx] = text
            done += 1
            status_write(f"Intelligence extraction: {done}/{len(chunks)} sections done…")

    status_write("Synthesising intelligence extracts…")
    combined = "\n\n---\n\n".join(f"[Section {i+1}/{len(chunks)}]\n{e}" for i, e in enumerate(extracts))
    synth_prompt = INTEL_SYNTHESIS_PROMPT.format(meeting_type=meeting_type, extracts=combined)
    return generate_with_retry(model, synth_prompt).text


# ── 11. SUMMARY GENERATION ─────────────────────────────────────────────────────

def _get_summary_prompt(meeting_type: str) -> str:
    return {
        "Expert Meeting":      SUMMARY_EXPERT_PROMPT,
        "Management Meeting":  SUMMARY_MANAGEMENT_PROMPT,
        "Internal Discussion": SUMMARY_INTERNAL_PROMPT,
    }.get(meeting_type, SUMMARY_EXPERT_PROMPT)


def generate_summary(intelligence: str, meeting_type: str, word_count: int, focus_block: str, model, status_write) -> str:
    """Generate a two-tier factual summary from the intelligence brief."""
    status_write(f"Generating ~{word_count}-word summary from intelligence brief…")
    prompt = _get_summary_prompt(meeting_type).format(
        word_count=word_count,
        focus_block=focus_block,
        intelligence=intelligence.strip(),
    )
    ph = st.empty()
    resp = generate_with_retry(model, prompt, stream=True)
    summary, _ = stream_and_collect(resp, ph)
    return summary


def refine_summary(current_summary: str, intelligence: str, instruction: str, model, status_write) -> str:
    """Apply a user refinement instruction to an existing summary."""
    status_write("Applying refinement…")
    prompt = SUMMARY_REFINEMENT_PROMPT.format(
        current_summary=current_summary,
        intelligence=intelligence,
        instruction=instruction,
    )
    ph = st.empty()
    resp = generate_with_retry(model, prompt, stream=True)
    revised, _ = stream_and_collect(resp, ph)
    return revised


def parse_two_tier_summary(text: str) -> Tuple[str, str]:
    """
    Split the two-tier output into (brief, detailed_summary).

    Strategy: split on the DETAILED SUMMARY header first (forward-only split),
    then strip the BRIEF header and any --- separators from the left portion.
    This is robust against --- dividers, varied capitalisation, and bold markdown
    on headings (e.g. ## **BRIEF**) that a single-pass regex would miss.
    """
    # Split on the DETAILED SUMMARY header — handles ## DETAILED SUMMARY,
    # ## Detailed Summary, ## **DETAILED SUMMARY**, etc.
    parts = re.split(
        r'\n##\s*\**\s*DETAILED\s+SUMMARY\s*\**\s*\n',
        text, maxsplit=1, flags=re.IGNORECASE
    )

    if len(parts) == 2:
        brief_section, detail = parts
        # Strip the ## BRIEF header (and any bold markers) from the left portion
        brief = re.sub(
            r'^.*?##\s*\**\s*BRIEF\s*\**\s*\n+', '', brief_section,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        # Strip any leading/trailing --- horizontal rule the model may have added
        brief = re.sub(r'^-{3,}\s*\n*', '', brief).strip()
        brief = re.sub(r'\n*-{3,}\s*$', '', brief).strip()
        return brief, detail.strip()

    # Fallback: model didn't follow the two-tier structure — return full text as detail
    return "", text.strip()


# ── 12. UI HELPERS ─────────────────────────────────────────────────────────────

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
        st.error("**GEMINI_API_KEY not set.** Add it to a `.env` file or Streamlit Secrets.")
        st.code('GEMINI_API_KEY = "your_key_here"', language="toml")
        st.stop()


def render_intelligence_panel(intelligence: str, meeting_type: str):
    """Render the intelligence brief with section-level icons for scannability."""
    icons = {
        "EXPERT BACKGROUND":              "👤",
        "CORE THESIS":                    "🎯",
        "KEY INSIGHTS":                   "★",
        "HARD DATA & FACTS":              "📊",
        "DIRECT QUOTES":                  "💬",
        "EXPRESSED RISKS & UNCERTAINTIES":"⚠️",
        "STATED NON-CONSENSUS VIEWS":     "🔍",
        "QUESTIONS & UNCERTAINTIES THE EXPERT RAISED": "❓",
        "MEETING CONTEXT":                "📋",
        "DECISIONS MADE":                 "✅",
        "ACTION ITEMS":                   "→",
        "KEY DEBATES":                    "⚖️",
        "UNRESOLVED ISSUES":              "🔓",
        "KEY DATA & FACTS REFERENCED":    "📊",
        "DISCUSSION GOAL":                "🎯",
        "VIEWS & POSITIONS EXPRESSED":    "💭",
        "WHERE THERE WAS AGREEMENT":      "🤝",
        "WHERE THERE WAS DISAGREEMENT":   "⚡",
        "CONCLUSIONS REACHED":            "✅",
        "NEXT STEPS":                     "→",
    }
    # Parse sections and render with icons
    lines = intelligence.split("\n")
    output_lines = []
    for line in lines:
        heading_match = re.match(r'^(##\s+)(.+)$', line)
        if heading_match:
            prefix, title = heading_match.groups()
            icon = icons.get(title.strip().upper(), "")
            output_lines.append(f"{prefix}{icon} {title}" if icon else line)
        else:
            output_lines.append(line)
    st.markdown("\n".join(output_lines))


# ── 13. PAGE: PROCESS ──────────────────────────────────────────────────────────

def page_process():
    api_key_check()
    st.header("Process Meeting")

    with st.sidebar:
        st.markdown("### Model Settings")
        notes_model_name = st.selectbox(
            "Notes model", list(MODELS.keys()), index=2,
            key="notes_model", help="Main note generation. Use 2.5 Pro for highest quality."
        )
        refine_model_name = st.selectbox(
            "Refinement model", list(MODELS.keys()), index=1,
            key="refine_model", help="Transcript clean-up pass."
        )
        transcription_model_name = st.selectbox(
            "Transcription model", list(MODELS.keys()), index=3,
            key="transcription_model", help="Audio-to-text."
        )
        intel_model_name = st.selectbox(
            "Intelligence model", list(MODELS.keys()), index=0,
            key="intel_model", help="Intelligence extraction from notes. 2.5 Flash is fast and accurate for this task."
        )
        refine_enabled = st.toggle(
            "Enable refinement pass", value=True, key="refine_toggle",
            help="Fixes grammar, labels speakers, translates non-English."
        )

    meeting_type = st.selectbox("Meeting type", MEETING_TYPES, key="meeting_type")

    detail_level = "Concise"
    if meeting_type == "Expert Meeting":
        detail_level = st.radio(
            "Note style", ["Concise", "Detailed"], horizontal=True, key="detail_level",
            help="**Concise**: information-dense, no filler.  **Detailed**: maximum verbosity, zero omission."
        )

    with st.expander("Context & additional instructions", expanded=False):
        st.caption(
            "Passed into transcription (to identify domain terms) and note generation (to guide structure)."
        )
        speakers = st.text_input(
            "Speaker names (comma-separated)", key="speakers",
            placeholder="e.g. John Smith (analyst), Dr. Patel (expert)"
        )
        extra_context = st.text_area(
            "Background / additional instructions", height=100, key="extra_context",
            placeholder="e.g. Expert call on Indian cement sector. Key companies: UltraTech, Shree Cement."
        )

    extra_context_combined = "\n".join(filter(None, [
        f"Speakers: {speakers}" if speakers.strip() else "",
        extra_context.strip(),
    ]))

    st.divider()
    input_method = st.radio(
        "Input method",
        ["Paste Text", "Upload File (PDF / Audio)", "Record Audio"],
        horizontal=True, key="input_method",
    )

    raw_text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    audio_ext: str = ".audio"
    is_audio = False

    if input_method == "Paste Text":
        raw_text = st.text_area("Paste transcript here", height=300, key="text_input",
                                placeholder="Paste your meeting transcript…")

    elif input_method == "Upload File (PDF / Audio)":
        uploaded = st.file_uploader(
            "Upload a PDF, TXT, or audio file",
            type=["pdf", "txt", "md", "wav", "mp3", "m4a", "ogg", "flac", "mp4", "mov"],
            key="uploaded_file",
        )
        if uploaded:
            ext = os.path.splitext(uploaded.name)[1].lower()
            size_mb = uploaded.size / (1024 * 1024)
            if ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".mov"]:
                if size_mb > MAX_AUDIO_MB:
                    st.error(f"Audio file too large ({size_mb:.1f} MB). Limit: {MAX_AUDIO_MB} MB.")
                else:
                    is_audio = True
                    audio_bytes = uploaded.getvalue()
                    audio_ext = ext
                    st.info(f"Audio loaded: **{uploaded.name}** ({size_mb:.1f} MB)")
            elif ext == ".pdf":
                if size_mb > MAX_PDF_MB:
                    st.error(f"PDF too large ({size_mb:.1f} MB). Limit: {MAX_PDF_MB} MB.")
                else:
                    try:
                        raw_text = extract_pdf_text(uploaded.getvalue())
                        st.success(f"PDF loaded: **{uploaded.name}** — {len(raw_text.split()):,} words extracted")
                    except Exception as e:
                        st.error(str(e))
            else:
                raw_text = uploaded.getvalue().decode("utf-8")
                st.success(f"File loaded: **{uploaded.name}** — {len(raw_text.split()):,} words")

    else:
        st.caption("Click the microphone to start recording. Click again to stop.")
        recording = st.audio_input("Record a voice note", key="audio_recording")
        if recording:
            audio_bytes = recording.getvalue()
            audio_ext = ".webm"
            is_audio = True
            st.success("Recording captured. Click **Process** when ready.")

    st.divider()
    if st.button("Process Meeting", type="primary", use_container_width=True):
        has_input = is_audio or (raw_text and raw_text.strip())
        if not has_input:
            st.error("Please provide a transcript (paste text, upload a file, or record audio).")
            st.stop()

        notes_model   = get_model(notes_model_name)
        refine_model  = get_model(refine_model_name)
        transcr_model = get_model(transcription_model_name)
        intel_model   = get_model(intel_model_name)

        with st.status("Processing…", expanded=True) as status:
            try:
                if is_audio:
                    transcript = transcribe_audio(audio_bytes, transcr_model, st.write,
                                                  context=extra_context_combined, file_ext=audio_ext)
                    st.write(f"✓ Transcription complete: **{len(transcript.split()):,} words**")
                else:
                    transcript = re.sub(r"\n{3,}", "\n\n", raw_text.strip())

                if refine_enabled:
                    transcript = refine_transcript(transcript, meeting_type, speakers, refine_model, st.write)
                    st.write(f"✓ Refinement complete: **{len(transcript.split()):,} words**")

                notes = generate_notes(transcript, meeting_type, detail_level,
                                       extra_context_combined, notes_model, st.write)
                if not notes or not notes.strip():
                    raise ValueError("The model returned empty notes. Please try again.")
                st.write(f"✓ Notes generated: **{len(notes.split()):,} words**")

                intelligence = extract_intelligence(notes, meeting_type, intel_model, st.write)
                if not intelligence or not intelligence.strip():
                    raise ValueError("Intelligence extraction returned empty output. Please try again.")
                st.write("✓ Intelligence extracted.")

                st.session_state["last_notes"]        = notes
                st.session_state["last_transcript"]   = transcript
                st.session_state["last_intelligence"] = intelligence
                st.session_state["last_meeting_type"] = meeting_type
                # Clear any prior summary history when new notes are processed
                st.session_state.pop("summary_history", None)

                status.update(label="Done!", state="complete")
                st.write("✓ Ready. Go to the **Summary** tab to generate a summary.")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    if "last_notes" in st.session_state:
        st.divider()
        notes       = st.session_state["last_notes"]
        intelligence = st.session_state.get("last_intelligence", "")
        meeting_type_label = st.session_state.get("last_meeting_type", "")

        tab_notes, tab_intel = st.tabs(["📄 Notes", "★ Intelligence Brief"])

        with tab_notes:
            col_title, col_copy = st.columns([3, 1])
            with col_title:
                st.subheader(f"Generated Notes  ({len(notes.split()):,} words)")
            with col_copy:
                copy_button(notes, "Copy Notes")
            st.markdown(notes)

        with tab_intel:
            if intelligence:
                col_title, col_copy = st.columns([3, 1])
                with col_title:
                    st.subheader(f"Intelligence Brief — {meeting_type_label}")
                with col_copy:
                    copy_button(intelligence, "Copy Brief")
                st.caption(
                    "This brief classifies and organises what was said — no inferences added. "
                    "Every point traces back to the notes."
                )
                st.divider()
                render_intelligence_panel(intelligence, meeting_type_label)
            else:
                st.info("Run **Process Meeting** to generate the intelligence brief.")


# ── 14. PAGE: SUMMARY ──────────────────────────────────────────────────────────

def page_summary():
    api_key_check()
    st.header("Summary")

    with st.sidebar:
        st.markdown("### Model Settings")
        intel_model_name = st.selectbox(
            "Intelligence model", list(MODELS.keys()), index=0,
            key="summary_intel_model",
            help="Used to extract intelligence if pasting notes manually."
        )
        summary_model_name = st.selectbox(
            "Summary model", list(MODELS.keys()), index=2,
            key="summary_model",
            help="Used to generate the summary. 2.5 Pro gives the highest quality output."
        )

    # ── Source selection ───────────────────────────────────────────────────────
    has_session = "last_intelligence" in st.session_state and "last_notes" in st.session_state
    source = st.radio(
        "Notes source",
        ["From last processed session", "Paste notes manually"],
        horizontal=True, key="summary_source",
    )

    notes_input     = ""
    intelligence    = None
    meeting_type    = "Expert Meeting"

    if source == "From last processed session":
        if not has_session:
            st.info("No processed session yet. Process a transcript on the **Process Meeting** page first, or paste notes manually.")
            return
        meeting_type = st.session_state["last_meeting_type"]
        notes_input  = st.session_state["last_notes"]
        intelligence = st.session_state["last_intelligence"]
        st.caption(
            f"Using intelligence brief from session — **{meeting_type}** | "
            f"{len(notes_input.split()):,} words in notes"
        )
    else:
        meeting_type = st.selectbox("Meeting type", MEETING_TYPES, key="manual_meeting_type")
        notes_input  = st.text_area(
            "Paste meeting notes", height=300, key="manual_notes",
            placeholder="Paste your meeting notes here…"
        )
        st.caption("Intelligence will be extracted from these notes before summary generation.")

    # ── Length ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Summary length** (applies to the Detailed Summary — the Brief is always ~100–150 words)")
    length_label = st.radio(
        "length", list(SUMMARY_PRESETS.keys()), horizontal=True,
        key="summary_length", label_visibility="collapsed"
    )
    word_count = SUMMARY_PRESETS[length_label]
    if word_count is None:
        word_count = st.number_input("Custom word count", min_value=100, value=1000, step=100, key="custom_wc")

    # ── Focus instructions + saved prompts ─────────────────────────────────────
    st.divider()
    st.markdown("**Focus instructions** *(optional)*")
    saved_data   = load_saved_prompts()
    saved_names  = list(saved_data["prompts"].keys())
    default_name = saved_data.get("default")
    dropdown_options = ["— none —"] + saved_names
    default_idx  = (dropdown_options.index(default_name) if default_name in dropdown_options else 0)
    selected_saved = st.selectbox("Saved focus prompts", dropdown_options,
                                  index=default_idx, key="saved_prompt_select")

    if "_last_loaded_prompt" not in st.session_state:
        st.session_state["_last_loaded_prompt"] = selected_saved

    if selected_saved != st.session_state["_last_loaded_prompt"]:
        if selected_saved != "— none —":
            st.session_state["focus_text_area"] = saved_data["prompts"][selected_saved]
        else:
            st.session_state["focus_text_area"] = ""
        st.session_state["_last_loaded_prompt"] = selected_saved

    focus_instructions = st.text_area(
        "Focus instructions", height=100, key="focus_text_area",
        placeholder="e.g. Focus on pricing dynamics and competitive positioning."
    )

    col_save, col_default, col_delete = st.columns(3)
    with col_save:
        save_name = st.text_input("Save as", key="save_name_input", placeholder="Prompt name")
        if st.button("Save", key="btn_save") and save_name.strip() and focus_instructions.strip():
            saved_data["prompts"][save_name.strip()] = focus_instructions.strip()
            write_saved_prompts(saved_data)
            st.success(f"Saved **{save_name.strip()}**")
            st.rerun()
    with col_default:
        if st.button("Set as default", key="btn_default") and selected_saved != "— none —":
            saved_data["default"] = selected_saved
            write_saved_prompts(saved_data)
            st.success(f"**{selected_saved}** set as default")
    with col_delete:
        if st.button("Delete", key="btn_delete") and selected_saved != "— none —":
            saved_data["prompts"].pop(selected_saved, None)
            if saved_data.get("default") == selected_saved:
                saved_data["default"] = None
            write_saved_prompts(saved_data)
            st.rerun()

    # ── Generate ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("Generate Summary", type="primary", use_container_width=True):
        if not notes_input.strip():
            st.error("Please provide notes.")
            st.stop()

        focus_text = focus_instructions.strip()
        focus_block = (
            "\n**FOCUS INSTRUCTIONS:** In addition to the standard structure, ensure the following "
            f"aspects are clearly covered:\n{focus_text}\n"
        ) if focus_text else ""

        intel_model   = get_model(intel_model_name)
        summary_model = get_model(summary_model_name)

        status_ph = st.empty()
        def _status(msg: str):
            status_ph.info(f"⏳ {msg}")

        try:
            # Extract intelligence if not already available from session
            if intelligence:
                _status("Using pre-extracted intelligence brief…")
                current_intel = intelligence
            else:
                current_intel = extract_intelligence(notes_input, meeting_type, intel_model, _status)
                st.session_state["last_intelligence"] = current_intel
                st.session_state["last_meeting_type"] = meeting_type

            summary = generate_summary(current_intel, meeting_type, word_count, focus_block, summary_model, _status)
            status_ph.empty()

            if not summary.strip():
                raise ValueError("Model returned an empty response.")

            # Initialise or append to summary history
            history = st.session_state.get("summary_history", [])
            history.append({
                "version":     len(history) + 1,
                "summary":     summary,
                "instruction": None,
                "word_count":  word_count,
            })
            st.session_state["summary_history"] = history

        except Exception as e:
            status_ph.empty()
            st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    history = st.session_state.get("summary_history", [])
    if not history:
        return

    current = history[-1]
    brief, detail = parse_two_tier_summary(current["summary"])

    st.divider()

    # Brief in a prominent callout
    col_brief_title, col_brief_copy = st.columns([3, 1])
    with col_brief_title:
        st.subheader("The Brief")
    with col_brief_copy:
        copy_button(brief, "Copy Brief")
    st.info(brief)

    st.divider()

    # Detailed summary
    actual_words = len(detail.split())
    col_detail_title, col_detail_copy = st.columns([3, 1])
    with col_detail_title:
        st.subheader(f"Detailed Summary  ({actual_words:,} words)")
    with col_detail_copy:
        copy_button(current["summary"], "Copy Full")
    st.markdown(detail)

    # ── Refinement ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Refine this summary**")
    st.caption(
        "Ask for structural or focus changes — e.g. *\"Expand the data section\"*, "
        "*\"Make the brief shorter\"*, *\"Focus more on the risk discussion.\"*  "
        "Refinements stay factual — the model only uses information already in the brief."
    )
    refinement_instruction = st.text_area(
        "Refinement instruction", height=80, key="refinement_input",
        placeholder="e.g. Expand the data section and shorten the Core Views section."
    )
    if st.button("Apply Refinement", key="btn_refine") and refinement_instruction.strip():
        current_intel = st.session_state.get("last_intelligence", "")
        if not current_intel:
            st.error("No intelligence brief in session. Generate a summary first.")
        else:
            summary_model = get_model(summary_model_name)
            status_ph2 = st.empty()
            def _status2(msg):
                status_ph2.info(f"⏳ {msg}")
            try:
                revised = refine_summary(current["summary"], current_intel,
                                         refinement_instruction.strip(), summary_model, _status2)
                status_ph2.empty()
                history.append({
                    "version":     len(history) + 1,
                    "summary":     revised,
                    "instruction": refinement_instruction.strip(),
                    "word_count":  len(revised.split()),
                })
                st.session_state["summary_history"] = history
                st.rerun()
            except Exception as e:
                status_ph2.empty()
                st.error(f"**Error:** {e}")

    # ── Version history ────────────────────────────────────────────────────────
    if len(history) > 1:
        st.divider()
        with st.expander(f"Version history  ({len(history)} versions)", expanded=False):
            for entry in reversed(history[:-1]):
                label = (
                    f"v{entry['version']} — after: *\"{entry['instruction']}\"*"
                    if entry["instruction"]
                    else f"v{entry['version']} — original (~{entry['word_count']} words)"
                )
                with st.expander(label, expanded=False):
                    b, d = parse_two_tier_summary(entry["summary"])
                    if b:
                        st.markdown("**Brief:**")
                        st.info(b)
                    st.markdown(d)
                    copy_button(entry["summary"], f"Copy v{entry['version']}")


# ── 15. PAGE: ANALYSE ──────────────────────────────────────────────────────────

def run_analysis(intelligence: str, question: str, model, status_write) -> str:
    """Run a single analysis question against the intelligence brief."""
    status_write("Analysing…")
    prompt = ANALYSIS_PROMPT.format(intelligence=intelligence.strip(), question=question.strip())
    ph = st.empty()
    resp = generate_with_retry(model, prompt, stream=True)
    result, _ = stream_and_collect(resp, ph)
    return result


def page_analyse():
    api_key_check()
    st.header("Analyse")

    # ── Prominent disclaimer ───────────────────────────────────────────────────
    st.warning(
        "**This page produces AI analysis and inferences — not a factual record of the meeting.** "
        "Responses are the model's interpretation of the meeting content and should be treated as "
        "analytical input, not as facts. The notes and summary pages contain only what was actually said.",
        icon="⚠️",
    )

    with st.sidebar:
        st.markdown("### Model Settings")
        analysis_model_name = st.selectbox(
            "Analysis model", list(MODELS.keys()), index=2,
            key="analysis_model",
            help="2.5 Pro gives the most rigorous analytical reasoning."
        )

    # ── Source ─────────────────────────────────────────────────────────────────
    has_session = "last_intelligence" in st.session_state
    source = st.radio(
        "Intelligence brief source",
        ["From last processed session", "Paste intelligence brief manually"],
        horizontal=True, key="analyse_source",
    )

    intelligence = ""
    meeting_type = "Expert Meeting"

    if source == "From last processed session":
        if not has_session:
            st.info(
                "No processed session yet. Process a transcript on the **Process Meeting** page first, "
                "or paste an intelligence brief manually."
            )
            return
        intelligence = st.session_state["last_intelligence"]
        meeting_type = st.session_state.get("last_meeting_type", "Expert Meeting")
        st.caption(f"Using intelligence brief from session — **{meeting_type}**")
        with st.expander("View intelligence brief", expanded=False):
            render_intelligence_panel(intelligence, meeting_type)
    else:
        meeting_type = st.selectbox("Meeting type", MEETING_TYPES, key="analyse_meeting_type")
        intelligence = st.text_area(
            "Paste intelligence brief", height=250, key="analyse_intel_input",
            placeholder="Paste the intelligence brief extracted from the Process Meeting page…"
        )

    st.divider()

    # ── Preset questions ───────────────────────────────────────────────────────
    presets = ANALYSIS_PRESETS.get(meeting_type, [])
    if presets:
        st.markdown("**Preset analysis questions**")
        st.caption("Select one to pre-fill the question box, or write your own below.")
        preset_cols = st.columns(1)
        selected_preset = st.selectbox(
            "Choose a preset", ["— write your own —"] + presets,
            key="preset_question", label_visibility="collapsed"
        )
        if ("_last_preset" not in st.session_state or
                st.session_state["_last_preset"] != selected_preset):
            if selected_preset != "— write your own —":
                st.session_state["analysis_question"] = selected_preset
            st.session_state["_last_preset"] = selected_preset

    # ── Question input ─────────────────────────────────────────────────────────
    st.markdown("**Your question**")
    question = st.text_area(
        "Analysis question", height=100, key="analysis_question",
        placeholder="e.g. What assumptions is the expert making that are not explicitly stated?"
    )

    st.divider()
    if st.button("Run Analysis", type="primary", use_container_width=True):
        if not intelligence.strip():
            st.error("Please provide an intelligence brief.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        analysis_model = get_model(analysis_model_name)
        status_ph = st.empty()

        try:
            result = run_analysis(intelligence, question, analysis_model,
                                  lambda msg: status_ph.info(f"⏳ {msg}"))
            status_ph.empty()

            if not result.strip():
                raise ValueError("Model returned an empty response.")

            # Store analysis history
            analyses = st.session_state.setdefault("analysis_history", [])
            analyses.append({"question": question.strip(), "answer": result.strip()})

        except Exception as e:
            status_ph.empty()
            st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    analyses = st.session_state.get("analysis_history", [])
    if not analyses:
        return

    # Show most recent first
    for i, entry in enumerate(reversed(analyses)):
        st.divider()
        entry_num = len(analyses) - i
        col_q, col_copy = st.columns([4, 1])
        with col_q:
            st.markdown(f"**Q{entry_num}: {entry['question']}**")
        with col_copy:
            copy_button(entry["answer"], f"Copy A{entry_num}")

        # Render in an amber-tinted container to visually distinguish from factual output
        st.warning(entry["answer"], icon="🔍")

    if len(analyses) > 1:
        if st.button("Clear analysis history", key="clear_analyses"):
            st.session_state.pop("analysis_history", None)
            st.rerun()


# ── 16. MAIN ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SynthNotes Pro", layout="wide", page_icon="★")
    nav = st.navigation([
        st.Page(page_process, title="Process Meeting", icon=":material/edit_note:"),
        st.Page(page_summary, title="Summary",         icon=":material/summarize:"),
        st.Page(page_analyse, title="Analyse",         icon=":material/psychology:"),
    ])
    nav.run()


if __name__ == "__main__":
    main()
