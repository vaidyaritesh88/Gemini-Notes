# /--------------------------\
# |   START OF app.py FILE   |
# \--------------------------/

# --- 1. IMPORTS ---
import streamlit as st
import google.generativeai as genai
import os
import io
import json
import time
from datetime import datetime
import uuid
import traceback
from dotenv import load_dotenv
import PyPDF2
from pydub import AudioSegment
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import re
import tempfile
import html as html_module
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import streamlit.components.v1 as components

# --- Local Imports ---
import database

# --- App-wide CSS ---
APP_CSS = """
<style>
/* ── Active navigation tab highlight ── */
[data-testid="stNavigation"] button[aria-selected="true"] {
    border-bottom: 3px solid var(--primary-color) !important;
    font-weight: 600 !important;
    color: var(--primary-color) !important;
}
[data-testid="stNavigation"] button {
    transition: border-bottom 0.15s ease, color 0.15s ease;
}

/* ── Reduce top padding for a tighter header ── */
.main .block-container {
    padding-top: 1.5rem !important;
}

/* ── Ultra-wide: cap content width for readability ── */
@media (min-width: 1800px) {
    .main .block-container {
        max-width: 1600px !important;
        margin: 0 auto !important;
    }
}

/* ── Section dividers: lighter, more breathing room ── */
hr {
    margin-top: 1.2rem !important;
    margin-bottom: 1.2rem !important;
    opacity: 0.3;
}

/* ── Note cards in history list: subtle hover lift ── */
[data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
    box-shadow: 0 2px 8px rgba(128,128,128,0.15);
    transform: translateY(-1px);
}

/* ── Tighten metric blocks ── */
[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}

/* ── Focus visibility for keyboard navigation (WCAG 2.1 AA) ── */
button:focus-visible,
[data-testid="stSelectbox"] select:focus-visible,
textarea:focus-visible,
input:focus-visible,
[role="tab"]:focus-visible {
    outline: 2px solid var(--primary-color) !important;
    outline-offset: 2px !important;
}

/* ── Text overflow: prevent long filenames from breaking layout ── */
[data-testid="stMarkdownContainer"] p {
    overflow-wrap: break-word;
    word-break: break-word;
}

/* ── Responsive: tablets (stack 4-col action bars into 2x2) ── */
@media (max-width: 1024px) and (min-width: 769px) {
    .main .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
}

/* ── Responsive: mobile ── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 0.5rem;
    }
    textarea {
        min-width: 100% !important;
    }
    /* 44px minimum touch target (WCAG 2.5.5) */
    button {
        min-height: 44px !important;
        padding: 0.5rem 1rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Stack the note header on mobile */
    h3 {
        font-size: 1.1rem !important;
    }
}

/* ── Copy button iframe ── */
iframe {
    min-height: 45px !important;
}

/* ── Print: hide navigation and interactive elements ── */
@media print {
    [data-testid="stNavigation"],
    [data-testid="stSidebar"],
    button,
    iframe,
    .stProgress {
        display: none !important;
    }
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
}

/* ── Smooth transitions globally ── */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
</style>
"""

# --- 2. CONSTANTS & CONFIG ---
load_dotenv()
try:
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"]:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    else:
        st.session_state.config_error = "🔴 GEMINI_API_KEY not found."
except Exception as e:
    st.session_state.config_error = f"🔴 Error configuring Google AI Client: {e}"

MAX_PDF_MB = 25
MAX_AUDIO_MB = 200
CHUNK_WORD_SIZE = 4000
CHUNK_WORD_OVERLAP = 400
# High output token ceiling for notes generation.
# Without this, Gemini defaults to ~8192 output tokens and silently
# truncates long, detailed notes — especially on later chunks.
MAX_OUTPUT_TOKENS = 65536

AVAILABLE_MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash", "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 2.0 Flash": "gemini-2.0-flash-lite", "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite", "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 3.0 Flash": "gemini-3-flash-preview", "Gemini 3.0 Pro": "gemini-3-pro-preview",
    "Gemini 3 Flash Preview": "gemini-3-flash-preview",
    "Gemini 3 Pro Preview": "gemini-3-pro-preview",
    "Gemini 3.1 Pro Preview": "gemini-3.1-pro-preview",
    "Gemini 3.1 Pro Preview (Custom Tools)": "gemini-3.1-pro-preview-customtools",
}
MEETING_TYPES = ["Expert Meeting", "Earnings Call", "Management Meeting", "Internal Discussion", "Custom"]
MAX_TOPIC_DISCOVERY_FILES = 4  # Number of PDFs to scan for topic discovery
EXPERT_MEETING_OPTIONS = ["Option 1: Detailed & Strict", "Option 2: Less Verbose", "Option 3: Less Verbose + Summary"]
EARNINGS_CALL_MODES = ["Generate New Notes", "Enrich Existing Notes"]

TONE_OPTIONS = ["As Is", "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
NUMBER_FOCUS_OPTIONS = ["No Numbers", "Light", "Moderate", "Data-Heavy"]
OTG_WORD_COUNT_OPTIONS = {
    "Short (~150 words)": "Approximately 150 words. Keep it very concise — only the most essential points.",
    "Medium (~300 words)": "Approximately 300 words. Short and direct but cover all key findings.",
    "Long (~500 words)": "Approximately 500 words. Cover all findings with enough detail for context.",
    "Detailed (~750 words)": "Approximately 750 words. Provide thorough coverage with supporting detail and nuance.",
}
NUMBER_FOCUS_INSTRUCTIONS = {
    "No Numbers": "Do NOT include any specific numbers, percentages, monetary values, or metrics. Describe trends and findings qualitatively using words like 'significant,' 'substantial,' 'modest,' etc.",
    "Light": "Include only the most critical 2-3 numbers that are essential to the narrative. Describe most findings qualitatively.",
    "Moderate": "Include key numbers, percentages, and metrics where they support your points. Balance data with narrative flow.",
    "Data-Heavy": "Include ALL specific numbers, percentages, monetary values, metrics, and data points from the notes. The output should be dense with quantitative evidence supporting every claim.",
}

MEETING_TYPE_HELP = {
    "Expert Meeting": "Q&A format with detailed factual extraction from expert consultations",
    "Earnings Call": "Financial data, management commentary, guidance, and analyst Q&A",
    "Management Meeting": "Decisions, action items, owners, and key discussion points",
    "Internal Discussion": "Perspectives, ideas, reasoning, conclusions, and next steps",
    "Custom": "Provide your own formatting instructions via the context field",
}

# --- PROMPT CONSTANTS ---

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

**(2.) Q&A format:**
Structure the main body in Question/Answer format.

**(2.A) Questions:**
-   Identify the core question being asked and rephrase it clearly in **bold**. Do NOT copy verbatim from the transcript — clean up filler and rambling into a clear, well-formed question.
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or any similar label. The bold question text stands alone.
-   **LONG QUESTIONS / PREAMBLE:** Sometimes a question is long because the interviewer provides substantial framing, background, or context before asking — this preamble is part of the question and must be preserved as part of the bold question text. Do NOT treat the preamble as part of the answer.
-   **SPACING:** Leave exactly one blank line between the end of one answer and the start of the next bold question, so each Q&A pair is visually separated.

**(2.B) Answers:**
-   Use bullet points (`-`) directly below the question (no blank line between the bold question and its first bullet).
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, examples, monetary values (`$`), percentages (`%`), metrics, and specific entities mentioned.
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Do not over-summarize. You must retain the following:
    -   **Sentiment & Tone:** Note if the speaker is optimistic, hesitant, confident, or speculative (e.g., "The expert was cautiously optimistic about...", "He speculated that...").
    -   **Qualifiers:** Preserve modifying words that change meaning (e.g., "usually," "in most cases," "rarely," "a potential risk is...").
    -   **Key Examples & Analogies:** If the speaker uses a specific example to illustrate a point, capture it, even if it's a few sentences long.
    -   **Cause & Effect:** Retain any reasoning provided (e.g., "...because of the new regulations," "...which led to a decrease in...")."""

EARNINGS_CALL_PROMPT = """### **NOTES STRUCTURE: EARNINGS CALL**

Generate detailed earnings call notes based on the transcript. Structure your notes under the following topics, using **bold headings** and bullet points for each:

{topic_instructions}

**PRIORITY #1: CAPTURE ALL FINANCIAL DATA.** Revenue, margins, EPS, guidance ranges, growth rates, basis points, dollar amounts — every number matters.
**PRIORITY #2: CAPTURE FORWARD GUIDANCE.** Any forward-looking statements, guidance ranges, management expectations, or outlook commentary.
**PRIORITY #3: PRESERVE MANAGEMENT TONE.** Note confidence, caution, hedging language, or changes from prior quarter tone.
**PRIORITY #4: CAPTURE SEGMENT/VERTICAL DETAIL.** Business segment breakdowns, geographic splits, and vertical-specific commentary."""

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

Structure the notes to capture the flow of ideas, key arguments, and conclusions.

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
4.  **NO META-COMMENTARY:** NEVER produce statements about the transcript itself, such as "the transcript does not contain an answer," "no relevant information in this section," "the chunk starts mid-conversation," or similar. If a chunk begins mid-answer, capture that content as a continuation of the relevant section. Always extract and document whatever substantive content exists.
5.  **MID-CHUNK STARTS:** If the chunk starts in the middle of a response, begin your notes by capturing that content under the most relevant heading from context. Do not skip or discard partial content.
6.  **MAINTAIN OUTPUT VOLUME:** This chunk contains the same amount of content as the first chunk. Your output for this chunk MUST be equally detailed and equally long. Do NOT produce a shorter or more condensed output just because this is a continuation. If the first chunk produced 30 bullet points, this chunk should produce a similar number. Do NOT taper off, summarize, or become briefer.

---
{base_instructions}
---

**MEETING TRANSCRIPT (NEW CHUNK):**
{chunk_text}
"""

VALIDATION_DETAILED_PROMPT = """You are a rigorous Transcript Completeness Auditor performing a fact-by-fact audit of processed meeting notes against the source transcript.

## INPUTS

### FULL PROCESSED NOTES (complete — for reference when checking missing content):
{full_notes}

### PORTION TO ANNOTATE ({chunk_info}):
{chunk_to_annotate}

### SOURCE TRANSCRIPT (Ground Truth):
{transcript}

## CRITICAL UNDERSTANDING

Notes are always paraphrased and restructured versions of the transcript — this is intentional and CORRECT. You must NEVER flag paraphrasing, rephrasing, reorganisation, or compression as errors. The note-taking AI's job is to restructure, not transcribe verbatim.

**Cross-chunk context:** The FULL PROCESSED NOTES above contain all Q&As from this call. When checking for missing content, check the FULL NOTES — if a piece of transcript content is captured *anywhere* in the full notes (even outside the PORTION TO ANNOTATE), do NOT flag it as missing.

## WHAT TO FIND — BE RIGOROUS

**1. MISSING CONTENT** (most important — go fact by fact)

Walk through the TRANSCRIPT systematically, exchange by exchange. For each expert response, check the FULL NOTES for every one of the following:

- **Every specific number, percentage, monetary value, metric, or growth rate** — even a single missing figure is a gap
- **Every named entity** — companies, people, product names, geographies, regulatory bodies, specific time periods
- **Every distinct example, anecdote, or case study** the expert used to illustrate a point — these are high-value and frequently dropped
- **Every qualifier or hedge that changes meaning** — "roughly," "typically," "only in certain cases," "except when," "approximately," "we think," "possibly" — omitting these alters the meaning materially
- **Every distinct reasoning chain or cause-effect link** — e.g., "because X, Y happened, which led to Z"
- **Every comparison or contrast** — if the expert compared two companies, sectors, or time periods, check both sides are captured
- **Every explicitly stated uncertainty or caveat** — if the expert said they were unsure, speculating, or hedging, that tone must be preserved

Only flag as MISSING if the fact, name, number, or nuance genuinely does not appear anywhere in the FULL NOTES.

**2. MISREPRESENTATION** (apply sparingly but precisely)

Content in the PORTION TO ANNOTATE that factually contradicts or distorts the transcript:
- Wrong number (transcript: 30%, notes: 20%)
- Wrong direction (transcript: declining, notes: growing)
- Wrong entity name or wrong speaker attribution
- Expert expressed uncertainty but notes state it as established fact, or vice versa
- A "could" or "might" in the transcript rendered as a definitive claim in the notes

**3. REPEATED Q&A** (check the FULL NOTES)

Scan the FULL PROCESSED NOTES for Q&A pairs that cover substantially the same question or repeat the same answer content. This happens when chunked note generation produces near-duplicate sections due to transcript overlap. Flag as repeated if:
- Two bold questions ask essentially the same thing (even if worded differently)
- An answer block appears twice with the same or very similar bullet points
- A topic or data point is covered in near-identical language in two separate Q&A pairs

## WHAT NOT TO FLAG

- Paraphrasing → CORRECT
- Restructuring or reordering → CORRECT
- Compression where key facts are still present → CORRECT
- Filler, false starts, rambling clean-up → CORRECT
- Minor synonym substitutions that preserve meaning → CORRECT
- A topic mentioned briefly in one Q&A and fully covered in another → NOT a repeat (only flag true duplicates)

## ANNOTATIONS — THREE TYPES ONLY

Do NOT use any markup other than these three exact formats.

**MISSING CONTENT** — insert immediately after the Q&A pair in the PORTION TO ANNOTATE where the gap is most relevant:
`<div style="background:#fef9c3;border-left:3px solid #ca8a04;padding:5px 10px;margin:6px 0;font-size:0.88em;color:#78350f">⚠️ <strong>Missing:</strong> [quote or precisely describe the specific fact, number, name, qualifier, or example from the transcript that is absent from the full notes]</div>`

**MISREPRESENTATION** — wrap only the specific wrong text, immediately followed by an inline correction:
`<del style="color:#dc2626">the wrong text as it appears in the notes</del><span style="color:#16a34a;font-size:0.9em"> → [what the transcript actually says]</span>`

**REPEATED Q&A** — insert immediately before the duplicate bold question in the PORTION TO ANNOTATE:
`<div style="background:#ede9fe;border-left:3px solid #7c3aed;padding:5px 10px;margin:6px 0;font-size:0.88em;color:#5b21b6">🔁 <strong>Duplicate:</strong> This Q&A substantially repeats [describe which earlier Q&A it duplicates and what the overlapping content is]</div>`

**Correct content** — leave exactly as-is. No annotation whatsoever.

## OUTPUT

Output ONLY the annotated PORTION TO ANNOTATE, preserving its exact structure (bold questions, bullet points, spacing). Do NOT output the full notes section, and do NOT add any summary, preamble, footer, or meta-commentary of any kind."""

def cleanup_stitched_notes(notes_text: str) -> str:
    """Deterministic cleanup of stitched notes — no LLM call, zero risk of content loss.

    Handles:
    1. Remove meta-commentary / processing artifacts from chunked generation
    2. Collapse duplicate consecutive headings from overlap regions
    3. Fix formatting: excessive blank lines, trailing whitespace
    """
    if not notes_text or not notes_text.strip():
        return notes_text

    # --- 1. Remove known meta-commentary artifacts ---
    artifact_patterns = [
        r'^[\-\*]*\s*(?:Note:|Disclaimer:)?\s*(?:The|This)\s+(?:transcript|section|chunk|portion|segment)\s+(?:does not|doesn\'t|appears to)\s+.*$',
        r'^[\-\*]*\s*(?:No relevant|No additional|No further|No substantive)\s+(?:information|content|data|details).*$',
        r'^[\-\*]*\s*(?:This section (?:appears|seems|is) (?:incomplete|empty|blank)).*$',
        r'^[\-\*]*\s*\[(?:No content|Empty|Continues|Continuation)\].*$',
    ]
    lines = notes_text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        is_artifact = any(re.match(p, stripped, re.IGNORECASE) for p in artifact_patterns)
        if not is_artifact:
            cleaned_lines.append(line)

    # --- 2. Collapse duplicate consecutive bold headings (from overlap stitching) ---
    # Pattern: **Heading** appears twice in a row (possibly separated by blank lines)
    result_lines = []
    last_heading = None
    for line in cleaned_lines:
        stripped = line.strip()
        heading_match = re.match(r'^(\*\*.+?\*\*)\s*$', stripped)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            if current_heading == last_heading:
                # Skip duplicate heading — keep first occurrence
                continue
            last_heading = current_heading
        elif stripped:  # Non-empty, non-heading line resets heading tracker
            last_heading = None
        result_lines.append(line)

    text = '\n'.join(result_lines)

    # --- 3. Collapse 3+ consecutive blank lines to 2 ---
    text = re.sub(r'\n{3,}', '\n\n', text)

    # --- 4. Strip trailing whitespace from each line ---
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    return text.strip()

EXECUTIVE_SUMMARY_PROMPT = """Generate a structured executive summary from the following meeting notes.

### STRUCTURE:
1. **Key Takeaways** (3-5 bullet points): The most important findings, decisions, or insights from the meeting.
2. **Critical Data Points**: All significant numbers, metrics, percentages, and financial figures mentioned.
3. **Notable Quotes/Positions**: Any strong opinions, definitive statements, or notable positions taken by participants.
4. **Risks & Concerns**: Any risks, challenges, or concerns raised during the meeting.
5. **Action Items / Next Steps**: Any follow-ups, commitments, or next steps identified.

### RULES:
- Be specific — include actual numbers, names, and dates rather than vague references.
- Keep each section concise but complete.
- Do not introduce information not present in the notes.

---
**MEETING NOTES:**
{notes}
"""

REFINEMENT_INSTRUCTIONS = {
    "Expert Meeting": "Pay special attention to industry jargon, technical terms, company names, and domain-specific terminology. Preserve all proper nouns exactly.",
    "Earnings Call": "Pay special attention to financial terminology (EPS, EBITDA, basis points, margin, guidance, revenue, etc.), company names, ticker symbols, analyst names, and numerical data. Preserve all figures exactly as spoken.",
    "Management Meeting": "Pay special attention to names of attendees, action item owners, project names, deadlines, and organizational terminology.",
    "Internal Discussion": "Pay special attention to participant names, project/product names, technical terms, and any referenced documents or systems.",
}

# --- OTG NOTES PROMPTS ---

OTG_EXTRACT_PROMPT = """Analyze the following meeting notes and extract structured metadata. Return ONLY valid JSON with no other text.

{{
  "entities": ["list of company names, product names, and proper nouns mentioned"],
  "people": ["list of people mentioned by name or role"],
  "sector": "the industry sector these notes relate to (e.g., Quick Commerce, Fintech, SaaS, Healthcare, etc.)",
  "topics": ["list of 5-12 distinct topics/themes discussed in the notes, each as a short phrase"]
}}

---
**NOTES:**
{notes}
"""

OTG_CONVERT_PROMPT = """You are writing informal channel check notes — the kind an equity research analyst sends to their team after speaking with industry contacts.

### TASK:
Convert the detailed meeting notes below into a short, plain-text research note.

### STYLE (follow exactly):

1. TITLE: A short, natural title on the first line. Examples: "Channel checks on Quick commerce", "Checks on Hero Motocorp", "Hero demand checks". Keep it simple — no formatting, no colons.

2. INTRO: One sentence starting with "We spoke with..." describing who you spoke with (role/expertise, NOT their name) and what you wanted to understand. Then on the same line or next: "Following were the KTAs:"

3. BODY: Write 4-7 short paragraphs of plain flowing text. Each paragraph makes one clear point.
   - ABSOLUTELY NO markdown formatting. No bold (**), no bullets (-), no numbered lists, no headers (#). Just plain text paragraphs.
   - Use simple, direct language. Write like you're sending a quick note to your team, not writing a formal report.
   - Attribute findings to the source naturally: "The expert estimates...", "She didn't share...", "Dealers felt...", "Managers mentioned...", "Our checks highlight...", "He pointed out..."
   - Weave in your own analyst commentary where relevant: "We will need to monitor...", "This makes it tricky because...", "We have observed earlier that..."

4. TONE: {tone}
   - As Is: Present findings exactly as stated in the notes. Do not add any positive or negative framing — reproduce the sentiment already present in the source material.
   - Very Positive: Frame findings constructively. Strengths, growth, advantages. Challenges are temporary.
   - Positive: Generally constructive. Risks acknowledged but opportunities emphasized.
   - Neutral: Balanced. Facts presented objectively.
   - Negative: Risks and structural problems emphasized. Positive developments are insufficient.
   - Very Negative: Fundamental weaknesses, unsustainable practices. Deeply problematic framing.

5. DATA: {number_focus_instruction}

6. LENGTH: {length_instruction}

7. FOCUS ENTITIES: Center the note around: {entities}. Other entities can appear for context.

8. FOCUS TOPICS: Focus on: {topics}

{custom_instructions_block}

### OUTPUT:
Return ONLY the plain-text note. No preamble, no commentary, no markdown formatting whatsoever.

---
SOURCE NOTES:
{notes}
"""

OTG_REFINE_CHUNK_PROMPT = """You are a research analyst extracting structured Q&A notes from a segment of meeting notes.

Your task: Identify all questions asked and their corresponding responses. Structure them clearly so key information is easy to find.

Rules:
- Restate each question clearly in **bold** on its own line — no "Q:" prefix, no label.
- Use bullet points (-) immediately below for each distinct answer point.
- ONLY capture content from responses/answers. Do NOT transcribe question text as note content.
- Preserve every specific detail: numbers (%, ₹, $, volumes, timelines), names, company mentions, data points.
- If a passage has no clear Q&A structure, organise it by **bold topic header** with bullet points.
- Be comprehensive — every substantive point in the answer gets its own bullet.
- Raw and unpolished is fine. Abbreviate freely (Rev, Vol, GM, EBITDA, QoQ, YoY, etc.).

---
NOTES SEGMENT {chunk_num} of {total_chunks}:
{chunk}"""


# --- INVESTMENT ANALYST PROCESSING PROMPTS ---

IA_MANAGEMENT_KTA_PROMPT = """You are a senior equity research analyst processing a Company Management Meeting transcript.

Generate exactly two sections in this order. You MUST use the exact section headers shown below — do not rename, reformat, or omit them:

KEY TAKEAWAYS

- Map findings to the framework below. Only include sections the meeting covered meaningfully.
- 5–6 bullets in total across all sections. Each bullet is one short, punchy sentence — no padding.
- Do NOT start a bullet with a label or category prefix (e.g., do NOT write "Revenue: ..." or "Execution: ...").
- Include numbers stated (%, bps, ₹, $, multiples, timelines). State direction where clear.
- If management was vague, say so in one brief phrase. No interpretation beyond what was stated.

Framework (in order): Strategy → Industry → Thematic → Org/Structure → Execution → Revenue → Margins → Capital Alloc. → Mgmt Culture

Format — No bold section header, then bullet(s):
- Volume-led growth expected in H2; no price increase guidance.
- EBITDA margin expansion of 20–30 bps expected over the next 2–3 quarters.
- Net debt declining; net-cash target by FY26 — capex quantum not shared.
- Supply chain on track; vague on exact timeline.

---

ROUGH NOTES

IMPORTANT: The text "ROUGH NOTES" above is your required section header. Output it exactly as shown — plain text, on its own line, preceded by "---". Do not use markdown formatting (no ##, no bold) for this header.

- Capture ALL substantive points — comprehensive, not selective.
- Neutral meeting notes. State what was said. No spin.
- Organise by topic with bold headers. Fewer, denser bullets — aim for ~25% of the bullet count you would otherwise use by consolidating related points into a single longer sentence.
- Each bullet should be a complete sentence that bundles multiple related details together (numbers, direction, caveats, qualitative colour) rather than splitting them across separate lines.
- Abbreviations: Mgmt, Rev, Vol, ASP, GM, EBITDA, QoQ, YoY, H1, H2, FY, bps, capex, opex, D/E, WC, etc.
- Include qualitative context alongside numbers — what was stressed, what was avoided.
- No positive/negative spin.
- If unclear or unquantified → note it inline within the sentence.
- In Q&A-style transcripts: capture ONLY management's responses. Use the question only to identify the topic heading.

Format: Bold topic headers, dashes (-) under each.
Use sentence case for all headings—capitalize only the first word and proper nouns; do not use title case.

---
TRANSCRIPT:
{transcript}
"""

IA_EXPERT_KTA_PROMPT = """You are a senior equity research analyst processing an Expert / Industry Expert / Channel Check Meeting transcript.

Generate exactly two sections in this order. You MUST use the exact section headers shown below — do not rename, reformat, or omit them:

KEY TAKEAWAYS

- Map findings to the framework below. Only include sections the meeting covered meaningfully.
- 5–6 bullets in total across all sections. Each bullet is one short, punchy sentence — no padding.
- Do NOT start a bullet with a label or category prefix (e.g., do NOT write "Inventory: ..." or "Demand: ...").
- Include numbers stated (%, bps, ₹, $, multiples, timelines, volumes). State direction where clear.
- Tag the source type naturally within the sentence — weave in [Expert view], [Channel check], or [Industry data] where relevant.
- If the expert was vague, say so briefly. No interpretation beyond what was stated.

Framework (in order): Industry → Demand → Channel/Trade → Inventory → Pricing → Margins → Competition → Regulatory/Macro → Outlook

Format:
- [Channel check] Dealer inventory at 45–60 days vs. norm of 30 — destocking ongoing.
- [Expert view] Demand weakening in Tier-2 cities; discretionary most hit.
- [Industry data] Organised players gaining ~200 bps share annually from unorganised.
- Expert unclear on recovery timeline; cautious on H1.

---

ROUGH NOTES

IMPORTANT: The text "ROUGH NOTES" above is your required section header. Output it exactly as shown — plain text, on its own line, preceded by "---". Do not use markdown formatting (no ##, no bold) for this header.

- Capture ALL substantive points — comprehensive, not selective.
- Neutral meeting notes. State what was said. No spin.
- Organise by topic with bold headers. Fewer, denser bullets — aim for ~25% of the bullet count you would otherwise use by consolidating related points into a single longer sentence.
- Each bullet should be a complete sentence that bundles multiple related details together (numbers, direction, caveats, qualitative colour) rather than splitting them across separate lines.
- Abbreviations: Expert, Ch-check, Rev, GM, EBITDA, QoQ, YoY, H1, H2, FY, bps, T2, T3, ASP, inv, dist, etc.
- Include qualitative context alongside numbers — what was stressed, what was avoided, any caveats.
- No positive/negative spin.
- If unclear or unquantified → note it inline within the sentence.
- In Q&A-style transcripts: capture ONLY the expert's responses. Use the question only to identify the topic heading.

Format: Bold topic headers, dashes (-) under each.
Format headings in sentence case. Only capitalize the first word and proper nouns. Do not capitalize every word.
---
TRANSCRIPT:
{transcript}
"""

IA_REFINE_CHUNK_PROMPT = """You are a research analyst cleaning up a segment of a meeting transcript.

Your task: Identify all questions asked and the corresponding management/expert responses. Restructure them clearly.

Rules:
- Restate each question clearly in **bold** on its own line — no "Q:" prefix, no label.
- Use bullet points (-) immediately below for each distinct point made in the response.
- ONLY capture content from responses/answers. Do NOT include question text as note content.
- Preserve every specific detail: numbers (%, bps, ₹, $, timelines), names, entities, data points, qualifiers.
- Preserve the speaker's tone and caveats (confident, cautious, vague, speculative).
- If a passage is not Q&A (e.g., opening remarks), organise it under a **bold topic header** with bullets.
- Be comprehensive — every substantive point in the answer gets its own bullet.
- Raw and unpolished is fine. Abbreviate freely.

---
TRANSCRIPT SEGMENT {chunk_num} of {total_chunks}:
{chunk}"""

IA_TONE_INSTRUCTIONS = {
    "Very Positive": "Frame Output 1 findings in the most constructive investment light. Lead with strengths, growth, and opportunity. Challenges are acknowledged only as temporary or manageable context.",
    "Positive": "Frame Output 1 findings constructively. Opportunities lead. Risks acknowledged but not alarming. Overall tone is favourable.",
    "Neutral": "Frame Output 1 findings objectively. Present facts as stated. Balanced where evidence is mixed. No tilting positive or negative.",
    "Negative": "Frame Output 1 findings with risks and headwinds leading. Positives are noted but insufficient to offset structural concerns.",
    "Very Negative": "Frame Output 1 findings around structural problems, execution gaps, and risks. Even positives are presented as temporary or inadequate.",
}

# --- EARNINGS CALL MULTI-FILE ANALYSIS PROMPTS ---

EC_TOPIC_DISCOVERY_PROMPT = """You are an expert equity research analyst. Analyze the following earnings call transcripts and identify the key topics discussed.

### TASK:
From the transcripts below, extract a structured topic hierarchy. The topics should reflect the actual business structure and discussion themes of this company/group.

### OUTPUT FORMAT:
Return ONLY valid JSON with no other text, using this exact structure:
{{
  "company_name": "The company or group name",
  "primary_topics": [
    {{
      "name": "Primary Topic Name (e.g., brand name, business segment, division)",
      "description": "Brief description of what this covers",
      "sub_topics": [
        "Sub-topic 1 (e.g., menu innovation, unit economics, store expansion)",
        "Sub-topic 2",
        "Sub-topic 3"
      ]
    }}
  ],
  "cross_cutting_topics": [
    {{
      "name": "Cross-cutting Topic Name (e.g., Capital Allocation, Management Changes, Macro Environment)",
      "description": "Brief description"
    }}
  ]
}}

### GUIDELINES:
1. **Primary topics** are business segments, brands, divisions, or major product lines (e.g., for Jubilant FoodWorks: "Dominos India", "Popeyes", "Dunkin Donuts", "Hong's Kitchen")
2. **Sub-topics** under each primary topic are recurring themes like: strategy, menu innovation, store expansion, unit economics, competitive positioning, pricing, customer acquisition, operational efficiency, supply chain, marketing, org structure changes, incentive changes, etc.
3. **Cross-cutting topics** span the entire company: capital allocation, management commentary, guidance, macro environment, regulatory, ESG, etc.
4. Be SPECIFIC to this business — don't use generic templates. The topics should reflect what is ACTUALLY discussed in these transcripts.
5. Include 3-8 primary topics and 3-10 sub-topics per primary topic, based on what the transcripts actually cover.

---
**TRANSCRIPTS:**

{transcripts}
"""

EC_MULTI_FILE_NOTES_PROMPT = """### **EARNINGS CALL NOTES — STRUCTURED BY TOPICS**

You are generating detailed earnings call notes from the transcript below. Structure your notes STRICTLY under the provided topic hierarchy.

### TOPIC STRUCTURE TO FOLLOW:
{topic_structure}

### RULES:
1. For each topic and sub-topic, extract ALL relevant information from the transcript.
2. If a topic/sub-topic has no relevant information in this transcript, write "No specific commentary in this quarter." under it — do NOT skip the heading.
3. Use **bold headings** for primary topics and sub-topics. Use bullet points for details.
4. **PRIORITY #1: CAPTURE ALL FINANCIAL DATA.** Revenue, margins, EPS, guidance ranges, growth rates, basis points, dollar amounts — every number matters.
5. **PRIORITY #2: CAPTURE FORWARD GUIDANCE.** Any forward-looking statements, guidance ranges, management expectations, or outlook commentary.
6. **PRIORITY #3: PRESERVE MANAGEMENT TONE.** Note confidence, caution, hedging language, or changes from prior quarter tone.
7. **PRIORITY #4: CAPTURE SEGMENT/VERTICAL DETAIL.** Business segment breakdowns, geographic splits, and vertical-specific commentary.
8. Include the quarter/period identifier at the top of your notes if mentioned in the transcript.

### FORMAT EXAMPLE:
**[Primary Topic: Brand/Segment Name]**

**[Sub-topic: Strategy]**
- Bullet point with detail...
- Another bullet point...

**[Sub-topic: Unit Economics]**
- Bullet point with detail...

---
**TRANSCRIPT ({file_label}):**
{transcript}
"""

EC_MULTI_FILE_STITCH_HEADER = """# Earnings Call Topic Analysis — {company_name}
*Generated on {date}*
*Files analyzed: {file_count}*

---

"""

# --- REPORT COMPARISON PROMPT CONSTANTS ---

RC_DIMENSION_DISCOVERY_PROMPT = """You are an expert equity research analyst specializing in annual report analysis. Analyze the following annual reports and identify the key QUALITATIVE dimensions that can be meaningfully compared across years.

### TASK:
From the reports below, extract a structured set of comparison dimensions. Focus ONLY on qualitative and strategic aspects — NOT financial numbers (those will differ year to year and are not the focus).

### OUTPUT FORMAT:
Return ONLY valid JSON with no other text, using this exact structure:
{{
  "company_name": "The company or group name",
  "report_years": ["Year 1", "Year 2", ...],
  "comparison_dimensions": [
    {{
      "name": "Dimension Name",
      "description": "Brief description of what this covers",
      "sub_dimensions": [
        "Sub-dimension 1",
        "Sub-dimension 2"
      ]
    }}
  ]
}}

### FOCUS AREAS (use these as guidance, but be specific to what is actually in the reports):
1. **Management Commentary & Tone** — How does the CEO/Chairman letter read? What is the tone — optimistic, cautious, defensive? What themes are emphasized?
2. **Strategic Direction & Priorities** — What strategic pillars are highlighted? How have priorities shifted? New initiatives vs. continued focus areas?
3. **Business Structure & Organization** — How is the business organized (segments, divisions, subsidiaries)? Any restructuring, new segments, or organizational changes?
4. **Leadership & Governance** — Board composition changes, key management changes, succession planning, governance structure evolution?
5. **Incentive Structures & Compensation** — How are executives compensated? What metrics drive bonuses/ESOPs? Any changes in incentive design?
6. **Risk Factors & Mitigation** — What risks are highlighted? How has the risk landscape changed? New risks vs. dropped risks?
7. **Capital Allocation Philosophy** — How does management talk about deploying capital? Dividends vs. buybacks vs. reinvestment priorities?
8. **ESG / Sustainability** — Environmental, social, governance initiatives. How prominent is ESG in the narrative? Any new commitments?
9. **Market & Competitive Positioning** — How does the company describe its competitive position? Market share commentary, moats, differentiation?
10. **Growth Levers & Outlook** — What growth avenues are highlighted? Organic vs. inorganic? Geographic vs. product expansion?
11. **Culture & People** — Employee-related commentary, talent strategy, culture statements, DEI initiatives?
12. **Technology & Digital** — Digital transformation initiatives, technology investments, IT strategy evolution?

### GUIDELINES:
- Be SPECIFIC to this company — identify dimensions that are actually discussed in these reports.
- Include 5-12 dimensions with 2-6 sub-dimensions each, based on what the reports actually cover.
- The dimensions should enable meaningful year-over-year comparison of QUALITATIVE changes.
- Do NOT include dimensions focused on specific numbers or financial metrics.

---
**ANNUAL REPORTS:**

{reports}
"""

RC_PER_REPORT_EXTRACTION_PROMPT = """### **ANNUAL REPORT ANALYSIS — QUALITATIVE EXTRACTION**

You are extracting qualitative information from an annual report for a specific set of comparison dimensions. Focus on WHAT management says, HOW they say it, and WHAT has changed — NOT on specific numbers.

### DIMENSIONS TO EXTRACT:
{dimension_structure}

### RULES:
1. For each dimension and sub-dimension, extract ALL relevant qualitative information from this report.
2. If a dimension/sub-dimension has no relevant information in this report, write "Not addressed in this report." — do NOT skip the heading.
3. Use **bold headings** for dimensions and sub-dimensions. Use bullet points for details.
4. **FOCUS ON:**
   - Management's language, tone, and emphasis
   - Strategic statements and directional commentary
   - Organizational descriptions and structural details
   - Policy descriptions (compensation, governance, risk)
   - Qualitative characterizations ("strong growth", "challenging environment", "transformational year")
   - Changes in emphasis or new themes compared to what might be typical
5. **AVOID:**
   - Specific revenue/profit/margin numbers (unless they illustrate a qualitative point about strategy)
   - Detailed financial tables or ratios
   - Restating numbers that will obviously differ between years
6. Capture DIRECT QUOTES from management where they are particularly revealing of tone or strategic intent.
7. Note the year/period this report covers at the top.

### FORMAT:
**[Dimension: Name]**

**[Sub-dimension: Name]**
- Bullet point with qualitative detail...
- Another bullet point...

---
**ANNUAL REPORT ({file_label}):**
{report_text}
"""

RC_COMPARISON_PROMPT = """### **ANNUAL REPORT COMPARISON — YEAR-OVER-YEAR QUALITATIVE ANALYSIS**

You are an expert analyst comparing annual reports from different years for the same company. Below are the qualitative extractions from each year's report. Your task is to produce a structured comparison highlighting what has CHANGED, what has STAYED THE SAME, and what is NEW or DROPPED.

### COMPANY: {company_name}
### REPORTS COMPARED: {report_labels}

### COMPARISON DIMENSIONS:
{dimension_structure}

### EXTRACTED DATA FROM EACH REPORT:
{per_report_extractions}

### YOUR TASK:
For each dimension and sub-dimension, produce a comparison that answers:
1. **What changed?** — Shifts in tone, emphasis, strategy, structure, or policy between years.
2. **What remained consistent?** — Themes or approaches that persisted across years.
3. **What is new?** — Themes, initiatives, or structural elements that appear in later reports but not earlier ones.
4. **What was dropped?** — Items emphasized in earlier reports but absent or de-emphasized in later ones.

### FORMAT:
For each dimension, structure your output as:

## [Dimension Name]

### [Sub-dimension Name]

**Evolution across years:**
- [Year-over-year comparison points as bullets]

**Key shifts:**
- [Most significant changes highlighted]

**Consistency:**
- [What stayed the same]

### RULES:
1. Be SPECIFIC — cite which year said what. Use phrases like "In FY2022, management emphasized X, while in FY2024, the focus shifted to Y."
2. Include direct management quotes where they illustrate a meaningful shift.
3. Do NOT simply list what each year said — actually COMPARE and CONTRAST.
4. Highlight the most SIGNIFICANT shifts prominently. Minor changes can be noted briefly.
5. If a dimension shows no meaningful change across years, say so explicitly — consistency is also a finding.
6. Order the comparison chronologically (earliest to latest year).
7. At the end, include a section called "## Key Takeaways" with 5-10 bullet points summarizing the most important qualitative shifts across all dimensions.

---
"""

RC_STITCH_HEADER = """# Annual Report Comparison — {company_name}
*Generated on {date}*
*Reports compared: {report_labels}*

---

"""

# --- 3. STATE & DATA MODELS ---
@dataclass
class AppState:
    input_method: str = "Paste Text"
    selected_meeting_type: str = "Expert Meeting"
    selected_note_style: str = "Option 2: Less Verbose"
    earnings_call_mode: str = "Generate New Notes"
    selected_sector: str = "IT Services"
    notes_model: str = "Gemini 2.5 Pro"
    refinement_model: str =  "Gemini 2.5 Flash Lite"
    transcription_model: str =  "Gemini 3.0 Flash"
    chat_model: str = "Gemini 2.5 Pro"
    refinement_enabled: bool = True
    add_context_enabled: bool = False
    context_input: str = ""
    speakers: str = ""
    earnings_call_topics: str = ""
    existing_notes_input: str = ""
    text_input: str = ""
    uploaded_file: Optional[Any] = None
    audio_recording: Optional[Any] = None
    processing: bool = False
    active_note_id: Optional[str] = None
    error_message: Optional[str] = None
    fallback_content: Optional[str] = None

# --- 4. CORE PROCESSING & UTILITY FUNCTIONS ---
def sanitize_input(text: str) -> str:
    """Removes keywords commonly used in prompt injection attacks."""
    if not isinstance(text, str):
        return ""
    injection_patterns = [
        r'ignore all previous instructions',
        r'you are now in.*mode',
        r'stop being an ai',
        r'disregard.*(?:above|prior|previous)',
        r'new instructions:',
        r'system:\s*override',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def safe_get_token_count(response):
    """Safely get the token count from a response, returning 0 if unavailable."""
    try:
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            return getattr(response.usage_metadata, 'total_token_count', 0)
    except (AttributeError, ValueError):
        pass
    return 0

def generate_with_retry(model, prompt_or_contents, max_retries=3, stream=False, generation_config=None):
    """Wrapper around generate_content with exponential backoff for transient API failures."""
    kwargs = {"stream": stream}
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt_or_contents, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_transient = any(kw in error_str for kw in [
                '429', '503', '500', 'deadline', 'timeout', 'unavailable', 'resource_exhausted'
            ])
            if is_transient and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

def stream_and_collect(response, placeholder=None):
    """Consume a streaming response, optionally displaying progress. Returns (full_text, token_count)."""
    full_text = ""
    update_counter = 0
    for chunk in response:
        if chunk.parts:
            full_text += chunk.text
            update_counter += 1
            # Throttle UI updates to every 5 chunks to reduce flickering
            if placeholder and update_counter % 5 == 0:
                word_count = len(full_text.split())
                placeholder.caption(f"Streaming... {word_count:,} words generated")
    if placeholder:
        placeholder.empty()
    token_count = safe_get_token_count(response)
    return full_text, token_count

def copy_to_clipboard_button(text: str, button_label: str = "Copy Notes"):
    """Render a button that copies text to the clipboard using the browser Clipboard API."""
    # Adapt button colors to the current theme (light vs dark mode)
    theme = st.context.theme
    bg_color = theme.get("primaryColor", "#FF4B4B")
    text_color = theme.get("backgroundColor", "#FFFFFF")

    # Use JSON encoding to safely embed arbitrary text in a JS string literal
    json_encoded = json.dumps(text)
    safe_label = html_module.escape(button_label)
    components.html(
        f"""
        <button onclick="copyText()" aria-label="{safe_label}" role="button" tabindex="0"
            onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();copyText();}}"
            style="
                background-color:{bg_color}; color:{text_color}; border:none; padding:0.4rem 1rem;
                border-radius:0.3rem; cursor:pointer; font-size:0.875rem; width:100%;
                transition: opacity 0.15s ease, box-shadow 0.15s ease;
            "
            onmouseover="this.style.opacity='0.85'"
            onmouseout="this.style.opacity='1'"
            onfocus="this.style.boxShadow='0 0 0 2px {bg_color}40'"
            onblur="this.style.boxShadow='none'"
        >{safe_label}</button>
        <script>
        function copyText() {{
            const text = {json_encoded};
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.querySelector('button');
                btn.textContent = 'Copied!';
                btn.setAttribute('aria-label', 'Copied to clipboard');
                setTimeout(() => {{
                    btn.textContent = {json.dumps(button_label)};
                    btn.setAttribute('aria-label', {json.dumps(button_label)});
                }}, 2000);
            }}).catch(() => {{
                const btn = document.querySelector('button');
                btn.textContent = 'Failed';
                setTimeout(() => btn.textContent = {json.dumps(button_label)}, 2000);
            }});
        }}
        </script>
        """,
        height=45,
    )

@st.cache_data(ttl=3600)
def get_file_content(uploaded_file, audio_recording=None) -> Tuple[Optional[str], str, Optional[bytes]]:
    """
    Returns: (text_content_or_indicator, file_name, pdf_bytes_if_pdf)
    """
    # Priority 1: Audio Recording
    if audio_recording:
        return "audio_file", "Microphone Recording.wav", None

    # Priority 2: Uploaded File
    if uploaded_file:
        name = uploaded_file.name
        file_bytes_io = io.BytesIO(uploaded_file.getvalue())
        ext = os.path.splitext(name)[1].lower()

        try:
            if ext == ".pdf":
                reader = PyPDF2.PdfReader(file_bytes_io)
                if reader.is_encrypted: return "Error: PDF is encrypted.", name, None
                content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                return (content, name, uploaded_file.getvalue()) if content else ("Error: No text found in PDF.", name, None)

            elif ext in [".txt", ".md"]:
                return file_bytes_io.read().decode("utf-8"), name, None

            elif ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
                return "audio_file", name, None

        except Exception as e:
            return f"Error: Could not process file {name}. Details: {str(e)}", name, None

    return None, "Unknown", None

@st.cache_data
def db_get_sectors() -> dict:
    return database.get_sectors()

def create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Creates overlapping chunks of text, ensuring the final fragment is always included."""
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if (i + chunk_size) >= len(words):
            break

    return chunks


def _create_enhanced_context_from_notes(notes_text, chunk_number=0):
    """Create richer context from previous notes"""
    if not notes_text or not notes_text.strip():
        return ""

    headings = re.findall(r"(\*\*.*?\*\*)", notes_text)

    if not headings:
        return ""

    context_headings = headings[-3:] if len(headings) >= 3 else headings

    context_parts = [
        f"**Chunk #{chunk_number} Context Summary:**",
        f"- Total sections processed so far: {len(headings)}",
        f"- Recent topics: {', '.join(q.strip('*') for q in context_headings[-2:])}",
        f"- Last section processed: {headings[-1]}"
    ]

    last_heading = headings[-1]
    answer_match = re.search(
        re.escape(last_heading) + r"(.*?)(?=\*\*|$)",
        notes_text,
        re.DOTALL
    )
    if answer_match:
        last_content = answer_match.group(1).strip()
        context_parts.append(f"- Last section content:\n{last_content[:300]}...")

    return "\n".join(context_parts)

def _get_base_prompt_for_type(state):
    """Returns the base prompt instructions for the selected meeting type."""
    mt = state.selected_meeting_type
    if mt == "Expert Meeting":
        if state.selected_note_style == "Option 1: Detailed & Strict":
            return EXPERT_MEETING_DETAILED_PROMPT
        else:
            return EXPERT_MEETING_CONCISE_PROMPT
    elif mt == "Earnings Call":
        topic_instructions = state.earnings_call_topics or "Identify logical themes and use them as bold headings."
        return EARNINGS_CALL_PROMPT.format(topic_instructions=topic_instructions)
    elif mt == "Management Meeting":
        return MANAGEMENT_MEETING_PROMPT
    elif mt == "Internal Discussion":
        return INTERNAL_DISCUSSION_PROMPT
    elif mt == "Custom":
        sanitized = sanitize_input(state.context_input)
        if sanitized:
            return f"Follow the user's instructions to generate meeting notes.\n\n**USER INSTRUCTIONS:**\n{sanitized}"
        return "Generate comprehensive meeting notes capturing all key points, decisions, data, and action items. Use **bold headings** to organize by topic and bullet points for details."
    return ""

def get_dynamic_prompt(state: AppState, transcript_chunk: str) -> str:
    base = _get_base_prompt_for_type(state)
    sanitized_context = sanitize_input(state.context_input)
    context_section = f"**ADDITIONAL CONTEXT:**\n{sanitized_context}" if state.add_context_enabled and sanitized_context else ""

    if state.selected_meeting_type == "Earnings Call" and state.earnings_call_mode == "Enrich Existing Notes":
        return f"Enrich the following existing notes based on the new transcript. Maintain the same structure and format.\n\n{base}\n\n**EXISTING NOTES:**\n{state.existing_notes_input}\n\n**NEW TRANSCRIPT:**\n{transcript_chunk}"

    return f"{base}\n\n{context_section}\n\n**MEETING TRANSCRIPT:**\n{transcript_chunk}"

def validate_inputs(state: AppState) -> Optional[str]:
    if state.input_method == "Paste Text" and not state.text_input.strip():
        return "Please paste a transcript."

    if state.input_method == "Upload / Record":
        if not state.uploaded_file and not state.audio_recording:
             return "Please upload a file or record audio."

        if state.uploaded_file and not state.audio_recording:
            size_mb = state.uploaded_file.size / (1024 * 1024)
            ext = os.path.splitext(state.uploaded_file.name)[1].lower()
            if ext == ".pdf" and size_mb > MAX_PDF_MB:
                return f"PDF is too large ({size_mb:.1f}MB). Limit: {MAX_PDF_MB}MB."
            elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac'] and size_mb > MAX_AUDIO_MB:
                return f"Audio is too large ({size_mb:.1f}MB). Limit: {MAX_AUDIO_MB}MB."

    if state.selected_meeting_type == "Earnings Call" and state.earnings_call_mode == "Enrich Existing Notes" and not state.existing_notes_input:
        return "Please provide existing notes for enrichment mode."
    return None

def _get_cached_model(model_display_name: str) -> genai.GenerativeModel:
    """Return a cached GenerativeModel instance, creating it only if the model name changed."""
    cache_key = "_model_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    # Defensive: handle invalid model names gracefully
    if model_display_name not in AVAILABLE_MODELS:
        model_display_name = list(AVAILABLE_MODELS.keys())[0]  # fallback to first model
    model_id = AVAILABLE_MODELS[model_display_name]
    if model_id not in st.session_state[cache_key]:
        st.session_state[cache_key][model_id] = genai.GenerativeModel(model_id)
    return st.session_state[cache_key][model_id]

def is_mobile_device() -> bool:
    """Check if likely a mobile device based on viewport. Returns False on server-side."""
    # Note: This is a heuristic. Streamlit doesn't expose device info directly.
    # We'll use a session state flag that can be set via JS, defaulting to False.
    return st.session_state.get("_is_mobile", False)

class ProgressTracker:
    """Manages progress bar and status updates during processing."""

    # Define the processing steps and their approximate weights (totaling 100)
    STEPS = {
        "prepare": {"weight": 5, "label": "Preparing Source Content"},
        "transcribe": {"weight": 15, "label": "Transcribing Audio"},
        "refine": {"weight": 25, "label": "Refining Transcript"},
        "generate": {"weight": 47, "label": "Generating Notes"},
        "cleanup": {"weight": 3, "label": "Cleaning Up Notes"},
        "save": {"weight": 5, "label": "Saving to Database"},
    }

    def __init__(self, status_container):
        self.status = status_container
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.current_progress = 0
        self.completed_steps = set()

    def update(self, step: str, sub_progress: float = 0, detail: str = ""):
        """
        Update progress bar and status.
        step: one of the STEPS keys
        sub_progress: 0-1 progress within the current step
        detail: additional detail text
        """
        # Calculate base progress from completed steps
        base = sum(self.STEPS[s]["weight"] for s in self.completed_steps)

        # Add current step's partial progress
        if step in self.STEPS:
            step_weight = self.STEPS[step]["weight"]
            current = base + (step_weight * sub_progress)
        else:
            current = base

        self.current_progress = min(current / 100, 1.0)
        self.progress_bar.progress(self.current_progress)

        # Update status text
        label = self.STEPS.get(step, {}).get("label", step)
        pct = int(self.current_progress * 100)
        status_msg = f"**{pct}%** - {label}"
        if detail:
            status_msg += f" ({detail})"
        self.status_text.markdown(status_msg)

    def complete_step(self, step: str):
        """Mark a step as completed."""
        self.completed_steps.add(step)
        self.update(step, 1.0)

    def finish(self):
        """Mark all progress complete."""
        self.progress_bar.progress(1.0)
        self.status_text.markdown("**100%** - Complete!")

def send_browser_notification(title: str, body: str):
    """Send a browser notification using the Notifications API."""
    safe_title = json.dumps(title)
    safe_body = json.dumps(body)

    components.html(
        f"""
        <script>
        (function() {{
            if (!("Notification" in window)) return;
            var opts = {{body: {safe_body}, icon: "https://placehold.co/64x64?text=SN", tag: "synthnotes-complete"}};
            if (Notification.permission === "granted") {{
                new Notification({safe_title}, opts);
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(permission) {{
                    if (permission === "granted") new Notification({safe_title}, opts);
                }});
            }}
        }})();
        </script>
        """,
        height=0,
    )

def process_and_save_task(state: AppState, status_ui, progress: ProgressTracker):
    start_time = time.time()
    notes_model = _get_cached_model(state.notes_model)
    refinement_model = _get_cached_model(state.refinement_model)
    transcription_model = _get_cached_model(state.transcription_model)

    progress.update("prepare", 0, "Loading input...")
    raw_transcript, file_name = "", "Pasted Text"
    pdf_bytes_data = None

    # Handle input (File, Recording, or Text)
    if state.input_method == "Upload / Record":
        file_type, name, pdf_bytes = get_file_content(state.uploaded_file, state.audio_recording)
        file_name = name
        pdf_bytes_data = pdf_bytes

        if file_type == "audio_file":
            progress.update("transcribe", 0, "Processing audio file...")

            if state.audio_recording:
                audio_bytes = state.audio_recording.getvalue()
            else:
                audio_bytes = state.uploaded_file.getvalue()

            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as audio_err:
                raise ValueError(f"Failed to process audio file. It may be corrupted or in an unsupported format. Details: {audio_err}")

            chunk_length_ms = 5 * 60 * 1000
            audio_chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

            all_transcripts, cloud_files, local_files = [], [], []
            try:
                for i, chunk in enumerate(audio_chunks):
                    try:
                        progress.update("transcribe", i / len(audio_chunks), f"Chunk {i+1}/{len(audio_chunks)}")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_f:
                            chunk.export(temp_f.name, format="wav")
                            local_files.append(temp_f.name)
                            cloud_ref = genai.upload_file(path=temp_f.name)
                            cloud_files.append(cloud_ref.name)
                            while cloud_ref.state.name == "PROCESSING": time.sleep(2); cloud_ref = genai.get_file(cloud_ref.name)
                            if cloud_ref.state.name != "ACTIVE": raise Exception(f"Audio chunk {i+1} cloud processing failed.")
                            response = generate_with_retry(transcription_model, ["Transcribe this audio.", cloud_ref])
                            all_transcripts.append(response.text)
                    except Exception as e:
                        raise Exception(f"Transcription failed on chunk {i+1}/{len(audio_chunks)}. Reason: {e}")

                raw_transcript = "\n\n".join(all_transcripts).strip()
                progress.complete_step("transcribe")
            finally:
                for path in local_files: os.remove(path)
                for cloud_name in cloud_files:
                    try: genai.delete_file(cloud_name)
                    except Exception as e: st.warning(f"Could not delete cloud file {cloud_name}: {e}")

        elif file_type is None or file_type.startswith("Error:"):
            raise ValueError(file_type or "Failed to read file content.")
        else:
            raw_transcript = file_type
    elif state.input_method == "Paste Text":
        raw_transcript = state.text_input

    if not raw_transcript or not raw_transcript.strip():
        raise ValueError("Source content is empty or contains only whitespace.")

    # Normalize whitespace and remove excessive blank lines
    raw_transcript = re.sub(r'\n{3,}', '\n\n', raw_transcript.strip())

    # Checkpoint: save raw transcript so it survives connection drops
    st.session_state["_checkpoint_raw_transcript"] = raw_transcript
    st.session_state["_checkpoint_file_name"] = file_name

    final_transcript, refined_transcript, total_tokens = raw_transcript, None, 0

    speakers = sanitize_input(state.speakers)
    speaker_info = f"Participants: {speakers}." if speakers else ""
    refinement_extra = REFINEMENT_INSTRUCTIONS.get(state.selected_meeting_type, "")

    # --- Step 2: Refinement ---
    if state.refinement_enabled:
        progress.complete_step("prepare")
        progress.update("refine", 0, "Starting refinement...")
        words = raw_transcript.split()

        lang_instruction = "IMPORTANT: Your entire output MUST be in English. If the transcript contains Hindi, Hinglish, or any other non-English language, translate all content into clear, natural English while preserving the original meaning, nuance, and speaker intent."

        if len(words) <= CHUNK_WORD_SIZE:
            refine_prompt = f"Refine the following transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info} {refinement_extra}\n{lang_instruction}\n\nTRANSCRIPT:\n{raw_transcript}"
            response = generate_with_retry(refinement_model, refine_prompt)
            refined_transcript = response.text
            total_tokens += safe_get_token_count(response)
        else:
            chunks = create_chunks_with_overlap(raw_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)

            # Pre-build all prompts using raw chunk tails as context (known upfront, enables parallelism)
            prompts = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    prompts.append(f"You are refining a transcript. Correct spelling, grammar, and punctuation. Label speakers clearly if possible. {speaker_info} {refinement_extra}\n{lang_instruction}\n\nTRANSCRIPT CHUNK TO REFINE:\n{chunk}")
                else:
                    prev_chunk_words = chunks[i - 1].split()
                    context = " ".join(prev_chunk_words[-CHUNK_WORD_OVERLAP:])
                    prompts.append(f"""You are continuing to refine a long transcript. Below is the tail end of the previous section for context. Your task is to refine the new chunk provided, ensuring a seamless and natural transition.
{speaker_info} {refinement_extra}
{lang_instruction}
---
CONTEXT FROM PREVIOUS CHUNK (FOR CONTINUITY ONLY):
...{context}
---
NEW TRANSCRIPT CHUNK TO REFINE:
{chunk}""")

            progress.update("refine", 0.1, f"{len(chunks)} chunks in parallel")

            # Process chunks in parallel (max 3 concurrent to respect API rate limits)
            all_refined_chunks = [None] * len(chunks)
            chunk_tokens = [0] * len(chunks)

            def refine_chunk(idx, prompt):
                resp = generate_with_retry(refinement_model, prompt)
                return idx, resp.text, safe_get_token_count(resp)

            with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as executor:
                futures = {executor.submit(refine_chunk, i, p): i for i, p in enumerate(prompts)}
                for future in as_completed(futures):
                    idx, text, tokens = future.result()
                    all_refined_chunks[idx] = text
                    chunk_tokens[idx] = tokens
                    done_count = sum(1 for c in all_refined_chunks if c is not None)
                    progress.update("refine", 0.1 + (0.9 * done_count / len(chunks)), f"{done_count}/{len(chunks)} chunks")

            total_tokens += sum(chunk_tokens)
            refined_transcript = "\n\n".join(c for c in all_refined_chunks if c) if any(all_refined_chunks) else ""

        final_transcript = refined_transcript
    else:
        # Refinement disabled - mark prepare and refine as complete
        progress.complete_step("prepare")
        progress.complete_step("refine")

    # Checkpoint: save refined transcript
    st.session_state["_checkpoint_refined_transcript"] = refined_transcript

    # --- Step 3: Generate Notes ---
    words = final_transcript.split()
    # Earnings calls should not be chunked: their topic-based structure causes
    # repeated sections when the same headings appear across multiple chunks.
    skip_chunking = state.selected_meeting_type == "Earnings Call"
    num_chunks = 1 if skip_chunking else max(1, (len(words) + CHUNK_WORD_SIZE - 1) // CHUNK_WORD_SIZE)
    progress.update("generate", 0, f"{len(words):,} words, {num_chunks} chunk{'s' if num_chunks > 1 else ''}")
    final_notes_content = ""

    if not skip_chunking and len(words) > CHUNK_WORD_SIZE:
        chunks = create_chunks_with_overlap(final_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)

        all_notes_chunks = []
        context_package = ""
        prompt_base = _get_base_prompt_for_type(state)

        for i, chunk in enumerate(chunks):
            progress.update("generate", i / len(chunks), f"Chunk {i+1}/{len(chunks)}")
            prompt_template = PROMPT_INITIAL if i == 0 else PROMPT_CONTINUATION
            prompt = prompt_template.format(base_instructions=prompt_base, chunk_text=chunk, context_package=context_package)

            stream_placeholder = st.empty()
            response = generate_with_retry(notes_model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
            current_notes_text, tokens = stream_and_collect(response, stream_placeholder)
            total_tokens += tokens

            all_notes_chunks.append(current_notes_text)

            cumulative_notes_for_context = "\n\n".join(all_notes_chunks)
            context_package = _create_enhanced_context_from_notes(cumulative_notes_for_context, chunk_number=i + 1)

        if not all_notes_chunks or not any(c.strip() for c in all_notes_chunks):
            raise ValueError("Failed to generate notes from any chunk. Please try again or use a different model.")
        else:
            final_notes_content = all_notes_chunks[0]
            for i in range(1, len(all_notes_chunks)):
                prev_notes = all_notes_chunks[i-1]
                current_notes = all_notes_chunks[i]

                # Match only standalone bold headings (full line), not inline bold within bullets.
                # The old r"(\*\*.*?\*\*)" pattern matched any bold text including inline
                # items like **$5B** or **important**, causing stitch points to land in the
                # middle of bullet content and silently drop portions of the notes.
                HEADING_RE = r"(?m)^(\*\*[^*\n]+\*\*)\s*$"

                last_q_match = list(re.finditer(HEADING_RE, prev_notes))
                if not last_q_match:
                    final_notes_content += "\n\n" + current_notes
                    continue

                last_heading = last_q_match[-1].group(1)

                # Use a line-anchored search so we don't accidentally land on an
                # inline occurrence of the same text inside a bullet point.
                stitch_match = re.search(r"(?m)^" + re.escape(last_heading) + r"\s*$", current_notes)
                stitch_point = stitch_match.start() if stitch_match else -1

                if stitch_point != -1:
                    next_q_match = re.search(HEADING_RE, current_notes[stitch_point + len(last_heading):])
                    if next_q_match:
                        final_notes_content += "\n\n" + current_notes[stitch_point + len(last_heading) + next_q_match.start():]
                    else:
                        final_notes_content += "\n\n" + current_notes[stitch_point + len(last_heading):]
                else:
                    st.warning(f"Could not find stitch point for chunk {i+1}. Appending full chunk; check for duplicates.")
                    final_notes_content += "\n\n" + current_notes

    else:
        prompt = get_dynamic_prompt(state, final_transcript)
        stream_placeholder = st.empty()
        response = generate_with_retry(notes_model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
        final_notes_content, tokens = stream_and_collect(response, stream_placeholder)
        total_tokens += tokens

    # Defensive: ensure we have content
    if not final_notes_content or not final_notes_content.strip():
        raise ValueError("The model returned empty notes. Please try again or use a different model.")

    # --- Step 4: Deterministic cleanup (replaces LLM proofread to avoid content loss) ---
    progress.complete_step("generate")
    was_chunked = not skip_chunking and len(final_transcript.split()) > CHUNK_WORD_SIZE
    if was_chunked:
        progress.update("cleanup", 0, "Cleaning stitching artifacts...")
        final_notes_content = cleanup_stitched_notes(final_notes_content)
        progress.update("cleanup", 1.0, "Done")
    else:
        progress.update("cleanup", 1.0, "Skipped (single-chunk)")
    progress.complete_step("cleanup")

    # --- Step 5: Executive Summary (Expert Meeting Option 3 only) ---
    if state.selected_note_style == "Option 3: Less Verbose + Summary" and state.selected_meeting_type == "Expert Meeting":
        progress.update("save", 0, "Generating executive summary...")
        summary_prompt = EXECUTIVE_SUMMARY_PROMPT.format(notes=final_notes_content)
        response = generate_with_retry(notes_model, summary_prompt)
        final_notes_content += f"\n\n---\n\n{response.text}"
        total_tokens += safe_get_token_count(response)

    # --- Step 6: Save ---
    progress.update("save", 0.5, "Writing to database...")

    note_data = {
        'id': str(uuid.uuid4()), 'created_at': datetime.now().isoformat(), 'meeting_type': state.selected_meeting_type,
        'file_name': file_name, 'content': final_notes_content, 'raw_transcript': raw_transcript,
        'refined_transcript': refined_transcript, 'token_usage': total_tokens,
        'processing_time': time.time() - start_time,
        'pdf_blob': pdf_bytes_data
    }
    try:
        database.save_note(note_data)
    except Exception as db_error:
        st.session_state.app_state.fallback_content = final_notes_content
        raise Exception(f"Processing succeeded, but failed to save the note to the database. You can download the unsaved note below. Error: {db_error}")
    return note_data

# --- 5. UI RENDERING FUNCTIONS ---
def on_sector_change():
    """Callback for sector selectbox. Reads new value from the widget key
    (st.session_state.sector_selector) because on_change fires BEFORE the
    selectbox return value updates state.selected_sector."""
    state = st.session_state.app_state
    new_sector = st.session_state.get("sector_selector", state.selected_sector)
    state.selected_sector = new_sector
    all_sectors = db_get_sectors()
    state.earnings_call_topics = all_sectors.get(new_sector, "")

def render_input_and_processing_tab(state: AppState):
    # --- Source Input ---
    state.input_method = st.pills("Input Method", ["Paste Text", "Upload / Record"], default=state.input_method, key="input_method_pills")

    if state.input_method == "Paste Text":
        state.text_input = st.text_area("Paste source transcript here:", value=state.text_input, height=250, key="text_input_main")
        state.uploaded_file = None
        state.audio_recording = None
    else:
        col_upload, col_record = st.columns(2)
        with col_upload:
            state.uploaded_file = st.file_uploader("Upload a File", type=['pdf', 'txt', 'mp3', 'm4a', 'wav', 'ogg', 'flac'], help="PDF, TXT, MP3, M4A, WAV, OGG, FLAC")
        with col_record:
            state.audio_recording = st.audio_input("Record Microphone")

    # --- Word count preview ---
    preview_text = ""
    if state.input_method == "Paste Text" and state.text_input:
        preview_text = state.text_input
    elif state.input_method == "Upload / Record" and state.uploaded_file:
        ext = os.path.splitext(state.uploaded_file.name)[1].lower()
        if ext not in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            content, _, _ = get_file_content(state.uploaded_file, None)
            if content and not str(content).startswith("Error:") and content != "audio_file":
                preview_text = content

    if preview_text:
        wc = len(preview_text.split())
        num_chunks = max(1, (wc + CHUNK_WORD_SIZE - 1) // CHUNK_WORD_SIZE)
        info = f"**{wc:,}** words"
        if num_chunks > 1:
            info += f" | **{num_chunks}** chunks"
        st.caption(info)
    elif state.input_method == "Upload / Record" and state.uploaded_file:
        ext = os.path.splitext(state.uploaded_file.name)[1].lower()
        if ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            st.caption("Audio file — word count available after transcription")

    st.divider()

    # --- Configuration ---
    st.subheader("Configuration")

    # Meeting type + style on one row for Expert Meeting
    if state.selected_meeting_type == "Expert Meeting":
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            state.selected_meeting_type = st.selectbox("Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type), help=MEETING_TYPE_HELP.get(state.selected_meeting_type, ""))
        with cfg_col2:
            state.selected_note_style = st.selectbox("Note Style", EXPERT_MEETING_OPTIONS, index=EXPERT_MEETING_OPTIONS.index(state.selected_note_style))
    else:
        state.selected_meeting_type = st.selectbox("Meeting Type", MEETING_TYPES, index=MEETING_TYPES.index(state.selected_meeting_type), help=MEETING_TYPE_HELP.get(state.selected_meeting_type, ""))

    if state.selected_meeting_type == "Earnings Call":
        state.earnings_call_mode = st.radio("Mode", EARNINGS_CALL_MODES, horizontal=True, index=EARNINGS_CALL_MODES.index(state.earnings_call_mode))

        all_sectors = db_get_sectors()
        sector_options = ["Other / Manual Topics"] + sorted(list(all_sectors.keys()))

        try:
            current_sector_index = sector_options.index(state.selected_sector)
        except ValueError:
            current_sector_index = 0

        sector_col, manage_col = st.columns([3, 1])
        with sector_col:
            state.selected_sector = st.selectbox("Sector (for Topic Templates)", sector_options, index=current_sector_index, on_change=on_sector_change, key="sector_selector")
        with manage_col:
            st.container(height=28, border=False)  # vertical spacer to align with selectbox
            with st.popover("Manage Sectors", use_container_width=True):
                st.markdown("**Edit or Delete Sector**")
                sector_to_edit = st.selectbox("Select Sector", sorted(list(all_sectors.keys())))

                if sector_to_edit:
                    topics_for_edit = st.text_area("Sector Topics", value=all_sectors[sector_to_edit], key=f"topics_{sector_to_edit}")
                    col1, col2 = st.columns([1,1])
                    if col1.button("Save Changes", key=f"save_{sector_to_edit}"):
                        database.save_sector(sector_to_edit, topics_for_edit); db_get_sectors.clear();
                        st.toast(f"Sector '{sector_to_edit}' updated!"); st.rerun()
                    if col2.button("Delete Sector", type="primary", key=f"delete_{sector_to_edit}"):
                        database.delete_sector(sector_to_edit); db_get_sectors.clear(); state.selected_sector = "Other / Manual Topics"; on_sector_change();
                        st.toast(f"Sector '{sector_to_edit}' deleted!"); st.rerun()

                st.divider()
                st.markdown("**Add a New Sector**")
                new_sector_name = st.text_input("New Sector Name")
                new_sector_topics = st.text_area("Topics for New Sector", key="new_sector_topics")

                if st.button("Add New Sector"):
                    if new_sector_name and new_sector_topics:
                        database.save_sector(new_sector_name, new_sector_topics); db_get_sectors.clear();
                        st.toast(f"Sector '{new_sector_name}' added!"); st.rerun()
                    else:
                        st.warning("Please provide both a name and topics for the new sector.")

        state.earnings_call_topics = st.text_area("Topic Instructions", value=state.earnings_call_topics, height=150, placeholder="Select a sector to load a template, or enter topics manually.")

        if state.earnings_call_mode == "Enrich Existing Notes":
            state.existing_notes_input = st.text_area("Paste Existing Notes to Enrich:", value=state.existing_notes_input)

    elif state.selected_meeting_type == "Custom":
        state.context_input = st.text_area("Custom Instructions", value=state.context_input, height=120, placeholder="Describe how you want the notes structured...")

    # --- Settings & Participants row ---
    col_settings, col_participants = st.columns(2)
    with col_settings:
        with st.popover("Settings & Models", use_container_width=True):
            state.refinement_enabled = st.toggle("Transcript Refinement", value=state.refinement_enabled)
            if state.selected_meeting_type != "Custom":
                state.add_context_enabled = st.toggle("Add General Context", value=state.add_context_enabled)
                if state.add_context_enabled: state.context_input = st.text_area("Context Details:", value=state.context_input, placeholder="e.g., Company Name, Date...")

            st.divider()
            state.notes_model = st.selectbox("Notes Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.notes_model))
            state.refinement_model = st.selectbox("Refinement Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.refinement_model))
            state.transcription_model = st.selectbox("Transcription Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.transcription_model), help="Used for audio files.")
            state.chat_model = st.selectbox("Chat Model", list(AVAILABLE_MODELS.keys()), index=list(AVAILABLE_MODELS.keys()).index(state.chat_model), help="Used for chatting with the final output.")

            st.divider()
            st.caption("Browser notifications for processing completion.")
            if st.button("Enable Notifications", key="enable_notif_btn", use_container_width=True):
                components.html(
                    """
                    <script>
                    if ("Notification" in window) {
                        Notification.requestPermission().then(function(permission) {
                            if (permission === "granted") {
                                new Notification("Notifications Enabled", {
                                    body: "You'll be notified when processing completes.",
                                    icon: "https://placehold.co/64x64?text=SN"
                                });
                            }
                        });
                    }
                    </script>
                    """,
                    height=0,
                )
    with col_participants:
        state.speakers = st.text_input("Participants (Optional)", value=state.speakers, placeholder="e.g., John Smith (Analyst), Jane Doe (CEO)")

    # --- Generate ---
    st.divider()
    validation_error = validate_inputs(state)

    if validation_error:
        st.warning(validation_error)

    if st.button("Generate Notes", type="primary", use_container_width=True, disabled=bool(validation_error)):
        state.processing = True; state.error_message = None; state.fallback_content = None; st.rerun()

    # --- Prompt Preview (collapsed) ---
    with st.expander("Prompt Preview", expanded=False):
        prompt_preview = get_dynamic_prompt(state, "[...transcript content...]")
        st.code(prompt_preview, language="markdown", height=200)

    # --- Processing ---
    if state.processing:
        with st.status("Processing your request...", expanded=True) as status:
            progress = ProgressTracker(status)
            try:
                final_note = process_and_save_task(state, status, progress)
                state.active_note_id = final_note['id']
                progress.finish()
                processing_time = final_note.get('processing_time', 0)
                word_count = len(final_note.get('content', '').split())
                status.update(
                    label=f"Done! {word_count:,} words generated in {processing_time:.1f}s. Switch to the **Output & History** tab to view your note.",
                    state="complete"
                )
                st.toast("Notes generated successfully!", icon="\u2705")
                # Send browser notification
                send_browser_notification(
                    "SynthNotes AI - Complete",
                    f"Your notes are ready! Processing took {processing_time:.1f}s"
                )
            except Exception as e:
                state.error_message = f"An error occurred during processing:\n{e}"
                status.update(label=f"Error: {e}", state="error")
                # Send error notification
                send_browser_notification(
                    "SynthNotes AI - Error",
                    "Processing failed. Check the app for details."
                )
        state.processing = False

    if state.error_message:
        st.error("Processing failed. See details below.")
        with st.expander("Error Details", expanded=True):
            st.code(state.error_message)
        err_col1, err_col2 = st.columns(2)
        if state.fallback_content:
            err_col1.download_button("Download Unsaved Note (.txt)", state.fallback_content, "synthnotes_fallback.txt", use_container_width=True)
        if err_col2.button("Dismiss Error", use_container_width=True):
            state.error_message = None
            state.fallback_content = None
            st.rerun()

@st.dialog("Delete Note")
def _confirm_delete_dialog(note_id: str, note_name: str):
    # Truncate long names in the dialog
    display_name = note_name if len(note_name) <= 50 else note_name[:47] + "..."
    st.markdown(f"Are you sure you want to delete **{display_name}**?")
    st.caption("This action cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True):
        st.rerun()
    if c2.button("Yes, Delete", use_container_width=True):
        database.delete_note(note_id)
        if st.session_state.app_state.active_note_id == note_id:
            st.session_state.app_state.active_note_id = None
        st.toast(f"Note '{display_name}' deleted.")
        st.rerun()

def run_validation_in_chunks(notes: str, transcript: str, model_name: str) -> list:
    """Run per-Q&A HTML-annotated validation.

    Always passes the FULL NOTES for context to both chunks so neither pass
    incorrectly flags content as missing that is actually captured in the other
    chunk. Splits only the PORTION TO ANNOTATE at Q&A boundaries.

    Returns a list of 1 or 2 annotated HTML strings.
    """
    model = genai.GenerativeModel(model_name)
    tx_limit = 40000  # characters of transcript per call

    # Find bold question lines — lines that start AND end with ** (markdown bold)
    note_lines = notes.split('\n')
    bold_indices = [
        i for i, line in enumerate(note_lines)
        if line.strip().startswith('**') and line.strip().endswith('**') and len(line.strip()) > 4
    ]

    # Single-pass when notes are short or have too few Q&As to justify splitting
    if len(bold_indices) < 4 or len(notes) < 8000:
        prompt = VALIDATION_DETAILED_PROMPT.format(
            chunk_info="Full Notes",
            full_notes=notes,
            chunk_to_annotate=notes,
            transcript=transcript[:tx_limit]
        )
        r = generate_with_retry(model, prompt)
        return [r.text]

    # Two-pass: split the PORTION TO ANNOTATE at the Q&A midpoint.
    # Both passes receive the FULL NOTES for context — this prevents Part 1
    # from flagging content as missing that is captured in Part 2 and vice versa.
    split_q = len(bold_indices) // 2
    split_line = bold_indices[split_q]
    chunk1_notes = '\n'.join(note_lines[:split_line]).strip()
    chunk2_notes = '\n'.join(note_lines[split_line:]).strip()

    # Transcript: transcript is sequential, so the first half maps to Part 1
    # Q&As and the second half maps to Part 2. Send the full slice to both so
    # neither is starved of context for edge cases.
    tx_slice = transcript[:tx_limit]

    prompt1 = VALIDATION_DETAILED_PROMPT.format(
        chunk_info="Part 1 of 2 — first half of Q&As",
        full_notes=notes,
        chunk_to_annotate=chunk1_notes,
        transcript=tx_slice
    )
    r1 = generate_with_retry(model, prompt1)

    prompt2 = VALIDATION_DETAILED_PROMPT.format(
        chunk_info="Part 2 of 2 — second half of Q&As",
        full_notes=notes,
        chunk_to_annotate=chunk2_notes,
        transcript=tx_slice
    )
    r2 = generate_with_retry(model, prompt2)

    return [r1.text, r2.text]


def render_output_and_history_tab(state: AppState):
    notes = database.get_all_notes()

    if not notes:
        st.markdown("""
### No notes yet

1. Switch to the **Input & Generate** tab
2. Paste text, upload a file (PDF, TXT), or record audio
3. Pick a meeting type and click **Generate Notes**

Your generated notes, transcripts, and chat history will appear here.
        """)
        return

    # --- Active Note ---
    if not state.active_note_id or not any(n['id'] == state.active_note_id for n in notes):
        state.active_note_id = notes[0]['id']

    active_note = database.get_note_by_id(state.active_note_id)
    if not active_note:
        active_note = database.get_note_by_id(notes[0]['id'])

    # --- Note header with inline metadata ---
    hdr_left, hdr_right = st.columns([3, 2])
    with hdr_left:
        # Truncate very long file names to prevent layout breakage
        display_name = active_note['file_name']
        if len(display_name) > 80:
            display_name = display_name[:77] + "..."
        st.markdown(f"### {display_name}")
        st.badge(active_note['meeting_type'])
    with hdr_right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Time", f"{active_note.get('processing_time', 0):.1f}s")
        m2.metric("Tokens", f"{active_note.get('token_usage', 0):,}")
        m3.metric("Date", datetime.fromisoformat(active_note['created_at']).strftime('%b %d'))

    # --- Side-by-side Notes & Transcript ---
    final_transcript = active_note.get('refined_transcript') or active_note.get('raw_transcript')
    transcript_source = "Refined" if active_note.get('refined_transcript') else "Raw"

    col_notes, col_transcript = st.columns([3, 2])
    with col_notes:
        view_mode = st.pills("View", ["Editor", "Preview"], default="Editor", key=f"view_mode_{active_note['id']}")
        if view_mode == "Editor":
            edited_content = st.text_area("Notes", value=active_note['content'], height=600, key=f"output_editor_{active_note['id']}")
            # Word count feedback for the notes editor
            note_wc = len(edited_content.split()) if edited_content else 0
            st.caption(f"{note_wc:,} words")
        else:
            edited_content = active_note['content']
            with st.container(height=600, border=True):
                st.markdown(edited_content)
            note_wc = len(edited_content.split()) if edited_content else 0
            st.caption(f"{note_wc:,} words")
    with col_transcript:
        st.markdown(f"**{transcript_source} Transcript**")
        if final_transcript:
            st.text_area("", value=final_transcript, height=600, disabled=True, label_visibility="collapsed", key=f"side_tx_{active_note['id']}")
        else:
            st.info("No transcript available.")

    # --- Actions bar ---
    note_id = active_note['id']
    fname = active_note.get('file_name', 'note')
    raw_tx = active_note.get('raw_transcript')

    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        copy_to_clipboard_button(edited_content)
    dl2.download_button(
        label="Notes (.txt)",
        data=edited_content,
        file_name=f"SynthNote_{fname}.txt",
        mime="text/plain",
        use_container_width=True
    )
    dl3.download_button(
        label="Notes (.md)",
        data=edited_content,
        file_name=f"SynthNote_{fname}.md",
        mime="text/markdown",
        use_container_width=True
    )
    if final_transcript:
        dl4.download_button(
            label=f"{transcript_source} Transcript",
            data=final_transcript,
            file_name=f"{transcript_source}_Transcript_{fname}.txt",
            mime="text/plain",
            use_container_width=True
        )
    elif raw_tx:
        dl4.download_button(
            label="Raw Transcript",
            data=raw_tx,
            file_name=f"Raw_Transcript_{fname}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        dl4.empty()

    st.feedback("thumbs", key=f"fb_{active_note['id']}")

    # --- VALIDATE OUTPUT COMPLETENESS ---
    if final_transcript:
        st.divider()
        st.subheader("Validate Output Completeness")
        st.caption(
            "Performs a detailed per-Q&A audit: each question and every answer bullet is checked "
            "against the source transcript. Long notes are automatically split into two parts. "
            "Results are displayed inline with colour-coded annotations."
        )

        val_key = f"validation_result_{active_note['id']}"

        if st.button("Validate Output Completeness", key=f"validate_btn_{active_note['id']}", type="secondary", use_container_width=False):
            with st.spinner("Running detailed per-Q&A validation — this may take a moment..."):
                try:
                    val_model_name = AVAILABLE_MODELS.get(state.chat_model, "gemini-2.5-pro")
                    chunks = run_validation_in_chunks(edited_content, final_transcript, val_model_name)
                    st.session_state[val_key] = chunks
                except Exception as e:
                    st.session_state[val_key] = [f"**Validation failed:** {str(e)}"]

        if val_key in st.session_state:
            chunks = st.session_state[val_key]
            # Legend
            st.markdown(
                "<div style='font-size:0.82em;margin:4px 0 8px 0;line-height:2'>"
                "<span style='background:#fef9c3;color:#78350f;border-radius:3px;"
                "padding:2px 6px;margin-right:8px'>⚠️ yellow</span>"
                "Missing content &nbsp;|&nbsp; "
                "<span style='color:#dc2626;text-decoration:line-through;margin-right:2px'>"
                "strikethrough</span>"
                "<span style='color:#16a34a;margin-right:8px'> → green</span>"
                "Misrepresentation &nbsp;|&nbsp; "
                "<span style='background:#ede9fe;color:#5b21b6;border-radius:3px;"
                "padding:2px 6px;margin-right:8px'>🔁 purple</span>"
                "Repeated / duplicate Q&A"
                "</div>",
                unsafe_allow_html=True
            )
            if len(chunks) == 2:
                tab1, tab2 = st.tabs(["Part 1", "Part 2"])
                for tab, chunk_html in zip([tab1, tab2], chunks):
                    with tab:
                        with st.container(height=620, border=True):
                            st.markdown(chunk_html, unsafe_allow_html=True)
            else:
                with st.container(height=620, border=True):
                    st.markdown(chunks[0], unsafe_allow_html=True)

    # --- CHAT ---
    st.divider()
    st.subheader("Chat with this Note")
    st.caption("Ask questions about the content. The model has access to both the notes and the source transcript for verbatim lookups.")

    st.session_state.chat_histories.setdefault(active_note['id'], [])

    for message in st.session_state.chat_histories[active_note['id']]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the note content..."):
        st.session_state.chat_histories[active_note['id']].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/progress_activity:"):
            full_response = ""
            try:
                transcript_context = final_transcript[:30000] if final_transcript else "Not available."
                truncation_note = ""
                if final_transcript and len(final_transcript) > 30000:
                    truncation_note = f"\n\nNote: The transcript was truncated from {len(final_transcript):,} to 30,000 characters. Some content at the end may be missing from the TRANSCRIPT section. The NOTES section contains the full meeting content."
                system_prompt = f"""You are an expert analyst. Your task is to answer questions based on the provided meeting notes and source transcript.
If the user asks for verbatim quotes or exact wording, refer to the TRANSCRIPT section. For analysis and summary questions, use the NOTES section.{truncation_note}

MEETING NOTES:
---
{edited_content}
---
SOURCE TRANSCRIPT:
---
{transcript_context}
---
"""
                chat_model_name = AVAILABLE_MODELS.get(state.chat_model, "gemini-1.5-flash")
                chat_model = genai.GenerativeModel(chat_model_name, system_instruction=system_prompt)
                messages_for_api = [{'role': "model" if m["role"] == "assistant" else "user", 'parts': [m['content']]} for m in st.session_state.chat_histories[active_note['id']]]

                chat = chat_model.start_chat(history=messages_for_api[:-1])
                response = chat.send_message(messages_for_api[-1]['parts'], stream=True)

                message_placeholder = st.empty()
                try:
                    for chunk in response:
                        if not chunk.parts:
                            continue
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "\u258c")
                except Exception as stream_err:
                    # Handle streaming interruption gracefully
                    if full_response:
                        full_response += f"\n\n*(Stream interrupted: {stream_err})*"
                    else:
                        raise stream_err
                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Sorry, an error occurred: {str(e)}"
                st.error(full_response)
                if st.session_state.chat_histories[active_note['id']]:
                    st.session_state.chat_histories[active_note['id']].pop()

        if 'full_response' in locals() and not full_response.startswith("Sorry"):
            st.session_state.chat_histories[active_note['id']].append({"role": "assistant", "content": full_response})

    # --- Analytics ---
    st.divider()
    st.subheader("History")

    raw_summary_data = database.get_analytics_summary()
    summary_dict = {}
    if isinstance(raw_summary_data, dict):
        summary_dict = raw_summary_data
    elif isinstance(raw_summary_data, tuple) and raw_summary_data:
        if isinstance(raw_summary_data[0], dict):
            summary_dict = raw_summary_data[0]
        else:
            summary_dict['total_notes'] = raw_summary_data[0] if len(raw_summary_data) > 0 else 0
            summary_dict['avg_time'] = raw_summary_data[1] if len(raw_summary_data) > 1 else 0.0
            summary_dict['total_tokens'] = raw_summary_data[2] if len(raw_summary_data) > 2 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Notes", summary_dict.get('total_notes', 0))
    c2.metric("Avg. Time / Note", f"{summary_dict.get('avg_time', 0.0):.1f}s")
    c3.metric("Total Tokens", f"{summary_dict.get('total_tokens', 0):,}")

    # --- Search & Filter ---
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        search_query = st.text_input("Search notes by file name", placeholder="Search notes...", label_visibility="collapsed")
    with filter_col2:
        type_filter = st.selectbox("Filter by meeting type", ["All Types"] + MEETING_TYPES, label_visibility="collapsed")

    filtered_notes = notes
    if search_query:
        filtered_notes = [n for n in filtered_notes if search_query.lower() in n.get('file_name', '').lower()]
    if type_filter != "All Types":
        filtered_notes = [n for n in filtered_notes if n.get('meeting_type') == type_filter]

    if not filtered_notes:
        st.info("No notes match your search. Try a different keyword or clear the filter.")

    for note in filtered_notes:
        is_active = note['id'] == state.active_note_id
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                # Truncate long filenames for card display
                card_name = note['file_name']
                if len(card_name) > 60:
                    card_name = card_name[:57] + "..."
                label = f"**{card_name}**"
                if is_active:
                    label += " &nbsp; `viewing`"
                st.markdown(label)
                content_text = note.get('content', '')
                if content_text:
                    snippet = content_text[:150].replace('\n', ' ').strip()
                    if len(content_text) > 150:
                        snippet += "..."
                    st.caption(snippet)
                # Badge + date on one line
                badge_col, date_col = st.columns([1, 2])
                badge_col.badge(note['meeting_type'])
                date_col.caption(datetime.fromisoformat(note['created_at']).strftime('%b %d, %Y %H:%M'))
            with col2:
                if st.button("View", key=f"view_{note['id']}", use_container_width=True, disabled=is_active):
                    state.active_note_id = note['id']
                    st.rerun()
                if st.button("Delete", key=f"del_{note['id']}", use_container_width=True, type="tertiary"):
                    _confirm_delete_dialog(note['id'], note['file_name'])

def _build_ia_prompt_template(meeting_type: str) -> str:
    """Return the IA prompt for the given meeting type; {transcript} left as placeholder."""
    return IA_MANAGEMENT_KTA_PROMPT if meeting_type == "management" else IA_EXPERT_KTA_PROMPT


def render_ia_processing(state: AppState):
    """Investment Analyst Processing: two-step transcript → dual output (KTAs + Rough Notes)."""

    # --- Session state init ---
    for key, default in [
        ("ia_meeting_type", "management"),
        ("ia_transcript", ""),
        ("ia_output", ""),
        ("ia_prompt_text", ""),
        ("ia_prompt_seed", ("", "")),
        ("ia_company_name", ""),
        ("ia_area", ""),
        ("ia_refine_enabled", False),
        ("ia_refined_transcript", ""),
        ("ia_tone", "Neutral"),
        ("ia_number_focus", "Moderate"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Step 1: Meeting type ---
    st.markdown("#### Step 1 — Meeting Type")
    meeting_type_opt = st.radio(
        "Meeting type",
        options=["1 — Company Management Meeting", "2 — Expert / Industry Expert / Channel Check Meeting"],
        index=0 if st.session_state.ia_meeting_type != "expert" else 1,
        label_visibility="collapsed",
        key="ia_meeting_type_radio",
    )
    st.session_state.ia_meeting_type = "management" if meeting_type_opt.startswith("1") else "expert"

    # Company / area fields
    _c1, _c2 = st.columns(2)
    with _c1:
        st.session_state.ia_company_name = st.text_input(
            "Company / Entity Name",
            value=st.session_state.ia_company_name,
            placeholder="e.g. Reliance, Hero MotoCorp",
            key="ia_company_name_input",
        )
    with _c2:
        if st.session_state.ia_meeting_type == "expert":
            st.session_state.ia_area = st.text_input(
                "Coverage Area / Sector",
                value=st.session_state.ia_area,
                placeholder="e.g. Two-Wheeler Dealerships, Quick Commerce",
                key="ia_area_input",
            )

    # --- Auto-reset prompt when meeting type or prompt version changes ---
    _IA_PROMPT_VERSION = "v7"  # bump when prompts are updated to force rebuild in existing sessions
    current_seed = (_IA_PROMPT_VERSION, st.session_state.ia_meeting_type)
    if st.session_state.ia_prompt_seed != current_seed or not st.session_state.ia_prompt_text:
        st.session_state.ia_prompt_text = _build_ia_prompt_template(
            st.session_state.ia_meeting_type
        )
        st.session_state.ia_prompt_seed = current_seed

    # --- Editable prompt ---
    with st.expander("Edit Prompt (optional)", expanded=False):
        reset_col, note_col = st.columns([1, 3])
        with reset_col:
            if st.button("Reset to Default", key="ia_reset_prompt", use_container_width=True):
                st.session_state.ia_prompt_text = _build_ia_prompt_template(
                    st.session_state.ia_meeting_type
                )
                st.rerun()
        with note_col:
            st.caption("`{transcript}` in the prompt will be replaced with your transcript at generation time.")
        st.session_state.ia_prompt_text = st.text_area(
            "Prompt template",
            value=st.session_state.ia_prompt_text,
            height=520,
            label_visibility="collapsed",
            key="ia_prompt_editor",
        )

    st.divider()

    # --- Step 2: Transcript input ---
    st.markdown("#### Step 2 — Paste the Transcript")
    st.session_state.ia_transcript = st.text_area(
        "Transcript",
        value=st.session_state.ia_transcript,
        height=320,
        placeholder="Paste the full meeting transcript here…",
        label_visibility="collapsed",
        key="ia_transcript_input",
    )

    if not st.session_state.ia_transcript.strip():
        st.info("Paste the transcript above to continue.")
        return

    wc = len(st.session_state.ia_transcript.split())
    st.caption(f"{wc:,} words")

    st.divider()

    # --- Tone and Data Emphasis ---
    ia_tone_col, ia_number_col = st.columns(2)
    with ia_tone_col:
        ia_tone = st.pills("Tone", TONE_OPTIONS, default=st.session_state.ia_tone, key="ia_tone_pills")
        if ia_tone:
            st.session_state.ia_tone = ia_tone
    with ia_number_col:
        ia_number_focus = st.pills("Data Emphasis", NUMBER_FOCUS_OPTIONS, default=st.session_state.ia_number_focus, key="ia_number_pills")
        if ia_number_focus:
            st.session_state.ia_number_focus = ia_number_focus

    st.divider()

    # --- Refinement toggle + model selector ---
    refine_col, model_col = st.columns([1, 1])
    with refine_col:
        ia_enable_refine = st.toggle(
            "Refine transcript before generating",
            value=st.session_state.ia_refine_enabled,
            key="ia_refine_toggle",
            help="Chunks the transcript and extracts structured Q&A from each chunk before generating. Improves output quality for long or messy transcripts.",
        )
    st.session_state.ia_refine_enabled = ia_enable_refine
    with model_col:
        _ia_model_keys = list(AVAILABLE_MODELS.keys())
        _ia_model_default = state.notes_model if state.notes_model in AVAILABLE_MODELS else _ia_model_keys[0]
        ia_model_name = st.selectbox(
            "Model",
            _ia_model_keys,
            index=_ia_model_keys.index(_ia_model_default),
            key="ia_model_select",
        )

    # --- Generate ---
    if st.button("Generate Investment Analysis", type="primary", use_container_width=True, key="ia_generate_btn"):
        try:
            model = _get_cached_model(ia_model_name)
            transcript_for_generation = st.session_state.ia_transcript
            st.session_state.ia_refined_transcript = ""

            # --- Optional refinement: chunk and extract Q&A ---
            if ia_enable_refine:
                raw_words = st.session_state.ia_transcript.split()
                chunks = create_chunks_with_overlap(st.session_state.ia_transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP) if len(raw_words) > CHUNK_WORD_SIZE else [st.session_state.ia_transcript]
                total_chunks = len(chunks)
                refined_parts = [None] * total_chunks

                with st.spinner(f"Refining transcript ({total_chunks} chunk{'s' if total_chunks > 1 else ''})..."):
                    def _refine_ia_chunk(idx, chunk):
                        prompt = IA_REFINE_CHUNK_PROMPT.format(
                            chunk_num=idx + 1,
                            total_chunks=total_chunks,
                            chunk=chunk,
                        )
                        resp = generate_with_retry(model, prompt)
                        return idx, resp.text

                    with ThreadPoolExecutor(max_workers=min(3, total_chunks)) as executor:
                        futures = {executor.submit(_refine_ia_chunk, i, c): i for i, c in enumerate(chunks)}
                        for future in as_completed(futures):
                            idx, text = future.result()
                            refined_parts[idx] = text

                transcript_for_generation = "\n\n---\n\n".join(p for p in refined_parts if p)
                st.session_state.ia_refined_transcript = transcript_for_generation

            with st.spinner("Generating key takeaways and rough notes…"):
                prompt_template = st.session_state.ia_prompt_text
                if "{transcript}" in prompt_template:
                    prompt = prompt_template.format(transcript=transcript_for_generation)
                else:
                    prompt = prompt_template + "\n\n---\nTRANSCRIPT:\n" + transcript_for_generation

                # Inject tone and data emphasis as additional instructions
                _ia_tone = st.session_state.get("ia_tone", "Neutral") or "Neutral"
                _ia_number_focus = st.session_state.get("ia_number_focus", "Moderate") or "Moderate"
                _ia_tone_descriptions = {
                    "As Is": "Present findings exactly as stated in the transcript. Do not add any positive or negative framing — reproduce the sentiment already present in the source material.",
                    "Very Positive": "Frame KEY TAKEAWAYS constructively — strengths, growth, advantages. Challenges are temporary or manageable.",
                    "Positive": "Frame KEY TAKEAWAYS positively. Risks acknowledged but opportunities emphasized.",
                    "Neutral": "Present KEY TAKEAWAYS objectively and balanced.",
                    "Negative": "Emphasize risks and structural problems in KEY TAKEAWAYS. Positive developments are insufficient.",
                    "Very Negative": "Frame KEY TAKEAWAYS around fundamental weaknesses and unsustainable practices. Deeply problematic framing.",
                }
                _addendum = []
                if _ia_tone != "Neutral":
                    _tone_desc = _ia_tone_descriptions.get(_ia_tone, "")
                    if _tone_desc:
                        _addendum.append(f"TONE FOR KEY TAKEAWAYS: {_tone_desc}")
                _number_instruction = NUMBER_FOCUS_INSTRUCTIONS.get(_ia_number_focus, "")
                if _ia_number_focus != "Moderate" and _number_instruction:
                    _addendum.append(f"DATA EMPHASIS FOR ROUGH NOTES: {_number_instruction}")
                if _addendum:
                    prompt += "\n\n---\nADDITIONAL FORMATTING INSTRUCTIONS:\n" + "\n".join(_addendum)

                response = generate_with_retry(model, prompt)
                st.session_state.ia_output = response.text
                st.rerun()
        except Exception as e:
            st.error(f"Failed to generate analysis: {e}")

    # --- Display output ---
    if st.session_state.ia_output:
        st.divider()
        if st.session_state.ia_refined_transcript:
            with st.expander("View refined Q&A transcript (intermediate step)", expanded=False):
                st.markdown(st.session_state.ia_refined_transcript)

        raw = st.session_state.ia_output

        # Split into KTA and Rough Notes sections
        kta_text = raw
        rough_text = ""
        raw_upper = raw.upper()
        # Try known markers first (ordered from most specific to least)
        for marker in (
            "OUTPUT 2: ROUGH NOTES",
            "OUTPUT 2:",
            "ROUGH NOTES",
            "MEETING NOTES",
            "RAW NOTES",
            "DETAILED NOTES",
        ):
            idx = raw_upper.find(marker)
            if idx != -1:
                kta_text = raw[:idx].strip()
                rough_text = raw[idx:].strip()
                break

        # Fallback: if no named marker found, try splitting on a "---" divider
        # that appears after meaningful KTA content (skip the first --- in prompts)
        if not rough_text:
            divider = "---"
            search_start = 0
            while True:
                div_idx = raw.find(divider, search_start)
                if div_idx == -1:
                    break
                candidate_kta = raw[:div_idx].strip()
                candidate_rough = raw[div_idx + len(divider):].strip()
                # Accept split only if both halves have substantial content
                if len(candidate_kta) > 50 and len(candidate_rough) > 50:
                    kta_text = candidate_kta
                    rough_text = candidate_rough
                    break
                search_start = div_idx + len(divider)

        # Build dynamic headings
        _co = st.session_state.ia_company_name.strip()
        _ar = st.session_state.ia_area.strip()
        if st.session_state.ia_meeting_type == "management":
            _kta_heading = f"KTAs — Management of {_co}" if _co else "Key Investment Takeaways — Management Meeting"
            _rough_heading = f"Meeting Notes — {_co} Management" if _co else "Rough Notes — Management Meeting"
        else:
            _kta_heading = (
                f"KTAs — Expert on {_co}" + (f" | {_ar}" if _ar else "")
                if _co else
                "Key Investment Takeaways — Expert Meeting"
            )
            _rough_heading = (
                f"Meeting Notes — Expert on {_co}" + (f" ({_ar})" if _ar else "")
                if _co else
                "Rough Notes — Expert Meeting"
            )

        col_kta, col_rough = st.columns(2, gap="large")

        with col_kta:
            st.markdown(f"### {_kta_heading}")
            with st.container(border=True):
                st.markdown(kta_text)
            copy_to_clipboard_button(kta_text, "Copy KTAs")
            st.download_button(
                "Download KTAs (.txt)",
                data=kta_text,
                file_name="Key_Investment_Takeaways.txt",
                mime="text/plain",
                use_container_width=True,
                key="ia_dl_kta",
            )

        with col_rough:
            st.markdown(f"### {_rough_heading}")
            with st.container(border=True):
                st.markdown(rough_text if rough_text else raw)
            copy_to_clipboard_button(rough_text if rough_text else raw, "Copy Rough Notes")
            st.download_button(
                "Download Rough Notes (.txt)",
                data=rough_text if rough_text else raw,
                file_name="Rough_Notes.txt",
                mime="text/plain",
                use_container_width=True,
                key="ia_dl_rough",
            )

        st.divider()
        copy_to_clipboard_button(raw, "Copy Full Output")
        st.download_button(
            "Download Full Output (.txt)",
            data=raw,
            file_name="Investment_Analysis_Full.txt",
            mime="text/plain",
            use_container_width=True,
            key="ia_dl_full",
        )


def render_otg_notes_tab(state: AppState):
    st.subheader("OTG Notes")

    # --- Top-level mode selector ---
    otg_mode = st.pills(
        "Mode",
        ["Research Style", "Investment Analyst"],
        default="Research Style",
        key="otg_mode_pills",
    )

    st.divider()

    if otg_mode == "Investment Analyst":
        render_ia_processing(state)
        return

    st.caption("Paste detailed meeting notes to convert them into concise, narrative-style research notes. Select entities, topics, tone, and data emphasis to control the output.")

    # --- OTG State init ---
    if "otg_input" not in st.session_state:
        st.session_state.otg_input = ""
    if "otg_extracted" not in st.session_state:
        st.session_state.otg_extracted = None
    if "otg_output" not in st.session_state:
        st.session_state.otg_output = ""
    if "otg_selected_topics" not in st.session_state:
        st.session_state.otg_selected_topics = []
    if "otg_selected_entities" not in st.session_state:
        st.session_state.otg_selected_entities = []
    if "otg_refine_enabled" not in st.session_state:
        st.session_state.otg_refine_enabled = False
    if "otg_refined_notes" not in st.session_state:
        st.session_state.otg_refined_notes = ""

    # --- Input: paste notes or load from existing ---
    input_source = st.pills("Source", ["Paste Notes", "From Saved Note"], default="Paste Notes", key="otg_source_pills")

    if input_source == "Paste Notes":
        st.session_state.otg_input = st.text_area(
            "Paste your detailed notes here:",
            value=st.session_state.otg_input,
            height=300,
            key="otg_paste_input"
        )
    else:
        notes = database.get_all_notes()
        if not notes:
            st.info("No saved notes. Generate notes first in the Input & Generate tab.")
            return
        note_labels = []
        note_id_by_label = {}
        for n in notes:
            label = n['file_name']
            # Disambiguate duplicate filenames by appending the date
            if label in note_id_by_label:
                created = datetime.fromisoformat(n['created_at']).strftime('%b %d %H:%M')
                label = f"{label} ({created})"
            note_labels.append(label)
            note_id_by_label[label] = n['id']
        selected_name = st.selectbox("Select a saved note", note_labels, key="otg_note_selector")
        if selected_name:
            selected_note = database.get_note_by_id(note_id_by_label[selected_name])
            if selected_note:
                st.session_state.otg_input = selected_note.get('content', '')
                with st.expander("Preview loaded notes", expanded=False):
                    st.markdown(st.session_state.otg_input[:2000] + ("..." if len(st.session_state.otg_input) > 2000 else ""))

    if not st.session_state.otg_input.strip():
        st.info("Paste notes above or load a saved note to get started.")
        return

    # Word count for OTG input
    otg_wc = len(st.session_state.otg_input.split())
    st.caption(f"{otg_wc:,} words in source notes")

    # --- Step 1: Extract entities, sector, topics ---
    st.divider()

    if st.button("Analyze Notes", use_container_width=True, key="otg_analyze_btn"):
        with st.spinner("Extracting entities, sector, and topics..."):
            try:
                extract_model = _get_cached_model(state.notes_model)
                prompt = OTG_EXTRACT_PROMPT.format(notes=st.session_state.otg_input)
                response = generate_with_retry(extract_model, prompt)
                raw_json = response.text.strip()
                # Strip markdown code fences if present (handle ```json and ``` variants)
                if raw_json.startswith("```"):
                    lines = raw_json.split("\n")
                    # Remove first line (```json or ```)
                    lines = lines[1:]
                    # Remove last line if it's just ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_json = "\n".join(lines).strip()

                # Try to find JSON object if there's extra text
                json_match = re.search(r'\{[\s\S]*\}', raw_json)
                if json_match:
                    raw_json = json_match.group(0)

                extracted = json.loads(raw_json)

                # Validate and normalize the extracted data
                if not isinstance(extracted, dict):
                    extracted = {}
                extracted.setdefault("entities", [])
                extracted.setdefault("people", [])
                extracted.setdefault("sector", "Unknown")
                extracted.setdefault("topics", [])

                # Ensure lists contain strings only
                extracted["entities"] = [str(e) for e in extracted["entities"] if e]
                extracted["people"] = [str(p) for p in extracted["people"] if p]
                extracted["topics"] = [str(t) for t in extracted["topics"] if t]

                st.session_state.otg_extracted = extracted
                st.session_state.otg_selected_topics = extracted.get("topics", [])
                st.session_state.otg_selected_entities = extracted.get("entities", [])
                st.session_state.otg_output = ""
                st.session_state.otg_refined_notes = ""
                st.rerun()
            except json.JSONDecodeError as je:
                st.error(f"Failed to parse analysis results. The model returned invalid JSON. Try again or use a different model.")
            except Exception as e:
                st.error(f"Failed to analyze notes: {e}")

    extracted = st.session_state.otg_extracted
    if not extracted:
        return

    # --- Display sector ---
    sector = extracted.get("sector", "Unknown")
    st.markdown(f"**Sector:** {sector}")

    st.divider()

    # --- Entity selection (use stable keys based on entity name hash) ---
    entities = extracted.get("entities", [])
    people = extracted.get("people", [])
    all_entity_names = list(dict.fromkeys(entities + people))  # Remove duplicates while preserving order
    if all_entity_names:
        st.markdown("**Select entities to focus on:**")
        selected_entities = st.multiselect(
            "Entities",
            options=all_entity_names,
            default=[e for e in st.session_state.otg_selected_entities if e in all_entity_names],
            key="otg_entity_multiselect",
            label_visibility="collapsed",
            accept_new_options=True,
            placeholder="Select or type to add new entities",
        )
        st.session_state.otg_selected_entities = selected_entities

    st.divider()

    # --- Topic selection ---
    topics = extracted.get("topics", [])
    if topics:
        st.markdown("**Select topics to focus on:**")
        selected_topics = st.multiselect(
            "Topics",
            options=topics,
            default=[t for t in st.session_state.otg_selected_topics if t in topics],
            key="otg_topic_multiselect",
            label_visibility="collapsed",
            accept_new_options=True,
            placeholder="Select or type to add new topics",
        )
        st.session_state.otg_selected_topics = selected_topics

    # --- Tone, Number Focus, and Length ---
    st.divider()
    tone_col, number_col = st.columns(2)
    with tone_col:
        tone = st.pills("Tone", TONE_OPTIONS, default="Neutral", key="otg_tone_pills")
    with number_col:
        number_focus = st.pills("Data Emphasis", NUMBER_FOCUS_OPTIONS, default="Moderate", key="otg_number_pills")

    word_count_options = list(OTG_WORD_COUNT_OPTIONS.keys())
    selected_word_count = st.select_slider(
        "Approximate Output Length",
        options=word_count_options,
        value=word_count_options[1],
        key="otg_word_count_slider",
    )

    # --- Custom instructions ---
    custom_instructions = st.text_area(
        "Additional Instructions (Optional)",
        placeholder="e.g., Emphasize competitive positioning vs Blinkit, keep the note under 200 words, mention the IPO timeline...",
        height=80,
        key="otg_custom_instructions"
    )

    # --- Generate OTG note ---
    st.divider()

    if not st.session_state.otg_selected_topics:
        st.warning("Select at least one topic to focus on.")
        return

    if not st.session_state.otg_selected_entities and all_entity_names:
        st.warning("Select at least one entity to focus on.")
        return

    # --- Refinement toggle ---
    refine_col, _ = st.columns([1, 2])
    with refine_col:
        enable_refine = st.toggle(
            "Refine notes before generating",
            value=st.session_state.otg_refine_enabled,
            key="otg_refine_toggle",
            help="Chunks the source notes and extracts structured Q&A from each chunk before generating the final note. Improves output quality for long or unstructured notes.",
        )
    st.session_state.otg_refine_enabled = enable_refine

    if st.button("Generate Research Note", type="primary", use_container_width=True, key="otg_generate_btn"):
        try:
            otg_model = _get_cached_model(state.notes_model)
            notes_for_generation = st.session_state.otg_input
            st.session_state.otg_refined_notes = ""

            # --- Optional refinement: chunk and extract Q&A ---
            if enable_refine:
                raw_words = st.session_state.otg_input.split()
                chunks = create_chunks_with_overlap(st.session_state.otg_input, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP) if len(raw_words) > CHUNK_WORD_SIZE else [st.session_state.otg_input]
                total_chunks = len(chunks)
                refined_parts = [None] * total_chunks

                with st.spinner(f"Refining notes ({total_chunks} chunk{'s' if total_chunks > 1 else ''})..."):
                    def _refine_otg_chunk(idx, chunk):
                        prompt = OTG_REFINE_CHUNK_PROMPT.format(
                            chunk_num=idx + 1,
                            total_chunks=total_chunks,
                            chunk=chunk,
                        )
                        resp = generate_with_retry(otg_model, prompt)
                        return idx, resp.text

                    with ThreadPoolExecutor(max_workers=min(3, total_chunks)) as executor:
                        futures = {executor.submit(_refine_otg_chunk, i, c): i for i, c in enumerate(chunks)}
                        for future in as_completed(futures):
                            idx, text = future.result()
                            refined_parts[idx] = text

                notes_for_generation = "\n\n---\n\n".join(p for p in refined_parts if p)
                st.session_state.otg_refined_notes = notes_for_generation

            with st.spinner("Generating research note..."):
                topics_str = ", ".join(st.session_state.otg_selected_topics)
                entities_str = ", ".join(st.session_state.otg_selected_entities) if st.session_state.otg_selected_entities else "all entities mentioned"
                number_instruction = NUMBER_FOCUS_INSTRUCTIONS.get(number_focus, NUMBER_FOCUS_INSTRUCTIONS["Moderate"])
                length_instruction = OTG_WORD_COUNT_OPTIONS.get(selected_word_count, OTG_WORD_COUNT_OPTIONS["Medium (~300 words)"])
                custom_block = f"9. ADDITIONAL INSTRUCTIONS FROM THE ANALYST: {custom_instructions}" if custom_instructions.strip() else ""
                prompt = OTG_CONVERT_PROMPT.format(
                    tone=tone,
                    topics=topics_str,
                    entities=entities_str,
                    number_focus_instruction=number_instruction,
                    length_instruction=length_instruction,
                    custom_instructions_block=custom_block,
                    notes=notes_for_generation,
                )
                response = generate_with_retry(otg_model, prompt)
                st.session_state.otg_output = response.text
                st.rerun()
        except Exception as e:
            st.error(f"Failed to generate research note: {e}")

    # --- Display output ---
    if st.session_state.otg_output:
        st.divider()
        if st.session_state.otg_refined_notes:
            with st.expander("View refined Q&A notes (intermediate step)", expanded=False):
                st.markdown(st.session_state.otg_refined_notes)
        st.markdown("### Generated Research Note")
        with st.container(border=True):
            st.markdown(st.session_state.otg_output)

        otg_sector_slug = sector.replace(' ', '_')
        out1, out2, out3 = st.columns(3)
        with out1:
            copy_to_clipboard_button(st.session_state.otg_output, "Copy Research Note")
        out2.download_button(
            label="Download (.txt)",
            data=st.session_state.otg_output,
            file_name=f"OTG_Note_{otg_sector_slug}.txt",
            mime="text/plain",
            use_container_width=True
        )
        out3.download_button(
            label="Download (.md)",
            data=st.session_state.otg_output,
            file_name=f"OTG_Note_{otg_sector_slug}.md",
            mime="text/markdown",
            use_container_width=True
        )

# --- EARNINGS CALL MULTI-FILE ANALYSIS ---

def _extract_pdf_texts(uploaded_files: list) -> List[Tuple[str, str]]:
    """Extract text from multiple uploaded PDF files.
    Returns list of (filename, text_content) tuples. Skips files with errors.
    """
    results = []
    for f in uploaded_files:
        name = f.name
        try:
            file_bytes_io = io.BytesIO(f.getvalue())
            reader = PyPDF2.PdfReader(file_bytes_io)
            if reader.is_encrypted:
                st.warning(f"Skipping encrypted PDF: {name}")
                continue
            content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            if not content or not content.strip():
                st.warning(f"No text found in: {name}")
                continue
            content = re.sub(r'\n{3,}', '\n\n', content.strip())
            results.append((name, content))
        except Exception as e:
            st.warning(f"Error reading {name}: {e}")
    return results


def _discover_topics(file_texts: List[Tuple[str, str]], model_name: str) -> dict:
    """Send first N transcripts to Gemini for topic discovery. Returns parsed JSON."""
    discovery_files = file_texts[:MAX_TOPIC_DISCOVERY_FILES]

    # Build combined transcript text with file labels
    transcript_parts = []
    for i, (fname, text) in enumerate(discovery_files, 1):
        # Limit each transcript to ~15k words to avoid context overflow
        words = text.split()
        truncated = " ".join(words[:15000])
        transcript_parts.append(f"--- TRANSCRIPT {i}: {fname} ---\n{truncated}")

    combined = "\n\n".join(transcript_parts)
    prompt = EC_TOPIC_DISCOVERY_PROMPT.format(transcripts=combined)

    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    raw_json = response.text.strip()

    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_json = "\n".join(lines).strip()

    json_match = re.search(r'\{[\s\S]*\}', raw_json)
    if json_match:
        raw_json = json_match.group(0)

    result = json.loads(raw_json)
    if not isinstance(result, dict):
        raise ValueError("Topic discovery returned invalid format.")

    result.setdefault("company_name", "Unknown Company")
    result.setdefault("primary_topics", [])
    result.setdefault("cross_cutting_topics", [])

    return result


def _build_topic_structure_text(selected_topics: dict) -> str:
    """Convert selected topics dict into a text structure for the notes prompt."""
    lines = []
    for primary in selected_topics.get("primary_topics", []):
        lines.append(f"**{primary['name']}**")
        if primary.get("description"):
            lines.append(f"  ({primary['description']})")
        for sub in primary.get("sub_topics", []):
            lines.append(f"  - {sub}")
        lines.append("")

    for cross in selected_topics.get("cross_cutting_topics", []):
        lines.append(f"**{cross['name']}**")
        if cross.get("description"):
            lines.append(f"  ({cross['description']})")
        lines.append("")

    return "\n".join(lines)


def _generate_notes_for_file(file_label: str, transcript: str, topic_structure_text: str,
                              model_name: str) -> Tuple[str, int]:
    """Generate earnings call notes for a single file under the given topic structure.
    Returns (notes_text, token_count).
    """
    prompt = EC_MULTI_FILE_NOTES_PROMPT.format(
        topic_structure=topic_structure_text,
        file_label=file_label,
        transcript=transcript
    )
    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, stream=True,
                                   generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    full_text, token_count = stream_and_collect(response)
    return full_text, token_count


def _stitch_multi_file_notes(company_name: str, file_notes: List[Tuple[str, str]]) -> str:
    """Stitch together notes from multiple files into a single output."""
    header = EC_MULTI_FILE_STITCH_HEADER.format(
        company_name=company_name,
        date=datetime.now().strftime("%B %d, %Y"),
        file_count=len(file_notes)
    )

    parts = [header]
    for i, (fname, notes) in enumerate(file_notes, 1):
        parts.append(f"## {i}. {fname}\n")
        parts.append(notes.strip())
        parts.append("\n\n---\n")

    return "\n".join(parts)


# --- REPORT COMPARISON HELPER FUNCTIONS ---

MAX_RC_DISCOVERY_FILES = 4  # Number of reports to scan for dimension discovery

def _discover_rc_dimensions(file_texts: List[Tuple[str, str]], model_name: str) -> dict:
    """Send reports to Gemini for qualitative dimension discovery. Returns parsed JSON."""
    discovery_files = file_texts[:MAX_RC_DISCOVERY_FILES]

    report_parts = []
    for i, (fname, text) in enumerate(discovery_files, 1):
        words = text.split()
        truncated = " ".join(words[:15000])
        report_parts.append(f"--- REPORT {i}: {fname} ---\n{truncated}")

    combined = "\n\n".join(report_parts)
    prompt = RC_DIMENSION_DISCOVERY_PROMPT.format(reports=combined)

    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    raw_json = response.text.strip()

    # Strip markdown code fences if present
    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_json = "\n".join(lines).strip()

    json_match = re.search(r'\{[\s\S]*\}', raw_json)
    if json_match:
        raw_json = json_match.group(0)

    result = json.loads(raw_json)
    if not isinstance(result, dict):
        raise ValueError("Dimension discovery returned invalid format.")

    result.setdefault("company_name", "Unknown Company")
    result.setdefault("report_years", [])
    result.setdefault("comparison_dimensions", [])

    return result


def _build_dimension_structure_text(selected_dimensions: dict) -> str:
    """Convert selected dimensions dict into a text structure for prompts."""
    lines = []
    for dim in selected_dimensions.get("comparison_dimensions", []):
        lines.append(f"**{dim['name']}**")
        if dim.get("description"):
            lines.append(f"  ({dim['description']})")
        for sub in dim.get("sub_dimensions", []):
            lines.append(f"  - {sub}")
        lines.append("")
    return "\n".join(lines)


def _extract_report_qualitative(file_label: str, report_text: str, dimension_structure_text: str,
                                 model_name: str) -> Tuple[str, int]:
    """Extract qualitative data from a single report for the given dimensions.
    Returns (extraction_text, token_count).
    """
    prompt = RC_PER_REPORT_EXTRACTION_PROMPT.format(
        dimension_structure=dimension_structure_text,
        file_label=file_label,
        report_text=report_text
    )
    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, stream=True,
                                   generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    full_text, token_count = stream_and_collect(response)
    return full_text, token_count


def _generate_rc_comparison(company_name: str, report_labels: str, dimension_structure_text: str,
                             per_report_extractions: str, model_name: str) -> Tuple[str, int]:
    """Generate the final year-over-year comparison from all per-report extractions.
    Returns (comparison_text, token_count).
    """
    prompt = RC_COMPARISON_PROMPT.format(
        company_name=company_name,
        report_labels=report_labels,
        dimension_structure=dimension_structure_text,
        per_report_extractions=per_report_extractions
    )
    model = _get_cached_model(model_name)
    response = generate_with_retry(model, prompt, stream=True,
                                   generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
    full_text, token_count = stream_and_collect(response)
    return full_text, token_count


def _stitch_rc_output(company_name: str, report_labels: str, comparison_text: str,
                       per_report_extractions: List[Tuple[str, str]]) -> str:
    """Stitch the comparison output with header and optional per-report appendix."""
    header = RC_STITCH_HEADER.format(
        company_name=company_name,
        date=datetime.now().strftime("%B %d, %Y"),
        report_labels=report_labels
    )
    parts = [header, comparison_text.strip()]

    # Add per-report extractions as appendix
    parts.append("\n\n---\n\n# Appendix: Per-Report Extractions\n")
    for i, (fname, extraction) in enumerate(per_report_extractions, 1):
        parts.append(f"## {i}. {fname}\n")
        parts.append(extraction.strip())
        parts.append("\n\n---\n")

    return "\n".join(parts)


# --- REPORT COMPARISON TAB ---

def render_report_comparison_tab(state: AppState):
    st.subheader("Annual Report Comparison")
    st.caption(
        "Upload annual reports (PDFs) from different years for the same company. "
        "The system will identify qualitative dimensions — management commentary, strategy, "
        "org structure, incentives, risk factors, etc. — and produce a year-over-year comparison. "
        "Numbers and financial data are intentionally de-emphasized; the focus is on narrative shifts."
    )

    # --- Session state initialization ---
    if "rc_files" not in st.session_state:
        st.session_state.rc_files = None
    if "rc_texts" not in st.session_state:
        st.session_state.rc_texts = []
    if "rc_discovered_dims" not in st.session_state:
        st.session_state.rc_discovered_dims = None
    if "rc_selected_dims" not in st.session_state:
        st.session_state.rc_selected_dims = None
    if "rc_comparison_output" not in st.session_state:
        st.session_state.rc_comparison_output = ""
    if "rc_processing" not in st.session_state:
        st.session_state.rc_processing = False
    if "rc_per_report_extractions" not in st.session_state:
        st.session_state.rc_per_report_extractions = []

    # --- Step 1: Upload PDFs ---
    st.markdown("### Step 1: Upload Annual Reports")
    uploaded_files = st.file_uploader(
        "Upload annual report PDFs (one per year)",
        type=["pdf"],
        accept_multiple_files=True,
        key="rc_pdf_uploader",
        help="Upload 2 or more annual report PDFs from different years for the same company."
    )

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("Upload at least 2 annual report PDFs to begin. Name files clearly (e.g., 'CompanyName_AR_2023.pdf').")
        if not uploaded_files:
            st.session_state.rc_texts = []
            st.session_state.rc_discovered_dims = None
            st.session_state.rc_selected_dims = None
            st.session_state.rc_comparison_output = ""
            st.session_state.rc_per_report_extractions = []
        return

    file_names = [f.name for f in uploaded_files]
    st.caption(f"**{len(uploaded_files)}** reports uploaded: {', '.join(file_names)}")

    # --- Step 2: Extract text and discover dimensions ---
    st.divider()
    st.markdown("### Step 2: Identify Comparison Dimensions")
    st.caption(
        f"Analyzes the first {min(len(uploaded_files), MAX_RC_DISCOVERY_FILES)} reports "
        "to identify qualitative dimensions for comparison (strategy, governance, incentives, etc.)."
    )

    analysis_model = st.selectbox(
        "Model for Dimension Discovery",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="rc_analysis_model_select"
    )

    if st.button("Identify Dimensions", use_container_width=True, key="rc_discover_btn"):
        with st.spinner("Extracting text from PDFs and identifying comparison dimensions..."):
            try:
                file_texts = _extract_pdf_texts(uploaded_files)
                if len(file_texts) < 2:
                    st.error("Could not extract text from at least 2 PDFs. Check file quality.")
                    return
                st.session_state.rc_texts = file_texts

                discovered = _discover_rc_dimensions(file_texts, analysis_model)
                st.session_state.rc_discovered_dims = discovered
                st.session_state.rc_selected_dims = copy.deepcopy(discovered)
                st.session_state.rc_comparison_output = ""
                st.session_state.rc_per_report_extractions = []
                st.rerun()
            except json.JSONDecodeError:
                st.error("Failed to parse dimension discovery results. Try again or use a different model.")
            except Exception as e:
                st.error(f"Dimension discovery failed: {e}")

    # --- Step 3: Display and select dimensions ---
    discovered = st.session_state.rc_discovered_dims
    if not discovered:
        return

    company_name = discovered.get("company_name", "Unknown Company")
    report_years = discovered.get("report_years", [])
    years_display = ", ".join(report_years) if report_years else "multiple years"
    st.success(f"Dimensions identified for **{company_name}** ({years_display})")

    st.divider()
    st.markdown("### Step 3: Select & Customize Dimensions")
    st.caption("Check the dimensions you want compared across years. You can also add custom dimensions.")

    comparison_dims = discovered.get("comparison_dimensions", [])

    selected_dims = []
    for d_idx, dim in enumerate(comparison_dims):
        d_name = dim.get("name", f"Dimension {d_idx+1}")
        d_desc = dim.get("description", "")
        sub_dims = dim.get("sub_dimensions", [])

        d_key = f"rc_dim_{d_idx}"
        d_enabled = st.checkbox(
            f"**{d_name}**" + (f" — {d_desc}" if d_desc else ""),
            value=True,
            key=d_key
        )

        if d_enabled:
            selected_subs = []
            sub_cols = st.columns(min(len(sub_dims), 4)) if sub_dims else []
            for s_idx, sub in enumerate(sub_dims):
                col = sub_cols[s_idx % len(sub_cols)] if sub_cols else st
                s_key = f"rc_sub_{d_idx}_{s_idx}"
                if col.checkbox(sub, value=True, key=s_key):
                    selected_subs.append(sub)

            # Add custom sub-dimension
            custom_sub = st.text_input(
                "Add custom sub-dimension",
                key=f"rc_custom_sub_{d_idx}",
                placeholder="e.g., succession planning, digital strategy..."
            )
            if custom_sub and custom_sub.strip():
                for cs in [s.strip() for s in custom_sub.split(",") if s.strip()]:
                    if cs not in selected_subs:
                        selected_subs.append(cs)

            if selected_subs:
                selected_dims.append({
                    "name": d_name,
                    "description": d_desc,
                    "sub_dimensions": selected_subs
                })

        st.divider()

    # Add custom dimension
    with st.expander("Add Custom Dimension", expanded=False):
        custom_d_name = st.text_input("Dimension Name", key="rc_custom_dim_name",
                                       placeholder="e.g., Regulatory Environment")
        custom_d_desc = st.text_input("Description (optional)", key="rc_custom_dim_desc")
        custom_d_subs = st.text_input("Sub-dimensions (comma-separated)", key="rc_custom_dim_subs",
                                       placeholder="e.g., compliance changes, new regulations, policy shifts")
        if custom_d_name and custom_d_name.strip():
            subs = [s.strip() for s in custom_d_subs.split(",") if s.strip()] if custom_d_subs else []
            selected_dims.append({
                "name": custom_d_name.strip(),
                "description": custom_d_desc.strip() if custom_d_desc else "",
                "sub_dimensions": subs
            })

    # Store final selection
    final_selection = {
        "company_name": company_name,
        "report_years": report_years,
        "comparison_dimensions": selected_dims
    }
    st.session_state.rc_selected_dims = final_selection

    # Preview selected structure
    with st.expander("Preview Selected Dimension Structure", expanded=False):
        structure_text = _build_dimension_structure_text(final_selection)
        st.code(structure_text, language="markdown")

    if not selected_dims:
        st.warning("Select at least one dimension to generate comparison.")
        return

    # --- Step 4: Generate comparison ---
    st.divider()
    st.markdown("### Step 4: Generate Comparison")

    file_texts = st.session_state.rc_texts
    if not file_texts:
        st.warning("Report texts not available. Please re-run dimension identification.")
        return

    st.caption(
        f"Will extract qualitative data from **{len(file_texts)}** reports, "
        "then produce a year-over-year comparison across selected dimensions."
    )

    notes_model = st.selectbox(
        "Model for Analysis",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="rc_notes_model_select"
    )

    if st.button("Generate Comparison", type="primary", use_container_width=True, key="rc_generate_btn"):
        st.session_state.rc_processing = True
        st.rerun()

    if st.session_state.rc_processing:
        dimension_structure_text = _build_dimension_structure_text(final_selection)
        total_tokens = 0
        all_extractions = []
        start_time = time.time()

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Phase 1: Extract qualitative data from each report
            num_files = len(file_texts)
            for i, (fname, text) in enumerate(file_texts):
                # Extraction is ~70% of the work, comparison is ~30%
                pct = (i / num_files) * 0.7
                progress_bar.progress(pct)
                status_text.markdown(
                    f"**{int(pct*100)}%** — Extracting qualitative data from **{fname}** ({i+1}/{num_files})"
                )

                extraction_text, tokens = _extract_report_qualitative(
                    fname, text, dimension_structure_text, notes_model
                )
                total_tokens += tokens
                all_extractions.append((fname, extraction_text))

            # Phase 2: Generate comparison
            progress_bar.progress(0.75)
            status_text.markdown("**75%** — Generating year-over-year comparison...")

            report_labels = ", ".join(fname for fname, _ in file_texts)
            per_report_combined = "\n\n---\n\n".join(
                f"### Report: {fname}\n\n{extraction}"
                for fname, extraction in all_extractions
            )

            comparison_text, comp_tokens = _generate_rc_comparison(
                company_name, report_labels, dimension_structure_text,
                per_report_combined, notes_model
            )
            total_tokens += comp_tokens

            # Stitch final output
            progress_bar.progress(0.95)
            status_text.markdown("**95%** — Assembling final output...")

            stitched = _stitch_rc_output(company_name, report_labels, comparison_text, all_extractions)

            st.session_state.rc_comparison_output = stitched
            st.session_state.rc_per_report_extractions = all_extractions

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.markdown(
                f"**100%** — Done! {len(file_texts)} reports compared, "
                f"{total_tokens:,} tokens used, {elapsed:.1f}s elapsed."
            )
            st.toast("Report comparison complete!", icon="\u2705")

            send_browser_notification(
                "SynthNotes AI - Report Comparison Complete",
                f"{len(file_texts)} annual reports compared in {elapsed:.1f}s"
            )

        except Exception as e:
            status_text.markdown(f"**Error:** {e}")
            st.error(f"Comparison generation failed: {e}")
        finally:
            st.session_state.rc_processing = False

    # --- Step 5: Display output ---
    if st.session_state.rc_comparison_output:
        st.divider()
        st.markdown("### Output")

        view_mode = st.pills(
            "View", ["Comparison", "Per-Report Extractions"],
            default="Comparison", key="rc_view_mode"
        )

        if view_mode == "Comparison":
            with st.container(height=600, border=True):
                st.markdown(st.session_state.rc_comparison_output)
            note_wc = len(st.session_state.rc_comparison_output.split())
            st.caption(f"{note_wc:,} words")
        else:
            extractions = st.session_state.rc_per_report_extractions
            if extractions:
                tab_names = [fname for fname, _ in extractions]
                tabs = st.tabs(tab_names)
                for tab, (fname, extraction) in zip(tabs, extractions):
                    with tab:
                        with st.container(height=500, border=True):
                            st.markdown(extraction)
                        wc = len(extraction.split())
                        st.caption(f"{wc:,} words")

        # Actions bar
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            copy_to_clipboard_button(st.session_state.rc_comparison_output, "Copy Comparison")
        dl2.download_button(
            label="Download (.txt)",
            data=st.session_state.rc_comparison_output,
            file_name=f"Report_Comparison_{company_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        dl3.download_button(
            label="Download (.md)",
            data=st.session_state.rc_comparison_output,
            file_name=f"Report_Comparison_{company_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Save to database
        st.divider()
        if st.button("Save to Notes History", use_container_width=True, key="rc_save_btn"):
            try:
                note_data = {
                    'id': str(uuid.uuid4()),
                    'created_at': datetime.now().isoformat(),
                    'meeting_type': 'Report Comparison',
                    'file_name': f"Report Comparison — {company_name}",
                    'content': st.session_state.rc_comparison_output,
                    'raw_transcript': "\n\n---\n\n".join(
                        f"--- {fname} ---\n{text[:5000]}..." if len(text) > 5000 else f"--- {fname} ---\n{text}"
                        for fname, text in st.session_state.rc_texts
                    ),
                    'refined_transcript': None,
                    'token_usage': 0,
                    'processing_time': 0,
                    'pdf_blob': None
                }
                database.save_note(note_data)
                st.toast("Saved to Notes History!", icon="\u2705")
                st.session_state.app_state.active_note_id = note_data['id']
            except Exception as e:
                st.error(f"Failed to save: {e}")


def render_ec_analysis_tab(state: AppState):
    st.subheader("Multi-Transcript Earnings Call Analysis")
    st.caption(
        "Upload multiple earnings call PDFs. The system will identify key topics from the first "
        f"{MAX_TOPIC_DISCOVERY_FILES} transcripts, let you select and customize topics, "
        "then generate structured notes for every file."
    )

    # --- Session state initialization ---
    if "ec_analysis_files" not in st.session_state:
        st.session_state.ec_analysis_files = None
    if "ec_analysis_texts" not in st.session_state:
        st.session_state.ec_analysis_texts = []
    if "ec_discovered_topics" not in st.session_state:
        st.session_state.ec_discovered_topics = None
    if "ec_selected_topics" not in st.session_state:
        st.session_state.ec_selected_topics = None
    if "ec_analysis_output" not in st.session_state:
        st.session_state.ec_analysis_output = ""
    if "ec_analysis_processing" not in st.session_state:
        st.session_state.ec_analysis_processing = False
    if "ec_file_notes" not in st.session_state:
        st.session_state.ec_file_notes = []

    # --- Step 1: Upload multiple PDFs ---
    st.markdown("### Step 1: Upload Earnings Call Transcripts")
    uploaded_files = st.file_uploader(
        "Upload PDF transcripts",
        type=["pdf"],
        accept_multiple_files=True,
        key="ec_multi_pdf_uploader",
        help="Upload 2 or more earnings call transcript PDFs for the same company."
    )

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("Upload at least 2 PDF transcripts to begin. For best topic discovery, upload 4 or more.")
        # Reset downstream state if files changed
        if not uploaded_files:
            st.session_state.ec_analysis_texts = []
            st.session_state.ec_discovered_topics = None
            st.session_state.ec_selected_topics = None
            st.session_state.ec_analysis_output = ""
            st.session_state.ec_file_notes = []
        return

    # Show file list
    file_names = [f.name for f in uploaded_files]
    st.caption(f"**{len(uploaded_files)}** files uploaded: {', '.join(file_names)}")

    # --- Step 2: Extract text and discover topics ---
    st.divider()
    st.markdown("### Step 2: Identify Topics")
    st.caption(
        f"Analyzes the first {min(len(uploaded_files), MAX_TOPIC_DISCOVERY_FILES)} transcripts "
        "to identify primary topics (brands, segments) and sub-topics (strategy, unit economics, etc.)."
    )

    # Model selection for analysis
    analysis_model = st.selectbox(
        "Model for Topic Discovery",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="ec_analysis_model_select"
    )

    if st.button("Identify Topics", use_container_width=True, key="ec_discover_btn"):
        with st.spinner("Extracting text from PDFs and identifying topics..."):
            try:
                # Extract texts
                file_texts = _extract_pdf_texts(uploaded_files)
                if len(file_texts) < 2:
                    st.error("Could not extract text from at least 2 PDFs. Check file quality.")
                    return
                st.session_state.ec_analysis_texts = file_texts

                # Discover topics
                discovered = _discover_topics(file_texts, analysis_model)
                st.session_state.ec_discovered_topics = discovered

                # Initialize selection with all topics selected
                st.session_state.ec_selected_topics = copy.deepcopy(discovered)
                st.session_state.ec_analysis_output = ""
                st.session_state.ec_file_notes = []
                st.rerun()
            except json.JSONDecodeError:
                st.error("Failed to parse topic discovery results. Try again or use a different model.")
            except Exception as e:
                st.error(f"Topic discovery failed: {e}")

    # --- Step 3: Display and select topics ---
    discovered = st.session_state.ec_discovered_topics
    if not discovered:
        return

    company_name = discovered.get("company_name", "Unknown Company")
    st.success(f"Topics identified for **{company_name}**")

    st.divider()
    st.markdown("### Step 3: Select & Customize Topics")
    st.caption("Check the topics you want included in the final notes. You can also add custom topics.")

    primary_topics = discovered.get("primary_topics", [])
    cross_cutting = discovered.get("cross_cutting_topics", [])

    # Build selected topics structure from user interaction
    selected_primary = []
    for p_idx, primary in enumerate(primary_topics):
        p_name = primary.get("name", f"Topic {p_idx+1}")
        p_desc = primary.get("description", "")
        sub_topics = primary.get("sub_topics", [])

        # Primary topic toggle
        p_key = f"ec_primary_{p_idx}"
        p_enabled = st.checkbox(
            f"**{p_name}**" + (f" — {p_desc}" if p_desc else ""),
            value=True,
            key=p_key
        )

        if p_enabled:
            # Sub-topic selection
            selected_subs = []
            sub_cols = st.columns(min(len(sub_topics), 4)) if sub_topics else []
            for s_idx, sub in enumerate(sub_topics):
                col = sub_cols[s_idx % len(sub_cols)] if sub_cols else st
                s_key = f"ec_sub_{p_idx}_{s_idx}"
                if col.checkbox(sub, value=True, key=s_key):
                    selected_subs.append(sub)

            # Add custom sub-topic
            custom_sub = st.text_input(
                "Add custom sub-topic",
                key=f"ec_custom_sub_{p_idx}",
                placeholder="e.g., digital transformation, new market entry..."
            )
            if custom_sub and custom_sub.strip():
                for cs in [s.strip() for s in custom_sub.split(",") if s.strip()]:
                    if cs not in selected_subs:
                        selected_subs.append(cs)

            if selected_subs:
                selected_primary.append({
                    "name": p_name,
                    "description": p_desc,
                    "sub_topics": selected_subs
                })

        st.divider()

    # Cross-cutting topics
    if cross_cutting:
        st.markdown("**Cross-Cutting Topics:**")
        selected_cross = []
        cross_cols = st.columns(min(len(cross_cutting), 4))
        for c_idx, cross in enumerate(cross_cutting):
            c_name = cross.get("name", f"Cross-topic {c_idx+1}")
            c_desc = cross.get("description", "")
            col = cross_cols[c_idx % len(cross_cols)]
            c_key = f"ec_cross_{c_idx}"
            if col.checkbox(c_name + (f" — {c_desc}" if c_desc else ""), value=True, key=c_key):
                selected_cross.append(cross)
        st.divider()
    else:
        selected_cross = []

    # Add custom primary topic
    with st.expander("Add Custom Primary Topic", expanded=False):
        custom_p_name = st.text_input("Primary Topic Name", key="ec_custom_primary_name",
                                       placeholder="e.g., New Business Segment")
        custom_p_desc = st.text_input("Description (optional)", key="ec_custom_primary_desc")
        custom_p_subs = st.text_input("Sub-topics (comma-separated)", key="ec_custom_primary_subs",
                                       placeholder="e.g., strategy, revenue, expansion")
        if custom_p_name and custom_p_name.strip():
            subs = [s.strip() for s in custom_p_subs.split(",") if s.strip()] if custom_p_subs else []
            selected_primary.append({
                "name": custom_p_name.strip(),
                "description": custom_p_desc.strip() if custom_p_desc else "",
                "sub_topics": subs
            })

    # Store final selection
    final_selection = {
        "company_name": company_name,
        "primary_topics": selected_primary,
        "cross_cutting_topics": selected_cross
    }
    st.session_state.ec_selected_topics = final_selection

    # Preview selected structure
    with st.expander("Preview Selected Topic Structure", expanded=False):
        structure_text = _build_topic_structure_text(final_selection)
        st.code(structure_text, language="markdown")

    if not selected_primary and not selected_cross:
        st.warning("Select at least one topic to generate notes.")
        return

    # --- Step 4: Generate notes for all files ---
    st.divider()
    st.markdown("### Step 4: Generate Notes")

    file_texts = st.session_state.ec_analysis_texts
    if not file_texts:
        st.warning("File texts not available. Please re-run topic identification.")
        return

    st.caption(f"Will generate structured notes for **{len(file_texts)}** transcripts under the selected topics.")

    # Model selection for note generation
    notes_model = st.selectbox(
        "Model for Notes Generation",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(state.notes_model),
        key="ec_notes_model_select"
    )

    if st.button("Generate All Notes", type="primary", use_container_width=True, key="ec_generate_all_btn"):
        st.session_state.ec_analysis_processing = True
        st.rerun()

    if st.session_state.ec_analysis_processing:
        topic_structure_text = _build_topic_structure_text(final_selection)
        total_tokens = 0
        all_file_notes = []
        start_time = time.time()

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for i, (fname, text) in enumerate(file_texts):
                pct = i / len(file_texts)
                progress_bar.progress(pct)
                status_text.markdown(f"**{int(pct*100)}%** — Processing **{fname}** ({i+1}/{len(file_texts)})")

                notes_text, tokens = _generate_notes_for_file(fname, text, topic_structure_text, notes_model)
                total_tokens += tokens
                all_file_notes.append((fname, notes_text))

            progress_bar.progress(1.0)
            status_text.markdown("**100%** — Stitching notes together...")

            # Stitch
            stitched = _stitch_multi_file_notes(company_name, all_file_notes)

            st.session_state.ec_analysis_output = stitched
            st.session_state.ec_file_notes = all_file_notes

            elapsed = time.time() - start_time
            status_text.markdown(
                f"**100%** — Done! {len(file_texts)} files processed, "
                f"{total_tokens:,} tokens used, {elapsed:.1f}s elapsed."
            )
            st.toast("Earnings call analysis complete!", icon="\u2705")

            # Browser notification
            send_browser_notification(
                "SynthNotes AI - EC Analysis Complete",
                f"{len(file_texts)} transcripts analyzed in {elapsed:.1f}s"
            )

        except Exception as e:
            status_text.markdown(f"**Error:** {e}")
            st.error(f"Notes generation failed: {e}")
        finally:
            st.session_state.ec_analysis_processing = False

    # --- Step 5: Display output ---
    if st.session_state.ec_analysis_output:
        st.divider()
        st.markdown("### Output")

        # View mode
        view_mode = st.pills("View", ["Combined", "Per-File"], default="Combined", key="ec_view_mode")

        if view_mode == "Combined":
            with st.container(height=600, border=True):
                st.markdown(st.session_state.ec_analysis_output)
            note_wc = len(st.session_state.ec_analysis_output.split())
            st.caption(f"{note_wc:,} words")
        else:
            # Per-file tabs
            file_notes = st.session_state.ec_file_notes
            if file_notes:
                tab_names = [fname for fname, _ in file_notes]
                tabs = st.tabs(tab_names)
                for tab, (fname, notes) in zip(tabs, file_notes):
                    with tab:
                        with st.container(height=500, border=True):
                            st.markdown(notes)
                        wc = len(notes.split())
                        st.caption(f"{wc:,} words")

        # Actions bar
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            copy_to_clipboard_button(st.session_state.ec_analysis_output, "Copy All Notes")
        dl2.download_button(
            label="Download (.txt)",
            data=st.session_state.ec_analysis_output,
            file_name=f"EC_Analysis_{company_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        dl3.download_button(
            label="Download (.md)",
            data=st.session_state.ec_analysis_output,
            file_name=f"EC_Analysis_{company_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        # Save to database option
        st.divider()
        if st.button("Save to Notes History", use_container_width=True, key="ec_save_btn"):
            try:
                note_data = {
                    'id': str(uuid.uuid4()),
                    'created_at': datetime.now().isoformat(),
                    'meeting_type': 'Earnings Call',
                    'file_name': f"EC Analysis — {company_name}",
                    'content': st.session_state.ec_analysis_output,
                    'raw_transcript': "\n\n---\n\n".join(
                        f"--- {fname} ---\n{text[:5000]}..." if len(text) > 5000 else f"--- {fname} ---\n{text}"
                        for fname, text in st.session_state.ec_analysis_texts
                    ),
                    'refined_transcript': None,
                    'token_usage': 0,
                    'processing_time': 0,
                    'pdf_blob': None
                }
                database.save_note(note_data)
                st.toast("Saved to Notes History!", icon="\u2705")
                st.session_state.app_state.active_note_id = note_data['id']
            except Exception as e:
                st.error(f"Failed to save: {e}")


# --- 6. MAIN APPLICATION RUNNER ---
def run_app():
    st.set_page_config(page_title="SynthNotes AI", layout="wide", page_icon="🤖")

    # Inject app-wide CSS (navigation highlights, spacing, responsive)
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.logo("https://placehold.co/64x64?text=SN", link="https://streamlit.io")

    # --- Header with dark mode toggle ---
    title_col, theme_col = st.columns([6, 1])
    with title_col:
        st.title("SynthNotes AI")
    with theme_col:
        # Detect current theme and show toggle
        current_theme = st.context.theme
        is_dark = current_theme.get("backgroundColor", "#ffffff").lower() in (
            "#0e1117", "#111111", "#000000", "#0e1118", "#262730",
        )
        dark_mode = st.toggle(
            "Dark" if is_dark else "Light",
            value=is_dark,
            key="dark_mode_toggle",
            help="Switch between light and dark mode",
        )
        if dark_mode != is_dark:
            # Inject JS to toggle Streamlit's theme via settings menu
            target_theme = "Dark" if dark_mode else "Light"
            components.html(
                f"""
                <script>
                // Toggle theme by updating localStorage and reloading
                const stTheme = '{target_theme.lower()}';
                try {{
                    const key = Object.keys(localStorage).find(k => k.includes('stActiveTheme')) || 'stActiveTheme-/-v1';
                    localStorage.setItem(key, JSON.stringify({{name: stTheme, themeInput: {{}}}}));
                    window.parent.location.reload();
                }} catch(e) {{
                    // Fallback: use URL params
                    const url = new URL(window.parent.location);
                    url.searchParams.set('embed_options', 'dark_theme' === stTheme ? 'dark_theme' : 'light_theme');
                    window.parent.location = url;
                }}
                </script>
                """,
                height=0,
            )

    if "config_error" in st.session_state:
        st.error(st.session_state.config_error); st.stop()

    try:
        database.init_db()
    except Exception as db_err:
        st.error(f"Failed to initialize database: {db_err}")
        st.info("The app may not be able to save notes. Check database permissions.")

    try:
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
            on_sector_change()
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = {}

        def _page_input():
            try:
                render_input_and_processing_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in Input tab: {tab_err}")

        def _page_output():
            try:
                render_output_and_history_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in Output tab: {tab_err}")

        def _page_otg():
            try:
                render_otg_notes_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in OTG Notes tab: {tab_err}")

        def _page_ec_analysis():
            try:
                render_ec_analysis_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in EC Analysis tab: {tab_err}")

        def _page_report_comparison():
            try:
                render_report_comparison_tab(st.session_state.app_state)
            except Exception as tab_err:
                st.error(f"Error in Report Comparison tab: {tab_err}")

        nav = st.navigation(
            [
                st.Page(_page_input, title="Input & Generate", icon=":material/edit_note:"),
                st.Page(_page_output, title="Output & History", icon=":material/history:"),
                st.Page(_page_ec_analysis, title="EC Analysis", icon=":material/analytics:"),
                st.Page(_page_report_comparison, title="Report Compare", icon=":material/compare:"),
                st.Page(_page_otg, title="OTG Notes", icon=":material/quick_phrases:"),
            ],
            position="top",
        )
        nav.run()

    except Exception as e:
        st.error("A critical application error occurred."); st.code(traceback.format_exc())

if __name__ == "__main__":
    run_app()

# /------------------------\
# |   END OF app.py FILE   |
# \------------------------/
