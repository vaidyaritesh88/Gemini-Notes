import streamlit as st
import google.generativeai as genai
import os, io, re, time, tempfile, json, html as html_module, subprocess, glob
from datetime import datetime
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
    "Gemini 3.5 Flash":              "gemini-3.5-flash",
    "Gemini 2.0 Flash":              "gemini-2.0-flash-lite",
    "Gemini 1.5 Flash":              "gemini-1.5-flash",
}

# Approximate pricing per 1M tokens (USD), under 200K-token context. Verified May 2026.
# Tuple = (input_price, output_price). Used only by the in-app cost panel; for
# authoritative billing see your Google Cloud project's billing reports. Audio input
# tokens are billed at the text-input rate in this table — a small underestimate for
# audio-heavy workflows, flagged in the cost-panel caption.
MODEL_PRICING = {
    "gemini-2.5-pro":         (1.25, 10.00),
    "gemini-2.5-flash":       (0.30,  2.50),
    "gemini-2.5-flash-lite":  (0.10,  0.40),
    "gemini-3.5-flash":       (0.50,  3.00),  # estimate — exact pricing TBD
    "gemini-3-flash-preview": (0.50,  3.00),
    "gemini-3.1-flash-lite":  (0.25,  1.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.0-flash-lite":  (0.075, 0.30),
    "gemini-1.5-flash":       (0.075, 0.30),
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
# Expert Meeting prompts are the original, proven, high-quality capture prompts.
# Management Meeting and Internal Discussion prompts use the SAME Q&A structure as
# Expert (because that format gives the highest-fidelity output), but with
# meeting-type-appropriate terminology — management is "management" or the named role
# (CEO/CFO/etc.), never "the expert"; internal participants are "the speaker(s)" or
# "the team", never "experts". The Pro difference still comes from the intelligence
# extraction layer below.

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

MANAGEMENT_MEETING_DETAILED_PROMPT = """### **PRIMARY DIRECTIVE: MAXIMUM DETAIL & STRICT COMPLETENESS**
Your goal is to produce the most thorough, granular notes possible from a **management meeting** (e.g. an analyst's call with company management, an investor meeting, or a one-on-one with an executive). Remove conversational filler ("um," "you know," repetition) but **nothing substantive should be omitted.** Every factual claim, example, explanation, aside, and data point in the transcript must appear in your notes. When in doubt, INCLUDE it. Err heavily on the side of over-inclusion. Longer, more detailed notes are always preferred over concise ones.

### **TERMINOLOGY — DO NOT MISLABEL THE SPEAKERS**
This is a **management meeting**, NOT an expert consultation. Refer to the company-side speakers as **"management"**, or by their specific role/name where known ("the CEO", "the CFO", "the Head of Strategy"). **NEVER refer to management as "the expert" or "the experts."** If specific names or roles are provided in the speaker context, use those. If the analyst side asks questions, refer to them as "the analyst" where needed.

### **NOTES STRUCTURE**

**(1.) Opening overview or Management background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or management introductions, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, titles, roles, tenure, prior positions, segments/divisions owned, company background where stated).
- **DO NOT:** Summarize or include sell-side/broker boilerplate.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body STRICTLY in Question/Answer format. Most analyst-management meetings ARE Q&A — capture them that way. If the transcript is monologue-style (e.g. prepared management commentary or an opening statement), convert each distinct topic management addresses into a bold question that captures what was at issue, and the substance becomes the bulleted answer below.

**(2.A) Questions:**
-   Identify the core question being asked and rephrase it clearly in **bold**. Do NOT copy the question verbatim from the transcript — clean up filler, false starts, and rambling phrasing into a clear, well-formed question that preserves the original intent.
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or any similar label. The bold question text stands alone.
-   If the analyst provides context, framing, or a multi-part question, capture the full scope — do not reduce a multi-part question to a single line.
-   **LONG QUESTIONS / PREAMBLE:** Substantive analyst framing or background is part of the question and must be preserved as part of the bold question text. Do NOT treat the preamble as part of the answer.
-   **SPACING:** Leave exactly one blank line between the end of one answer and the start of the next bold question, so each Q&A pair is visually separated.

**(2.B) Answers (management's response):**
-   Use bullet points (`-`) directly below the question (no blank line between the bold question and its first bullet).
-   Each bullet point must convey specific factual information in a clear, complete sentence.
-   Use **multiple bullet points** per answer — do NOT collapse a detailed response into a single bullet.
-   **ZERO SKIPPING RULE:** If management said it with substance, it must appear in your notes. Do NOT skip examples, anecdotes, specific sentences, or supporting details even if they seem minor. Every distinct point gets its own bullet. If an answer contains 8 substantive points, you must produce at least 8 bullets — never condense them into 3-4.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** This includes all names, monetary values (`$`, `₹`), percentages (`%`), revenue/EBITDA/margin figures, growth rates, capex, capacity, store counts, market shares, customer counts, named segments, brand names, product names, customer names, geographies, and time periods.
-   **PRIORITY #2: CAPTURE ALL NUANCE & REASONING.** Do not over-summarize or reduce complex answers to surface-level statements. You must retain:
    -   **Sentiment & Tone:** Note if management is confident, cautious, hedging, defensive, or enthusiastic (e.g., "management was highly confident that...," "the CFO cautioned that...," "management hedged on...").
    -   **Qualifiers & Conditions:** Preserve modifying words that change meaning (e.g., "typically," "in most cases," "subject to," "we expect," "roughly," "approximately," "should").
    -   **Guidance & Outlook:** Preserve forward-looking statements with their exact qualifiers ("we are targeting...", "we expect...", "we are guiding to...", "we believe we can achieve...").
    -   **Key Examples & Anecdotes:** If management uses a specific example, customer win, deal reference, plant/store-level anecdote, or case study to illustrate a point, capture it in full — these are often the highest-signal parts of a management call.
    -   **Cause & Effect:** Retain any reasoning chains provided (e.g., "...because input costs eased," "...which drove the 80bps margin expansion").
    -   **Comparisons & Contrasts:** If management compares segments, geographies, time periods, competitors, or business lines, capture both sides with the specific details for each.
    -   **Tangential but relevant points:** If management volunteers additional context, background, or related information beyond the direct question, include it — do NOT discard it as off-topic.
-   **PRIORITY #3: PRESERVE MULTI-STEP EXPLANATIONS.** If an answer involves a sequence of steps, a timeline, a capital allocation plan, or a logical chain, preserve the full sequence rather than summarizing the conclusion only.
-   **PRIORITY #4: PRESERVE COMMITMENTS, TARGETS, AND DECISIONS INLINE.** When management announces a target, deadline, capex commitment, or strategic decision, capture it within the relevant Q&A bullet — do NOT strip these into a separate "decisions" or "action items" section. They belong with the context in which management discussed them."""

MANAGEMENT_MEETING_CONCISE_PROMPT = """### **PRIMARY DIRECTIVE: EFFICIENT & NUANCED**
Your goal is to be **efficient**, not just brief, when capturing a **management meeting** (analyst-management call, investor meeting, one-on-one with an executive). Remove conversational filler ("um," "you know," repetition) but **preserve all substantive information**. Your output should be concise yet information-dense.

### **TERMINOLOGY — DO NOT MISLABEL THE SPEAKERS**
This is a **management meeting**, NOT an expert consultation. Refer to the company-side speakers as **"management"**, or by their specific role/name where known ("the CEO", "the CFO", "the Head of Strategy"). **NEVER refer to management as "the expert" or "experts".** If the analyst side asks questions, refer to them as "the analyst" where needed.

### **NOTES STRUCTURE**

**(1.) Opening overview or Management background (Conditional):**
- If the transcript chunk begins with an overview, agenda, or management introductions, include it FIRST as bullet points.
- **DO:** Capture ALL details (names, titles, roles, segments owned).
- **DO NOT:** Summarize.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format:**
Structure the main body in Question/Answer format. If the transcript is monologue-style (prepared management commentary, opening remarks), convert each distinct topic management addresses into a bold question with the substance as bulleted answers below.

**(2.A) Questions:**
-   Identify the core question and rephrase it clearly in **bold**. Do NOT copy verbatim — clean up filler and rambling into a clear, well-formed question.
-   **NO LABELS:** Do NOT prefix questions with "Q:", "Q.", "Question:", or similar.
-   **PREAMBLE:** Preserve substantive analyst framing as part of the bold question text.
-   **SPACING:** One blank line between the end of one answer and the next bold question.

**(2.B) Answers (management's response):**
-   Use bullet points (`-`) directly below the question.
-   Each bullet must convey specific factual information in a clear, complete sentence.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** Numbers, percentages, monetary figures, growth rates, capex, capacity, market shares, guidance, targets, named segments, brands, geographies, time periods.
-   **PRIORITY #2: CAPTURE ALL NUANCE.** Sentiment (confident, cautious, hedging, defensive), qualifiers ("we expect", "we are targeting", "subject to"), key examples and anecdotes, cause & effect chains, comparisons across segments/geographies/competitors.
-   **PRIORITY #3: PRESERVE COMMITMENTS INLINE.** Targets, deadlines, capex commitments, strategic decisions — capture within the relevant Q&A bullet; do NOT strip into a separate section."""

INTERNAL_DISCUSSION_DETAILED_PROMPT = """### **PRIMARY DIRECTIVE: MAXIMUM DETAIL & STRICT COMPLETENESS**
Your goal is to produce the most thorough, granular notes possible from an **internal team discussion** (e.g. a research-team debate, an investment committee discussion, a strategy meeting among colleagues). Remove conversational filler ("um," "you know," repetition) but **nothing substantive should be omitted.** Every factual claim, example, position, counter-argument, and data point in the transcript must appear in your notes. When in doubt, INCLUDE it. Longer, more detailed notes are always preferred over concise ones.

### **TERMINOLOGY — DO NOT MISLABEL THE SPEAKERS**
This is an **internal discussion among colleagues**, NOT an expert consultation. Refer to participants as **"the speaker(s)"**, **"the team"**, or by name where named in the transcript or speaker context. **NEVER refer to participants as "experts" or "the expert".** Where the transcript attributes views to specific people, preserve that attribution (e.g., "X argued that...", "Y countered with...").

### **NOTES STRUCTURE**

**(1.) Opening overview or Discussion context (Conditional):**
- If the transcript chunk begins with an overview, agenda, or context-setting, include it FIRST as bullet points.
- **DO:** Capture stated purpose, the question or decision being discussed, participants (names, roles), any background framing.
- **DO NOT:** Summarize or include meeting logistics.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format (adapted for discussion):**
Structure the main body STRICTLY in Question/Answer format. An internal discussion is rarely a literal Q&A — to apply this structure, **treat each distinct issue, question, or topic raised in the discussion as a bold "question"**, and capture the discussion that followed as the bulleted "answer" below it. The bold "question" should be a clean, well-formed statement of *what was at issue*; the bullets capture *what the team said about it*.

**(2.A) Questions (issues / topics raised):**
-   Identify each distinct issue, question, or decision point raised in the discussion and rephrase it clearly in **bold**. Examples: *"Whether to add to the position given the Q3 miss"*, *"How to interpret the channel-check feedback on pricing"*, *"What the regulatory change means for the thesis"*.
-   **NO LABELS:** Do NOT prefix with "Q:", "Topic:", "Issue:", or any similar label. The bold statement stands alone.
-   If an issue has multiple sub-questions, capture the full scope — do not reduce to a single line.
-   **PREAMBLE:** Substantive framing that introduces the issue (background, prior context, why it was raised) is part of the bold statement and must be preserved within it.
-   **SPACING:** Leave exactly one blank line between the end of one discussion and the start of the next bold issue.

**(2.B) Answers (the discussion that followed):**
-   Use bullet points (`-`) directly below the bold issue (no blank line between them).
-   Each bullet point must convey a specific position, argument, piece of evidence, or data point in a clear, complete sentence.
-   **ATTRIBUTION:** Where the transcript attributes a view to a specific person, preserve it in the bullet ("X argued that...", "Y countered..."). Where views are shared or unattributed, state them without forcing attribution.
-   **ZERO SKIPPING RULE:** Every distinct point, argument, counter-argument, and piece of evidence raised must appear as its own bullet. Do NOT condense multiple positions into one bullet. If the discussion contains 8 substantive points, you must produce at least 8 bullets.
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** Numbers, percentages, dates, named companies, named products/people/geographies, prior decisions or positions referenced, external data sources cited.
-   **PRIORITY #2: CAPTURE ALL POSITIONS, REASONING, AND DISAGREEMENT.** Do not over-summarize. Retain:
    -   **All sides:** Capture every distinct view raised, including dissenting and minority views. Do NOT favour the dominant view or strip out the dissent.
    -   **Reasoning chains:** Preserve the "why" behind each position (e.g., "...because the comparable in 2019 was different," "...assuming the new capacity comes online by Q2").
    -   **Evidence cited:** Capture any data, examples, prior calls, prior notes, or external references invoked to support a position.
    -   **Sentiment & confidence:** Note where the speaker was confident, uncertain, hedging, or changing their view mid-discussion ("X was initially sceptical but came around when...").
    -   **Agreement vs disagreement:** Make these explicit within the bullets ("the team agreed that...", "X and Y disagreed on...").
    -   **Caveats and risks raised:** Preserve in the speaker's own framing.
-   **PRIORITY #3: PRESERVE MULTI-STEP REASONING.** If a participant builds an argument step-by-step, preserve the full chain rather than the conclusion only.
-   **PRIORITY #4: PRESERVE CONCLUSIONS AND OPEN ITEMS INLINE.** If the discussion reaches a conclusion or leaves something open, capture it within the relevant Q&A bullet — do NOT strip these into a separate "next steps" or "conclusions" section. They belong with the discussion that produced them."""

INTERNAL_DISCUSSION_CONCISE_PROMPT = """### **PRIMARY DIRECTIVE: EFFICIENT & NUANCED**
Your goal is to be **efficient**, not just brief, when capturing an **internal team discussion** (research-team debate, investment committee, strategy meeting among colleagues). Remove conversational filler ("um," "you know," repetition) but **preserve all substantive information**. Your output should be concise yet information-dense.

### **TERMINOLOGY — DO NOT MISLABEL THE SPEAKERS**
This is an **internal discussion among colleagues**, NOT an expert consultation. Refer to participants as **"the speaker(s)"**, **"the team"**, or by name where named in the transcript or speaker context. **NEVER refer to participants as "experts" or "the expert".** Preserve attribution where the transcript provides it ("X argued that...", "Y countered...").

### **NOTES STRUCTURE**

**(1.) Opening overview or Discussion context (Conditional):**
- If the transcript chunk begins with an overview or context-setting, include it FIRST as bullet points.
- **DO:** Capture stated purpose, the question being discussed, participants.
- **DO NOT:** Summarize.
- If no intro exists, OMIT this section entirely.

**(2.) Q&A format (adapted for discussion):**
Structure the main body in Question/Answer format. **Treat each distinct issue, question, or topic raised in the discussion as a bold "question"**, and capture the discussion that followed as the bulleted "answer".

**(2.A) Questions (issues / topics raised):**
-   Identify each distinct issue and rephrase it clearly in **bold**. Examples: *"Whether to add to the position given the Q3 miss"*, *"How to interpret the channel-check feedback on pricing"*.
-   **NO LABELS:** Do NOT prefix with "Q:", "Topic:", "Issue:", or similar.
-   **PREAMBLE:** Preserve substantive framing as part of the bold statement.
-   **SPACING:** One blank line between the end of one discussion and the next bold issue.

**(2.B) Answers (the discussion that followed):**
-   Use bullet points (`-`) directly below the bold issue.
-   Each bullet must convey a specific position, argument, or data point.
-   **ATTRIBUTION:** Preserve where the transcript provides it ("X argued...", "Y countered...").
-   **PRIORITY #1: CAPTURE ALL HARD DATA.** Numbers, percentages, named companies, products, geographies, dates, prior decisions or references invoked.
-   **PRIORITY #2: CAPTURE ALL POSITIONS AND REASONING.** All sides including minority views; the "why" behind each position; evidence cited; sentiment / confidence; agreement vs disagreement made explicit.
-   **PRIORITY #3: PRESERVE CONCLUSIONS INLINE.** Capture within the relevant Q&A bullet; do NOT strip into a separate "next steps" or "conclusions" section."""

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

INTEL_MANAGEMENT_PROMPT = """You are a research analyst extracting structured intelligence from management meeting notes.

**CRITICAL RULE: Only include what management actually said. Do not infer, interpret, or add your own analysis. Every point must be traceable to a specific statement in the notes.**

**TERMINOLOGY: Refer to the company-side speakers as "management" or by their specific role ("the CEO", "the CFO", "the COO") — NEVER as "the expert" or "experts".**

---

## MANAGEMENT BACKGROUND
[Only if the notes contain background information about the speakers, their roles, tenure, or the company/segments they cover. Skip this section entirely if no such background is present in the notes.]

## CORE THESIS
Management's 2–3 overarching views as stated in the meeting — their headline positions on the business, the market, or the strategic direction. Use management's own framing where possible.

## KEY INSIGHTS
The most important, substantive things management said — prioritising non-obvious points, strategic shifts, and specific commitments over generic context. Each bullet should be a specific, complete statement. Aim for 6–10 bullets.
- Do NOT include widely-known facts unless management made a specific new claim about them.
- Do NOT collapse multiple distinct points into one bullet.

## HARD DATA & FACTS
Every specific figure, statistic, target, or named reference management provided:
- Numbers, percentages, growth rates, margins, capex, capacity, market shares, customer counts
- Guidance, targets, and forward-looking numbers with their stated qualifiers
- Named companies, products, brands, segments, geographies with specific context
- Dates, timelines, durations
Format: one bullet per data point with just enough context to understand it.

## DIRECT QUOTES
2–4 verbatim sentences from management that best capture their views or commitments. Choose the most specific and quotable lines.
Format: *"[exact quote]"* — [brief topic label, with speaker role if known]

## EXPRESSED RISKS & UNCERTAINTIES
Things management themselves flagged as risks, concerns, headwinds, downside scenarios, or areas of uncertainty. Note their stated confidence level where apparent (e.g., "management cautioned that...", "the CFO was uncertain about...").
Only include risks management actually raised — do NOT add risks of your own.

## STATED NON-CONSENSUS VIEWS
Views management themselves described as contrarian, differentiated, or against conventional wisdom — or views that are clearly at odds with a commonly-held position as stated in the notes.
Leave this section blank if the notes contain no such views.

## QUESTIONS & UNCERTAINTIES MANAGEMENT RAISED
Open questions, unresolved issues, or areas management themselves said need further work, monitoring, or future visibility. Only include things management explicitly flagged — not questions you think are interesting.

---
MEETING NOTES:
{notes}
"""

INTEL_INTERNAL_PROMPT = """You are a research analyst extracting structured intelligence from internal team discussion notes.

**CRITICAL RULE: Only include what was actually said in the discussion. Do not infer, interpret, or add your own analysis. Every point must be traceable to a specific statement in the notes.**

**TERMINOLOGY: Refer to participants as "the speaker(s)", "the team", or by name where named — NEVER as "experts" or "the expert".**

---

## DISCUSSION CONTEXT
[Only if the notes contain background about the discussion's purpose, the question or decision being debated, or the participants. Skip this section entirely if no such context is present in the notes.]

## CORE THESIS
The 2–3 overarching views or working hypotheses that emerged in the discussion — the headline positions held by the team. Where attribution exists in the notes, preserve it.

## KEY INSIGHTS
The most important, substantive points raised in the discussion — prioritising non-obvious arguments, new evidence, and shifts in view over restating prior beliefs. Each bullet should be a specific, complete statement. Aim for 6–10 bullets.
- Capture insights from across all participants, not just the dominant voice.
- Do NOT collapse multiple distinct points into one bullet.

## HARD DATA & FACTS
Every specific figure, statistic, or named reference raised in the discussion:
- Numbers, percentages, prices, dates, ratios
- Named companies, products, people, geographies with specific context
- References to prior notes, prior calls, or external data sources
Format: one bullet per data point with just enough context to understand it.

## DIRECT QUOTES
2–4 verbatim sentences from the discussion that best capture the most important arguments or conclusions. Choose the most specific and quotable lines.
Format: *"[exact quote]"* — [speaker name/role if known, brief topic label]

## EXPRESSED RISKS & UNCERTAINTIES
Risks, concerns, or downside scenarios that participants themselves raised in the discussion. Note stated confidence where apparent (e.g., "the speaker was cautious about...", "X flagged the risk that...").
Only include risks participants actually raised — do NOT add risks of your own.

## STATED NON-CONSENSUS VIEWS
Views participants described as contrarian, against the team's prior consensus, or differing from the dominant industry/market view — as recorded in the notes.
Leave this section blank if the notes contain no such views.

## QUESTIONS & UNCERTAINTIES THE TEAM RAISED
Open questions, items the team deferred, or areas participants themselves said need further work or monitoring. Only include things the team explicitly flagged in the discussion — not questions you think are interesting.

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

**LANGUAGE RULES — read carefully before writing:**
- Write in plain, neutral, declarative sentences. No rhetorical flourishes.
- Do NOT use evaluative words you are adding yourself — words like "notably", "importantly", "significantly", "strikingly", "key", "critical", "impressive", "concerning", "surprisingly" — unless the brief itself uses them. These words frame importance; that is the reader's job, not yours.
- Do NOT add connective language ("however", "despite", "yet", "although", "this contrasts with") to imply a tension or relationship unless the brief explicitly records that contrast or tension. Only use a causal connector ("because", "as a result") if the brief states the causal link.
- Do NOT use words that subtly talk up or talk down a point: "only", "merely", "just", "even", "still" (when used for rhetorical effect), "despite only", "managed to", "failed to".
- Quotes should appear exactly as they are in the brief — do not paraphrase into reported speech that softens or sharpens the original.
- The goal is a transcript-faithful account. Coherence comes from grouping related facts together, not from editorial framing.

Target: approximately {word_count} words for the Detailed Summary. The Brief should be ~150 words.
{focus_block}
---

## BRIEF
[~150 words. A concise factual overview: who the expert is (if background is in the brief), the main subject of the call, and the expert's core position. Plain prose, no headers or bullets. Written so it stands alone as a quick reference. Apply the language rules above — no evaluative framing.]

---

## DETAILED SUMMARY

**HOW TO STRUCTURE THE DETAILED SUMMARY:**

Step 1 — Identify the main topics discussed in this call from the intelligence brief. Let the topics emerge from the content — do not force it into a pre-defined skeleton. A typical call has 3–6 main topics.

Step 2 — For each topic, write a **bold topic heading** followed by 2–4 sentences of factual prose. The prose should:
- Group related facts from the brief into sentences rather than presenting each fact as an isolated bullet
- Weave data points, specific numbers, and company names in as part of sentences
- Include a direct quote from the brief inline if one is relevant to this topic (format: *"quote"*) — use the exact wording from the brief
- Only use connective language where the brief records the relationship (e.g. do not write "however" to imply a contrast the brief does not state)
- Capture the expert's stated confidence level or caveats using the expert's own phrasing where possible

Step 3 — If a topic has more than 3–4 distinct data points that are hard to weave into prose without cluttering it, follow the prose with a compact indented list of those specifics only.

Step 4 — After all topic sections, add:

### Topics Noted but Not Covered in Depth
A brief list of subjects that appear in the intelligence brief but were not given their own section — either because they were mentioned briefly or were less central to the call. Format: one line per topic with a short note on what was mentioned.
*This section lets the reader audit what was deprioritised.*

---
INTELLIGENCE BRIEF:
{intelligence}
"""

SUMMARY_MANAGEMENT_PROMPT = """You are writing a factual summary of a management meeting for a professional reader.

**CRITICAL RULE: Only include information that appears in the intelligence brief below. Do not add your own analysis, inferences, or interpretations. If something is not in the brief, do not include it.**

**TERMINOLOGY: Refer to the company-side speakers as "management" or by their specific role ("the CEO", "the CFO") — NEVER as "the expert" or "experts".**

**LANGUAGE RULES — read carefully before writing:**
- Write in plain, neutral, declarative sentences. No rhetorical flourishes.
- Do NOT use evaluative words you are adding yourself — words like "notably", "importantly", "significantly", "strikingly", "key", "critical", "impressive", "concerning", "surprisingly" — unless the brief itself uses them. These words frame importance; that is the reader's job, not yours.
- Do NOT add connective language ("however", "despite", "yet", "although", "this contrasts with") to imply a tension or relationship unless the brief explicitly records that contrast or tension. Only use a causal connector ("because", "as a result") if the brief states the causal link.
- Do NOT use words that subtly talk up or talk down a point: "only", "merely", "just", "even", "still" (when used for rhetorical effect), "managed to", "failed to".
- Quotes should appear exactly as they are in the brief — do not paraphrase into reported speech that softens or sharpens the original.
- The goal is a transcript-faithful account. Coherence comes from grouping related facts together, not from editorial framing.

Target: approximately {word_count} words for the Detailed Summary. The Brief should be ~150 words.
{focus_block}
---

## BRIEF
[~150 words. A concise factual overview: who the management speakers are (if background is in the brief), the main subject of the meeting, and management's core position. Plain prose, no headers or bullets. Written so it stands alone as a quick reference. Apply the language rules above — no evaluative framing.]

---

## DETAILED SUMMARY

**HOW TO STRUCTURE THE DETAILED SUMMARY:**

Step 1 — Identify the main topics discussed in this meeting from the intelligence brief. Let the topics emerge from the content — do not force it into a pre-defined skeleton. A typical management meeting has 3–6 main topics.

Step 2 — For each topic, write a **bold topic heading** followed by 2–4 sentences of factual prose. The prose should:
- Group related facts from the brief into sentences rather than presenting each fact as an isolated bullet
- Weave data points, specific numbers, guidance, and named segments/brands in as part of sentences
- Include a direct quote from the brief inline if one is relevant to this topic (format: *"quote"*) — use the exact wording from the brief
- Only use connective language where the brief records the relationship (e.g. do not write "however" to imply a contrast the brief does not state)
- Capture management's stated confidence level or caveats using management's own phrasing where possible

Step 3 — If a topic has more than 3–4 distinct data points that are hard to weave into prose without cluttering it, follow the prose with a compact indented list of those specifics only.

Step 4 — After all topic sections, add:

### Topics Noted but Not Covered in Depth
A brief list of subjects that appear in the intelligence brief but were not given their own section — either because they were mentioned briefly or were less central to the meeting. Format: one line per topic with a short note on what was mentioned.
*This section lets the reader audit what was deprioritised.*

---
INTELLIGENCE BRIEF:
{intelligence}
"""

SUMMARY_INTERNAL_PROMPT = """You are writing a factual summary of an internal team discussion for a professional reader.

**CRITICAL RULE: Only include information that appears in the intelligence brief below. Do not add your own analysis, inferences, or interpretations. If something is not in the brief, do not include it.**

**TERMINOLOGY: Refer to participants as "the speaker(s)", "the team", or by name where named in the brief — NEVER as "experts" or "the expert".**

**LANGUAGE RULES — read carefully before writing:**
- Write in plain, neutral, declarative sentences. No rhetorical flourishes.
- Do NOT use evaluative words you are adding yourself — words like "notably", "importantly", "significantly", "strikingly", "key", "critical", "concerning", "surprisingly" — unless the brief itself uses them. These words frame importance; that is the reader's job, not yours.
- Do NOT add connective language ("however", "despite", "yet", "although", "this contrasts with") to imply a tension or relationship unless the brief explicitly records that contrast or tension. Only use a causal connector ("because", "as a result") if the brief states the causal link.
- Do NOT use words that subtly talk up or talk down a point: "only", "merely", "just", "even", "still" (when used for rhetorical effect), "managed to", "failed to".
- Quotes should appear exactly as they are in the brief — do not paraphrase.
- Where the brief attributes a view to a specific speaker, preserve that attribution. Where the brief gives no attribution, do not invent one.
- The goal is a transcript-faithful account. Coherence comes from grouping related facts together, not from editorial framing.

Target: approximately {word_count} words for the Detailed Summary. The Brief should be ~150 words.
{focus_block}
---

## BRIEF
[~150 words. A concise factual overview: what the discussion was about, the main positions held by the team, and any conclusion reached. Plain prose, no headers or bullets. Written so it stands alone as a quick reference. Apply the language rules above — no evaluative framing.]

---

## DETAILED SUMMARY

**HOW TO STRUCTURE THE DETAILED SUMMARY:**

Step 1 — Identify the main topics or issues discussed from the intelligence brief. Let the topics emerge from the content — do not force it into a pre-defined skeleton. A typical internal discussion has 3–5 main topics.

Step 2 — For each topic, write a **bold topic heading** followed by 2–4 sentences of factual prose. The prose should:
- Group related facts from the brief into sentences rather than presenting each fact as an isolated bullet
- Weave data points, specific numbers, and named references in as part of sentences
- Include a direct quote from the brief inline if one is relevant to this topic (format: *"quote"*) — use the exact wording from the brief
- Where attribution exists in the brief, preserve it ("X argued that...", "Y held that...", "the team agreed that...") — do not invent attribution where the brief gives none
- Only use connective language where the brief records the relationship

Step 3 — If a topic has more than 3–4 distinct data points that are hard to weave into prose without cluttering it, follow the prose with a compact indented list of those specifics only.

Step 4 — After all topic sections, add:

### Topics Noted but Not Covered in Depth
A brief list of subjects that appear in the intelligence brief but were not given their own section — either because they were mentioned briefly or were less central to the discussion. Format: one line per topic with a short note on what was mentioned.
*This section lets the reader audit what was deprioritised.*

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

Apply the instruction and return the complete revised summary. Maintain the two-tier structure (## BRIEF and ## DETAILED SUMMARY). Only modify what the instruction asks for. Preserve all factual accuracy.

**LANGUAGE RULES (apply throughout):** Plain, neutral, declarative sentences. No evaluative words ("notably", "importantly", "significantly", "concerning") unless they appear in the intelligence brief. No connective language that implies a relationship the brief does not state. No words that talk up or talk down a point ("only", "merely", "managed to", "failed to") used rhetorically."""


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

NOTES_QA_PROMPT = """You are helping a professional get answers from their meeting notes.

The notes below are a factual record of what was said.

**HOW TO RESPOND:**
- If the question is factual ("What did the expert say about X?", "What was decided on Y?"): answer directly from the notes, quoting or closely paraphrasing the relevant section. Clearly state where in the notes the answer comes from.
- If the question is analytical ("What are the implications of X?", "What assumptions is the expert making?"): first state the relevant facts from the notes, then clearly label your own analysis using phrases like "This suggests..." or "A possible interpretation is...".
- If the answer is not in the notes: say so clearly — do not fabricate or fill gaps.

MEETING NOTES:
{notes}

---
QUESTION:
{question}
"""

QUESTION_SUGGESTION_PROMPT = """You are helping a professional analyst decide what to analyse from a meeting.

Read the intelligence brief below and suggest 5 analysis questions that would be genuinely useful to explore.

**RULES FOR GOOD QUESTIONS:**
- Questions must be specific to the content of THIS brief — not generic questions that could apply to any meeting.
- Each question should require reasoning BEYOND what is directly stated in the brief (otherwise it can just be read, not analysed).
- Questions should surface something that a thoughtful analyst would want to think through — hidden assumptions, unstated risks, tensions between views, what the data implies, what is missing.
- Do NOT ask questions that are already directly answered in the brief.
- Do NOT ask generic questions like "What are the risks?" — make them specific to what this expert actually said.

**OUTPUT FORMAT:**
Return exactly 5 questions, one per line, numbered 1–5. No preamble, no explanation — just the questions.

---
INTELLIGENCE BRIEF:
{intelligence}
"""


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


# ── Usage / cost tracking ──────────────────────────────────────────────────────
# Every LLM call records its (input_tokens, output_tokens, model, stage) into the
# session-state log. The cost panel reads this log to render a per-stage table and
# a session-to-date dollar total. Tracking is best-effort: failures here NEVER
# break the pipeline — see the broad except in _record_usage.

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
        pass  # Tracking must never break the pipeline


def compute_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """USD cost for given token counts using the MODEL_PRICING table.
    Returns 0.0 for unknown models (so the panel still renders, just without $)."""
    pricing = MODEL_PRICING.get(model_id)
    if not pricing:
        return 0.0
    in_price, out_price = pricing
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def generate_with_retry(model, prompt, max_retries: int = 3, stream: bool = False,
                        generation_config=None, stage: str = ""):
    """Call the model with retry on transient errors. Records usage for non-streaming
    responses; for streaming, tags the response so stream_and_collect can record
    after iteration completes (usage_metadata is only finalised post-iteration)."""
    kwargs = {"stream": stream}
    if generation_config:
        kwargs["generation_config"] = generation_config
    # google-generativeai prefixes model_name with "models/" — strip for pricing lookup
    model_id = getattr(model, "model_name", "") or ""
    if model_id.startswith("models/"):
        model_id = model_id[len("models/"):]
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, **kwargs)
            if not stream:
                _record_usage(model_id, response, stage)
            else:
                # Tag the streaming response object so stream_and_collect can find it
                try:
                    response._tracked_model_id = model_id
                    response._tracked_stage    = stage
                except (AttributeError, TypeError):
                    pass  # Some response types block dynamic attrs; tracking just skips
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
    # Record usage if generate_with_retry tagged the response on its way out
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
            resp = generate_with_retry(model, [transcription_instruction, cloud], stage="Transcription")
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
        return generate_with_retry(model, prompt, stage="Refinement").text
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
        return idx, generate_with_retry(model, prompt, stage="Refinement").text
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
    detailed = detail_level == "Detailed"
    if meeting_type == "Expert Meeting":
        base = EXPERT_MEETING_DETAILED_PROMPT if detailed else EXPERT_MEETING_CONCISE_PROMPT
    elif meeting_type == "Management Meeting":
        base = MANAGEMENT_MEETING_DETAILED_PROMPT if detailed else MANAGEMENT_MEETING_CONCISE_PROMPT
    else:
        base = INTERNAL_DISCUSSION_DETAILED_PROMPT if detailed else INTERNAL_DISCUSSION_CONCISE_PROMPT
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
        resp = generate_with_retry(model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS}, stage="Notes")
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
        resp = generate_with_retry(model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS}, stage="Notes")
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
        return generate_with_retry(model, base_prompt.format(notes=notes.strip()), stage="Intelligence").text

    # Multi-chunk: extract in parallel, then synthesise
    status_write(f"Extracting intelligence from {len(chunks)} note sections in parallel…")
    extracts: list[str] = [""] * len(chunks)

    def _extract_one(idx: int, chunk_text: str) -> tuple[int, str]:
        return idx, generate_with_retry(model, base_prompt.format(notes=chunk_text), stage="Intelligence").text

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
    return generate_with_retry(model, synth_prompt, stage="Intelligence synthesis").text


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
    resp = generate_with_retry(model, prompt, stream=True, stage="Summary")
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
    resp = generate_with_retry(model, prompt, stream=True, stage="Summary refinement")
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

# ── Filename + auto-download helpers ──────────────────────────────────────────
# Auto-download mitigates the Streamlit session-loss problem: as soon as a run
# completes, the transcript / notes / intelligence brief / summary are written to
# the user's disk. Even if the session times out or the tab is refreshed, files
# are already saved locally.

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


def filename_for(meeting_name: str, kind: str, ext: str) -> str:
    """Build a filename like '20260628_JPMorgan_expert_call_notes.md'.
    Date is today's date in YYYYMMDD; kind is the document type label."""
    date_str = datetime.now().strftime("%Y%m%d")
    name  = _sanitize_filename_component(meeting_name)
    kind_safe = _sanitize_filename_component(kind, fallback="output")
    ext_safe  = ext.lstrip(".")
    return f"{date_str}_{name}_{kind_safe}.{ext_safe}"


def auto_download_files(files: List[Tuple[str, str, str]]) -> None:
    """Trigger browser downloads for one or more text files via injected JS.

    Args:
        files: list of (filename, content, mime_type) tuples.

    Mechanism: creates an invisible iframe (height=0), inside it a small script
    builds a Blob URL per file, creates an <a download> element, and clicks it.
    Multiple files are staggered by 600ms so most browsers don't block them.

    Browser quirk: first multi-file auto-download per origin prompts the user
    to allow multiple downloads on this site. After that it's silent.
    """
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


def _consume_pending_auto_download():
    """Pop the pending_auto_download flag from session state and trigger downloads.
    Call this at the top of the OUTPUT area on any page — the flag was staged by
    the last successful run so downloads fire exactly once on the next render."""
    pending = st.session_state.pop("pending_auto_download", None)
    if pending:
        auto_download_files(pending)
        st.success(
            "✓ Auto-downloaded to your downloads folder: " +
            " · ".join(f"`{f[0]}`" for f in pending) +
            "  *(first multi-file download per site may prompt your browser to allow it)*"
        )


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
        # New section headers used by the Q&A-format Management & Internal intel prompts.
        # (Old headers above are kept for backward compatibility with any in-session briefs
        # generated before the prompts were rewritten.)
        "MANAGEMENT BACKGROUND":                            "👔",
        "QUESTIONS & UNCERTAINTIES MANAGEMENT RAISED":      "❓",
        "DISCUSSION CONTEXT":                               "📋",
        "QUESTIONS & UNCERTAINTIES THE TEAM RAISED":        "❓",
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


def render_usage_panel():
    """Render a session-to-date usage & cost panel from st.session_state['usage_log'].
    Displays per-stage breakdown (input tokens, output tokens, USD cost) plus a total.
    Renders nothing if no LLM calls have been logged yet."""
    log = st.session_state.get("usage_log", [])
    if not log:
        return

    # Aggregate by stage, summing token counts and cost; track models used per stage
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
            "Approximate cost based on the hardcoded `MODEL_PRICING` table — verify against "
            "your Google Cloud project's billing reports for authoritative numbers. "
            "Audio input is billed at the text-input rate here (slight underestimate). "
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


# ── 13. PAGE: PROCESS ──────────────────────────────────────────────────────────

def page_process():
    api_key_check()

    # If the Transcribe page staged a transcript via the "Use in Process Meeting" button,
    # pre-fill the Paste Text input here. The pop ensures this fires only on the first
    # visit after staging — subsequent reruns don't re-clobber user edits.
    staged = st.session_state.pop("staged_transcript_for_process", None)
    if staged:
        st.session_state["text_input"]   = staged
        st.session_state["input_method"] = "Paste Text"

    st.header("Process Meeting")

    # ── Meeting name (used in auto-downloaded filenames) ──────────────────────
    meeting_name = st.text_input(
        "Meeting name / company (used in auto-downloaded filenames; optional)",
        placeholder="e.g. Hitachi Energy management meeting  /  Dr Patel expert call",
        key="meeting_name",
        help=(
            "Used to name auto-downloaded files: "
            "**YYYYMMDD_<name>_transcript.txt**, **_notes.md/.txt**, **_intelligence.md/.txt**. "
            "If blank, files are named with 'untitled' instead. Special characters "
            "become underscores so the name is filesystem-safe."
        ),
    )

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

    # Concise vs Detailed now available for ALL meeting types — each type has its own
    # Concise and Detailed prompt variant. Default ("Concise") is the first option.
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

                # ── Stage auto-download — fires once on the next render ─────
                # Three artefacts x formats: transcript.txt, notes .md/.txt,
                # intelligence .md/.txt. Saves everything of value to disk
                # so session loss doesn't wipe the run.
                st.session_state["pending_auto_download"] = [
                    (filename_for(meeting_name, "transcript",   "txt"), transcript,   "text/plain"),
                    (filename_for(meeting_name, "notes",        "md"),  notes,        "text/markdown"),
                    (filename_for(meeting_name, "notes",        "txt"), notes,        "text/plain"),
                    (filename_for(meeting_name, "intelligence", "md"),  intelligence, "text/markdown"),
                    (filename_for(meeting_name, "intelligence", "txt"), intelligence, "text/plain"),
                ]

                status.update(label="Done!", state="complete")
                st.write("✓ Ready. Go to the **Summary** tab to generate a summary.")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    if "last_notes" in st.session_state:
        st.divider()

        # Trigger any pending auto-download from the last successful run
        _consume_pending_auto_download()

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

    render_usage_panel()


# ── 14. PAGE: SUMMARY ──────────────────────────────────────────────────────────

def page_summary():
    api_key_check()
    st.header("Summary")

    # ── Meeting name (used in auto-downloaded filenames) ──────────────────────
    # Shared session-state key with Process page — if user already typed a name there,
    # it appears here pre-filled; edits here also persist back.
    meeting_name = st.text_input(
        "Meeting name / company (used in auto-downloaded filenames; optional)",
        placeholder="e.g. Hitachi Energy management meeting",
        key="meeting_name",
        help=(
            "Same key as on the Process Meeting page — filling it in either place is enough. "
            "Used to name auto-downloaded summary files: **YYYYMMDD_<name>_summary.md/.txt**."
        ),
    )

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

            # Stage auto-download for the summary in .md + .txt
            st.session_state["pending_auto_download"] = [
                (filename_for(meeting_name, "summary", "md"),  summary, "text/markdown"),
                (filename_for(meeting_name, "summary", "txt"), summary, "text/plain"),
            ]

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

    # Trigger any pending auto-download from the last successful summary run
    _consume_pending_auto_download()

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
                # Stage auto-download for the refined summary too
                st.session_state["pending_auto_download"] = [
                    (filename_for(meeting_name, "summary", "md"),  revised, "text/markdown"),
                    (filename_for(meeting_name, "summary", "txt"), revised, "text/plain"),
                ]
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

    render_usage_panel()


# ── 15. PAGE: ANALYSE ──────────────────────────────────────────────────────────

def run_analysis(intelligence: str, question: str, model, status_write) -> str:
    """Run a single analysis question against the intelligence brief."""
    status_write("Analysing…")
    prompt = ANALYSIS_PROMPT.format(intelligence=intelligence.strip(), question=question.strip())
    ph = st.empty()
    resp = generate_with_retry(model, prompt, stream=True, stage="Analysis")
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
    # Three modes:
    #   Session  — uses pre-extracted intelligence brief (fastest, no extra step)
    #   Notes    — paste raw notes; answers come directly from the notes
    #   Brief    — paste a previously extracted intelligence brief
    has_session = "last_intelligence" in st.session_state
    source = st.radio(
        "Source",
        ["From last processed session", "Paste notes", "Paste intelligence brief"],
        horizontal=True, key="analyse_source",
    )

    intelligence = ""    # used for session / brief modes
    raw_notes    = ""    # used for notes mode
    meeting_type = "Expert Meeting"
    using_notes  = (source == "Paste notes")

    if source == "From last processed session":
        if not has_session:
            st.info(
                "No processed session yet. Process a transcript on the **Process Meeting** page first, "
                "or paste your notes directly."
            )
            return
        intelligence = st.session_state["last_intelligence"]
        meeting_type = st.session_state.get("last_meeting_type", "Expert Meeting")
        st.caption(f"Using intelligence brief from session — **{meeting_type}**")
        with st.expander("View intelligence brief", expanded=False):
            render_intelligence_panel(intelligence, meeting_type)

    elif source == "Paste notes":
        raw_notes = st.text_area(
            "Paste meeting notes", height=250, key="analyse_notes_input",
            placeholder="Paste your meeting notes here — ask any question about them below."
        )
        st.caption("You can ask factual questions (*'What did the expert say about X?'*) or analytical ones (*'What does the view on X imply?'*).")

    else:  # Paste intelligence brief
        meeting_type = st.selectbox("Meeting type", MEETING_TYPES, key="analyse_meeting_type")
        intelligence = st.text_area(
            "Paste intelligence brief", height=250, key="analyse_intel_input",
            placeholder="Paste the intelligence brief extracted from the Process Meeting page…"
        )

    # The active content for suggestion generation — brief if available, notes otherwise
    suggestion_source = intelligence.strip() or raw_notes.strip()

    st.divider()

    # ── Suggested questions ────────────────────────────────────────────────────
    st.markdown("**Suggested questions for this call**")
    st.caption("Generated from the content of this specific meeting — not generic templates.")

    col_suggest, _ = st.columns([1, 3])
    with col_suggest:
        suggest_clicked = st.button(
            "✦ Suggest questions", key="btn_suggest",
            help="Generates 5 questions specific to this call's content."
        )

    if suggest_clicked:
        if not suggestion_source:
            st.error("Please provide notes or a brief first.")
        else:
            analysis_model = get_model(analysis_model_name)
            with st.spinner("Generating questions for this call…"):
                try:
                    raw = generate_with_retry(
                        analysis_model,
                        QUESTION_SUGGESTION_PROMPT.format(intelligence=suggestion_source),
                        stage="Suggest questions",
                    ).text
                    questions = [
                        re.sub(r'^\s*\d+[\.\)]\s*', '', line).strip()
                        for line in raw.strip().splitlines()
                        if re.match(r'^\s*\d+[\.\)]', line)
                    ]
                    st.session_state["suggested_questions"] = questions[:5]
                except Exception as e:
                    st.error(f"Could not generate suggestions: {e}")

    suggested = st.session_state.get("suggested_questions", [])
    if suggested:
        for q in suggested:
            if st.button(f"↳ {q}", key=f"sq_{hash(q)}", use_container_width=True):
                st.session_state["analysis_question"] = q
                st.rerun()

    # ── Question input ─────────────────────────────────────────────────────────
    st.markdown("**Your question**")
    question = st.text_area(
        "Question", height=100, key="analysis_question",
        placeholder="Click a suggested question above, or type your own…"
    )

    st.divider()
    if st.button("Run Analysis", type="primary", use_container_width=True):
        active_content = raw_notes.strip() if using_notes else intelligence.strip()
        if not active_content:
            st.error("Please paste notes or an intelligence brief.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        analysis_model = get_model(analysis_model_name)
        status_ph = st.empty()

        try:
            # Choose prompt based on source — notes get the direct Q&A prompt,
            # brief/session get the structured inference prompt
            if using_notes:
                prompt = NOTES_QA_PROMPT.format(
                    notes=active_content, question=question.strip()
                )
                status_ph.info("⏳ Searching notes for an answer…")
                ph = st.empty()
                resp = generate_with_retry(analysis_model, prompt, stream=True, stage="Analysis")
                result, _ = stream_and_collect(resp, ph)
                status_ph.empty()
            else:
                result = run_analysis(active_content, question, analysis_model,
                                      lambda msg: status_ph.info(f"⏳ {msg}"))
                status_ph.empty()

            if not result.strip():
                raise ValueError("Model returned an empty response.")

            analyses = st.session_state.setdefault("analysis_history", [])
            analyses.append({
                "question": question.strip(),
                "answer":   result.strip(),
                "mode":     "notes" if using_notes else "brief",
            })

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
        mode_label = "📄 from notes" if entry.get("mode") == "notes" else "★ from brief"
        col_q, col_copy = st.columns([4, 1])
        with col_q:
            st.markdown(f"**Q{entry_num}:** {entry['question']}  `{mode_label}`")
        with col_copy:
            copy_button(entry["answer"], f"Copy A{entry_num}")

        # Amber box — visually distinct from factual notes/summary output
        st.warning(entry["answer"], icon="🔍")

    if len(analyses) > 1:
        if st.button("Clear analysis history", key="clear_analyses"):
            st.session_state.pop("analysis_history", None)
            st.rerun()

    render_usage_panel()


# ── 16. PAGE: TRANSCRIBE ───────────────────────────────────────────────────────
# Audio-only workflow: upload/record audio → raw transcript → optional refined transcript.
# Does NOT run notes generation, intelligence extraction, or summary — for use cases where
# the user just needs a usable transcript to take elsewhere (e.g. paste into another tool,
# share with a colleague, archive for the record).

def page_transcribe():
    api_key_check()
    st.header("Transcribe")
    st.caption(
        "Upload audio and get a usable transcript — without running the full notes pipeline. "
        "Useful when you just need the words for something else."
    )

    with st.sidebar:
        st.markdown("### Model Settings")
        transcription_model_name = st.selectbox(
            "Transcription model", list(MODELS.keys()), index=3,
            key="t_transcription_model", help="Audio-to-text."
        )
        refine_model_name = st.selectbox(
            "Refinement model", list(MODELS.keys()), index=1,
            key="t_refine_model", help="Transcript clean-up pass."
        )
        refine_enabled = st.toggle(
            "Enable refinement pass", value=True, key="t_refine_toggle",
            help="Fixes grammar, labels speakers where possible, translates Hindi/Hinglish into English."
        )

    with st.expander("Context (optional — improves transcription accuracy on names/terms)", expanded=False):
        speakers = st.text_input(
            "Speaker names (comma-separated)", key="t_speakers",
            placeholder="e.g. John Smith, Dr. Patel, Priya Krishnan"
        )
        extra_context = st.text_area(
            "Background / domain terms", height=80, key="t_extra_context",
            placeholder="e.g. Call on Indian cement sector. Key companies: UltraTech, Shree Cement."
        )

    extra_context_combined = "\n".join(filter(None, [
        f"Speakers: {speakers}" if speakers.strip() else "",
        extra_context.strip(),
    ]))

    st.divider()
    input_method = st.radio(
        "Audio source",
        ["Upload File", "Record Audio"],
        horizontal=True, key="t_input_method",
    )

    audio_bytes: Optional[bytes] = None
    audio_ext: str = ".audio"
    source_filename: str = "transcript"

    if input_method == "Upload File":
        uploaded = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac", "mp4", "mov"],
            key="t_uploaded_file",
        )
        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            if size_mb > MAX_AUDIO_MB:
                st.error(f"Audio file too large ({size_mb:.1f} MB). Limit: {MAX_AUDIO_MB} MB.")
            else:
                audio_bytes = uploaded.getvalue()
                audio_ext = os.path.splitext(uploaded.name)[1].lower()
                source_filename = os.path.splitext(uploaded.name)[0] or "transcript"
                st.info(f"Audio loaded: **{uploaded.name}** ({size_mb:.1f} MB)")
    else:
        st.caption("Click the microphone to start recording. Click again to stop.")
        recording = st.audio_input("Record a voice note", key="t_audio_recording")
        if recording:
            audio_bytes = recording.getvalue()
            audio_ext = ".webm"
            source_filename = "recording"
            st.success("Recording captured. Click **Transcribe** when ready.")

    st.divider()
    if st.button("Transcribe", type="primary", use_container_width=True):
        if not audio_bytes:
            st.error("Please upload or record an audio file first.")
            st.stop()

        transcr_model = get_model(transcription_model_name)
        refine_model  = get_model(refine_model_name)

        with st.status("Transcribing…", expanded=True) as status:
            try:
                raw_transcript = transcribe_audio(
                    audio_bytes, transcr_model, st.write,
                    context=extra_context_combined, file_ext=audio_ext,
                )
                if not raw_transcript or not raw_transcript.strip():
                    raise ValueError("Transcription returned empty output.")
                st.write(f"✓ Raw transcription complete: **{len(raw_transcript.split()):,} words**")

                refined_transcript = ""
                if refine_enabled:
                    # Empty meeting_type → refine_transcript skips the meeting-type-specific
                    # instruction and just does grammar / speaker-labelling / translation.
                    refined_transcript = refine_transcript(
                        raw_transcript, "", speakers, refine_model, st.write,
                    )
                    st.write(f"✓ Refined transcript complete: **{len(refined_transcript.split()):,} words**")

                st.session_state["t_raw_transcript"]     = raw_transcript
                st.session_state["t_refined_transcript"] = refined_transcript
                st.session_state["t_source_filename"]    = source_filename

                status.update(label="Done!", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    if "t_raw_transcript" in st.session_state:
        st.divider()
        raw      = st.session_state["t_raw_transcript"]
        refined  = st.session_state.get("t_refined_transcript", "")
        filename = st.session_state.get("t_source_filename", "transcript")

        if refined:
            # Refined first (most users will want the clean version)
            tab_refined, tab_raw = st.tabs(["✨ Refined Transcript", "📄 Raw Transcript"])

            with tab_refined:
                col_title, col_copy, col_dl = st.columns([2, 1, 1])
                with col_title:
                    st.subheader(f"Refined Transcript  ({len(refined.split()):,} words)")
                with col_copy:
                    copy_button(refined, "Copy")
                with col_dl:
                    st.download_button(
                        "Download .txt", data=refined,
                        file_name=f"{filename}_refined.txt", mime="text/plain",
                        use_container_width=True, key="dl_refined",
                    )
                st.markdown(refined)
                if st.button("→ Use this refined transcript in Process Meeting",
                             key="use_refined_in_process", use_container_width=True):
                    st.session_state["staged_transcript_for_process"] = refined
                    st.switch_page(PAGE_PROCESS)

            with tab_raw:
                col_title, col_copy, col_dl = st.columns([2, 1, 1])
                with col_title:
                    st.subheader(f"Raw Transcript  ({len(raw.split()):,} words)")
                with col_copy:
                    copy_button(raw, "Copy")
                with col_dl:
                    st.download_button(
                        "Download .txt", data=raw,
                        file_name=f"{filename}_raw.txt", mime="text/plain",
                        use_container_width=True, key="dl_raw",
                    )
                st.markdown(raw)
                if st.button("→ Use this raw transcript in Process Meeting",
                             key="use_raw_in_process", use_container_width=True):
                    st.session_state["staged_transcript_for_process"] = raw
                    st.switch_page(PAGE_PROCESS)
        else:
            col_title, col_copy, col_dl = st.columns([2, 1, 1])
            with col_title:
                st.subheader(f"Transcript  ({len(raw.split()):,} words)")
            with col_copy:
                copy_button(raw, "Copy")
            with col_dl:
                st.download_button(
                    "Download .txt", data=raw,
                    file_name=f"{filename}.txt", mime="text/plain",
                    use_container_width=True, key="dl_raw_only",
                )
            st.markdown(raw)
            if st.button("→ Use this transcript in Process Meeting",
                         key="use_raw_only_in_process", use_container_width=True):
                st.session_state["staged_transcript_for_process"] = raw
                st.switch_page(PAGE_PROCESS)

    render_usage_panel()


# ── 17. MAIN ───────────────────────────────────────────────────────────────────

# Page objects are defined at module scope so st.switch_page() can reference them
# (used by the "Use in Process Meeting →" button on the Transcribe page output).
PAGE_PROCESS    = st.Page(page_process,    title="Process Meeting", icon=":material/edit_note:")
PAGE_SUMMARY    = st.Page(page_summary,    title="Summary",         icon=":material/summarize:")
PAGE_ANALYSE    = st.Page(page_analyse,    title="Analyse",         icon=":material/psychology:")
PAGE_TRANSCRIBE = st.Page(page_transcribe, title="Transcribe",      icon=":material/mic:")


def main():
    st.set_page_config(page_title="SynthNotes Pro", layout="wide", page_icon="★")
    nav = st.navigation([PAGE_PROCESS, PAGE_SUMMARY, PAGE_ANALYSE, PAGE_TRANSCRIBE])
    nav.run()


if __name__ == "__main__":
    main()
