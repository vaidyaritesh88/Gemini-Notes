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
MAX_OUTPUT_TOKENS  = 65536
MAX_PDF_MB         = 25
MAX_AUDIO_MB       = 200

MODELS = {
    "Gemini 2.5 Flash (Fast)":      "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite (Cheap)":"gemini-2.5-flash-lite",
    "Gemini 2.5 Pro (Best)":        "gemini-2.5-pro",
    "Gemini 3.0 Flash":             "gemini-3-flash-preview",
    "Gemini 2.0 Flash":             "gemini-2.0-flash-lite",
    "Gemini 1.5 Flash":             "gemini-1.5-flash",
}

MEETING_TYPES = ["Expert Meeting", "Management Meeting", "Internal Discussion"]

SUMMARY_PRESETS = {
    "Brief (~100 words)":    100,
    "Short (~300 words)":    300,
    "Medium (~500 words)":   500,
    "Detailed (~750 words)": 750,
    "Custom":                None,
}

REFINEMENT_INSTRUCTIONS = {
    "Expert Meeting":      "Pay special attention to industry jargon, technical terms, company names, and domain-specific terminology. Preserve all proper nouns exactly.",
    "Management Meeting":  "Pay special attention to names of attendees, action item owners, project names, deadlines, and organizational terminology.",
    "Internal Discussion": "Pay special attention to participant names, project/product names, technical terms, and any referenced documents or systems.",
}

# Path to the JSON file that persists saved summary prompts across sessions.
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_prompts.json")


# ── 2. PROMPTS ─────────────────────────────────────────────────────────────────

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

SUMMARY_PROMPT = """You are generating a structured summary of meeting notes for a busy professional.

Target length: approximately {word_count} words — stay within 10% of this target.

### STRUCTURE:
1. **Meeting Overview** (1–2 sentences): Briefly state what kind of meeting this was and the main subject discussed.
2. **Main Themes** (the bulk of your summary): For each major topic discussed, use a **bold heading** for the theme, then bullet points (- ) for the key findings, conclusions, and data points under that theme.
3. **Key Takeaways**: End with a **Key Takeaways** section containing 3–5 concise bullets — the most important things to remember from this meeting.

### RULES:
- Hit the target word count: not significantly shorter, not significantly longer.
- Keep all critical data: numbers, percentages, names of people/companies, specific claims, dates.
- Do NOT add information that is not present in the notes.
- Language: direct and professional. Avoid filler phrases like "It was noted that..." or "The discussion covered...".
- Each bullet under a heading should make one clear, complete, self-contained point.
- Sub-bullets (  - indented) are allowed when a point has meaningful supporting detail.
{focus_block}
---
MEETING NOTES:
{notes}
"""


# ── 3. SAVED PROMPTS HELPERS ──────────────────────────────────────────────────

def load_saved_prompts() -> dict:
    """Load saved summary prompts from disk. Returns {'default': str|None, 'prompts': {name: text}}."""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure expected keys exist even if file is from an older version
                data.setdefault("default", None)
                data.setdefault("prompts", {})
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"default": None, "prompts": {}}


def write_saved_prompts(data: dict):
    """Persist saved prompts to disk."""
    try:
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        st.error(f"Could not save prompts file: {e}")


# ── 4. CORE PROCESSING FUNCTIONS ──────────────────────────────────────────────

def get_model(display_name: str) -> genai.GenerativeModel:
    """Return a cached GenerativeModel instance."""
    cache = st.session_state.setdefault("_model_cache", {})
    model_id = MODELS.get(display_name, "gemini-2.5-flash")
    if model_id not in cache:
        cache[model_id] = genai.GenerativeModel(model_id)
    return cache[model_id]


def generate_with_retry(model, prompt, max_retries: int = 3, stream: bool = False, generation_config=None):
    """Call generate_content with exponential backoff for transient API errors."""
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
    """Consume a streaming response, optionally showing live word count. Returns (text, tokens)."""
    full_text, counter = "", 0
    for chunk in response:
        if chunk.parts:
            full_text += chunk.text
            counter += 1
            if placeholder and counter % 5 == 0:
                placeholder.caption(f"Streaming... {len(full_text.split()):,} words generated")
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
    """Split text into overlapping word-based chunks."""
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
    """Build a lightweight context summary from already-generated notes to pass to the next chunk."""
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
    """Deterministic post-processing: remove LLM meta-commentary artifacts and duplicate headings."""
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
                continue  # skip duplicate consecutive heading
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


def transcribe_audio(audio_bytes: bytes, model, status_write, context: str = "", file_ext: str = ".audio") -> str:
    """Split audio into 5-min WAV chunks using FFmpeg, upload to Gemini Files API, transcribe.

    Uses FFmpeg (installed via packages.txt) instead of pydub, so it works on any Python version
    and handles every audio format FFmpeg supports (WAV, MP3, M4A, OGG, FLAC, WebM, etc.).

    context:  optional domain context to improve transcription accuracy
    file_ext: original file extension hint (e.g. ".mp3", ".webm") — helps FFmpeg probe the format
    """
    transcription_instruction = "Transcribe this audio accurately, preserving the speaker's words as closely as possible."
    if context.strip():
        transcription_instruction += (
            f"\n\nContext to help with accurate transcription "
            f"(use this to correctly identify domain-specific terms, names, and abbreviations):\n{context.strip()}"
        )

    local_paths, cloud_names, transcripts = [], [], []

    # Write the full audio bytes to a temp file so FFmpeg can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
        f.write(audio_bytes)
        input_path = f.name
    local_paths.append(input_path)

    try:
        # Use FFmpeg to split into 5-minute mono 16kHz WAV chunks.
        # -f segment + -segment_time 300 splits at silence-friendly boundaries.
        chunk_pattern = input_path + "_chunk_%03d.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-f", "segment", "-segment_time", "300",
                "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                chunk_pattern,
            ],
            capture_output=True, timeout=300,
        )

        chunk_files = sorted(glob.glob(input_path + "_chunk_*.wav"))
        local_paths.extend(chunk_files)

        # If FFmpeg didn't produce chunks (very short recording), convert the whole file instead
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
            status_write(f"Transcribing audio chunk {i+1} of {len(chunk_files)}...")
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


def refine_transcript(raw: str, meeting_type: str, speakers: str, model, status_write) -> str:
    """
    Clean up the raw transcript: fix grammar/spelling, label speakers, translate to English if needed.
    Long transcripts are chunked and refined IN PARALLEL (chunks are independent at this stage).
    """
    lang_instr = (
        "IMPORTANT: Your entire output MUST be in English. "
        "If the transcript contains Hindi, Hinglish, or any other non-English language, "
        "translate all content into clear, natural English while preserving the original meaning."
    )
    extra = REFINEMENT_INSTRUCTIONS.get(meeting_type, "")
    speaker_info = f"Participants: {speakers}." if speakers.strip() else ""

    words = raw.split()

    if len(words) <= CHUNK_WORD_SIZE:
        status_write("Refining transcript (single chunk)...")
        prompt = (
            f"Refine the following transcript. Correct spelling, grammar, and punctuation. "
            f"Label speakers clearly if possible. {speaker_info} {extra}\n{lang_instr}\n\n"
            f"TRANSCRIPT:\n{raw}"
        )
        return generate_with_retry(model, prompt).text

    chunks = create_chunks_with_overlap(raw, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
    status_write(f"Refining transcript ({len(chunks)} chunks in parallel)...")

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
                f"---\nCONTEXT (previous chunk tail — for continuity only):\n...{ctx_tail}\n---\n"
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
            status_write(f"Refinement: {done}/{len(chunks)} chunks done...")

    return "\n\n".join(r for r in results if r)


def generate_notes(transcript: str, meeting_type: str, detail_level: str, extra_context: str, model, status_write) -> str:
    """
    Generate structured notes from the (refined) transcript.
    Long transcripts are chunked and processed SERIALLY — each chunk's prompt
    includes a context summary built from the previous chunk's OUTPUT notes.
    """
    base = _build_base_prompt(meeting_type, detail_level, extra_context)
    words = transcript.split()

    if len(words) <= CHUNK_WORD_SIZE:
        status_write("Generating notes (single chunk)...")
        prompt = f"{base}\n\n**MEETING TRANSCRIPT:**\n{transcript}"
        ph = st.empty()
        resp = generate_with_retry(model, prompt, stream=True, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS})
        notes, _ = stream_and_collect(resp, ph)
        return notes

    chunks = create_chunks_with_overlap(transcript, CHUNK_WORD_SIZE, CHUNK_WORD_OVERLAP)
    all_chunk_notes = []
    context_package = ""

    for i, chunk in enumerate(chunks):
        status_write(f"Generating notes: chunk {i+1} of {len(chunks)}...")
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
        # Context for next chunk: lightweight summary of output so far (not raw transcript)
        context_package = create_context_from_notes("\n\n".join(all_chunk_notes), i + 1)

    # Stitch: find the last bold heading in chunk N-1, skip overlap in chunk N
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


# ── 4. SHARED UI HELPERS ───────────────────────────────────────────────────────

def copy_button(text: str, label: str = "Copy"):
    """Render a theme-aware clipboard copy button via a small HTML/JS component."""
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


# ── 5. PAGE: GENERATE NOTES ────────────────────────────────────────────────────

def page_generate():
    api_key_check()
    st.header("Generate Notes")

    # Sidebar: model selection
    with st.sidebar:
        st.markdown("### Model Settings")
        notes_model_name = st.selectbox(
            "Notes model", list(MODELS.keys()), index=2,
            key="notes_model", help="Used for the main note generation pass."
        )
        refine_model_name = st.selectbox(
            "Refinement model", list(MODELS.keys()), index=1,
            key="refine_model", help="Used to clean up the transcript before note generation."
        )
        transcription_model_name = st.selectbox(
            "Transcription model", list(MODELS.keys()), index=3,
            key="transcription_model", help="Used to convert audio to text."
        )
        refine_enabled = st.toggle(
            "Enable refinement pass", value=True, key="refine_toggle",
            help="Cleans up grammar, labels speakers, and translates non-English before generating notes."
        )

    # Meeting type
    meeting_type = st.selectbox("Meeting type", MEETING_TYPES, key="meeting_type")

    detail_level = "Concise"
    if meeting_type == "Expert Meeting":
        detail_level = st.radio(
            "Note style", ["Concise", "Detailed"], horizontal=True, key="detail_level",
            help="**Concise**: information-dense, no filler.  **Detailed**: maximum verbosity, zero omission — use for high-value calls."
        )

    # Optional context
    with st.expander("Context & additional instructions", expanded=False):
        st.caption(
            "This context is passed into **both** the transcription step (to help identify "
            "domain-specific terms and names in audio) and the note generation step (to guide "
            "how notes are structured or what to focus on)."
        )
        speakers = st.text_input("Speaker names (comma-separated)", key="speakers",
                                  placeholder="e.g. John Smith (analyst), Dr. Patel (expert)")
        extra_context = st.text_area("Background / additional instructions", height=100, key="extra_context",
                                      placeholder="e.g. This is an expert call on the Indian cement sector. Key companies: UltraTech, Shree Cement. Focus on pricing dynamics and demand in Tier-2 cities.")

    extra_context_combined = "\n".join(filter(None, [
        f"Speakers: {speakers}" if speakers.strip() else "",
        extra_context.strip(),
    ]))

    # Input
    st.divider()
    input_method = st.radio(
        "Input method",
        ["Paste Text", "Upload File (PDF / Audio)", "Record Audio"],
        horizontal=True,
        key="input_method",
    )

    raw_text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    audio_ext: str = ".audio"
    is_audio = False

    if input_method == "Paste Text":
        raw_text = st.text_area("Paste transcript here", height=300, key="text_input",
                                 placeholder="Paste your meeting transcript...")

    elif input_method == "Upload File (PDF / Audio)":
        uploaded = st.file_uploader(
            "Upload a PDF, TXT, or audio file",
            type=["pdf", "txt", "md", "wav", "mp3", "m4a", "ogg", "flac"],
            key="uploaded_file",
        )
        if uploaded:
            ext = os.path.splitext(uploaded.name)[1].lower()
            size_mb = uploaded.size / (1024 * 1024)
            if ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
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

    else:  # Record Audio
        st.caption("Click the microphone to start recording. Click again to stop. Your recording is processed locally — nothing is stored.")
        recording = st.audio_input("Record a voice note", key="audio_recording")
        if recording:
            audio_bytes = recording.getvalue()
            audio_ext = ".webm"   # browsers record in WebM/Opus
            is_audio = True
            st.success("Recording captured. Click **Generate Notes** when ready.")

    # Generate button
    st.divider()
    if st.button("Generate Notes", type="primary", use_container_width=True):
        has_input = is_audio or (raw_text and raw_text.strip())
        if not has_input:
            st.error("Please provide a transcript (paste text, upload a file, or upload audio).")
            st.stop()

        notes_model   = get_model(notes_model_name)
        refine_model  = get_model(refine_model_name)
        transcr_model = get_model(transcription_model_name)

        with st.status("Processing...", expanded=True) as status:
            try:
                # Step 1: audio → raw transcript
                if is_audio:
                    transcript = transcribe_audio(audio_bytes, transcr_model, st.write, context=extra_context_combined, file_ext=audio_ext)
                    st.write(f"✓ Transcription complete: **{len(transcript.split()):,} words**")
                else:
                    transcript = re.sub(r"\n{3,}", "\n\n", raw_text.strip())

                # Step 2: refine
                if refine_enabled:
                    transcript = refine_transcript(transcript, meeting_type, speakers, refine_model, st.write)
                    st.write(f"✓ Refinement complete: **{len(transcript.split()):,} words**")

                # Step 3: generate notes
                notes = generate_notes(transcript, meeting_type, detail_level, extra_context_combined, notes_model, st.write)

                if not notes or not notes.strip():
                    raise ValueError("The model returned empty notes. Please try again.")

                st.session_state["last_notes"] = notes
                st.session_state["last_transcript"] = transcript
                status.update(label="Done!", state="complete")
                st.write("✓ Notes generated and saved to session.")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"**Error:** {e}")

    # Output
    if "last_notes" in st.session_state:
        st.divider()
        notes = st.session_state["last_notes"]
        col_title, col_copy = st.columns([3, 1])
        with col_title:
            st.subheader("Generated Notes")
        with col_copy:
            copy_button(notes, "Copy Notes")
        st.markdown(notes)
        st.info("Go to the **Summary** tab to generate a summary from these notes.", icon="ℹ️")


# ── 6. PAGE: SUMMARY ───────────────────────────────────────────────────────────

def page_summary():
    api_key_check()
    st.header("Summary")

    # Sidebar: model
    with st.sidebar:
        st.markdown("### Model Settings")
        summary_model_name = st.selectbox(
            "Summary model", list(MODELS.keys()), index=0,
            key="summary_model",
        )

    # ── Notes input ────────────────────────────────────────────────────────────
    auto_fill = st.session_state.get("last_notes", "")
    notes_input = st.text_area(
        "Paste notes to summarise",
        value=auto_fill,
        height=250,
        key="summary_notes_input",
        placeholder="Paste your meeting notes here, or generate notes on the previous tab and they will appear automatically...",
    )
    if auto_fill and notes_input == auto_fill:
        st.caption(f"Auto-filled from last generated notes ({len(auto_fill.split()):,} words)")

    # ── Length ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Summary length**")
    length_label = st.radio(
        "length_radio", list(SUMMARY_PRESETS.keys()),
        horizontal=True, key="summary_length", label_visibility="collapsed"
    )
    word_count = SUMMARY_PRESETS[length_label]
    if word_count is None:
        word_count = st.number_input(
            "Custom word count", min_value=50, max_value=2000, value=400, step=50, key="custom_wc"
        )

    # ── Focus instructions + saved prompts ─────────────────────────────────────
    st.divider()

    saved_data = load_saved_prompts()
    saved_names = list(saved_data["prompts"].keys())
    default_name = saved_data.get("default")

    # Pre-select the saved default on first load (only if session key not yet set)
    if "summary_selected_prompt" not in st.session_state:
        st.session_state["summary_selected_prompt"] = default_name if default_name in saved_names else "None"

    with st.expander("Focus instructions & saved prompts", expanded=True):

        # ── Row 1: dropdown to pick a saved prompt ──
        dropdown_options = ["None"] + saved_names
        selected_name = st.selectbox(
            "Saved focus prompts",
            options=dropdown_options,
            index=dropdown_options.index(st.session_state["summary_selected_prompt"])
                  if st.session_state["summary_selected_prompt"] in dropdown_options else 0,
            key="summary_selected_prompt",
            help="Select a saved prompt to load it into the text area below.",
        )

        # Resolve the text to show: saved prompt text if one is selected, else empty
        prompt_text_from_saved = saved_data["prompts"].get(selected_name, "") if selected_name != "None" else ""

        # ── Row 2: focus instructions text area ──
        st.caption(
            "Tell the model what to focus on or emphasise in the summary. "
            "This is injected into the prompt in addition to the standard structure."
        )
        # Use the saved prompt text as the value only when a saved prompt is selected;
        # otherwise leave the area blank (or preserve what the user typed).
        if "focus_text_override" not in st.session_state:
            st.session_state["focus_text_override"] = prompt_text_from_saved

        # When the dropdown changes, sync the text area
        if prompt_text_from_saved != st.session_state.get("_last_loaded_prompt", ""):
            st.session_state["focus_text_override"] = prompt_text_from_saved
            st.session_state["_last_loaded_prompt"] = prompt_text_from_saved

        focus_instructions = st.text_area(
            "Focus instructions",
            value=st.session_state["focus_text_override"],
            height=120,
            key="focus_instructions_input",
            label_visibility="collapsed",
            placeholder=(
                "e.g. Focus on management's commentary about margins and pricing. "
                "Highlight any guidance given for the next quarter. "
                "Call out any risks or uncertainties mentioned."
            ),
        )

        # ── Row 3: save / manage controls ──
        st.markdown("**Save this as a prompt**")
        col_name, col_save = st.columns([3, 1])
        with col_name:
            new_prompt_name = st.text_input(
                "Prompt name", key="new_prompt_name", label_visibility="collapsed",
                placeholder="e.g. Margin & Pricing Focus"
            )
        with col_save:
            if st.button("Save", use_container_width=True):
                name = new_prompt_name.strip()
                text = focus_instructions.strip()
                if not name:
                    st.error("Enter a name before saving.")
                elif not text:
                    st.error("The focus instructions are empty — nothing to save.")
                else:
                    saved_data["prompts"][name] = text
                    write_saved_prompts(saved_data)
                    st.session_state["summary_selected_prompt"] = name
                    st.success(f'Saved as **"{name}"**.')
                    st.rerun()

        # Manage existing saved prompts
        if saved_names:
            st.markdown("**Manage saved prompts**")
            col_setdef, col_del = st.columns(2)
            with col_setdef:
                set_def_options = ["— pick one —"] + saved_names
                set_def_pick = st.selectbox(
                    "Set as default", set_def_options,
                    key="set_default_pick", label_visibility="visible",
                    help="The default prompt is auto-loaded every time you open this tab.",
                )
                if st.button("Set as default", use_container_width=True):
                    if set_def_pick == "— pick one —":
                        st.error("Select a prompt first.")
                    else:
                        saved_data["default"] = set_def_pick
                        write_saved_prompts(saved_data)
                        st.success(f'**"{set_def_pick}"** is now the default.')
                        st.rerun()
            with col_del:
                del_options = ["— pick one —"] + saved_names
                del_pick = st.selectbox(
                    "Delete a prompt", del_options,
                    key="delete_pick", label_visibility="visible",
                )
                if st.button("Delete", use_container_width=True):
                    if del_pick == "— pick one —":
                        st.error("Select a prompt to delete.")
                    else:
                        del saved_data["prompts"][del_pick]
                        if saved_data["default"] == del_pick:
                            saved_data["default"] = None
                        write_saved_prompts(saved_data)
                        if st.session_state.get("summary_selected_prompt") == del_pick:
                            st.session_state["summary_selected_prompt"] = "None"
                        st.success(f'**"{del_pick}"** deleted.')
                        st.rerun()

            if default_name:
                st.caption(f"Current default: **{default_name}**")

    # ── Generate ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("Generate Summary", type="primary", use_container_width=True):
        if not notes_input.strip():
            st.error("Please paste notes to summarise.")
            st.stop()

        # Build the optional focus block to inject into the prompt
        focus_text = focus_instructions.strip()
        if focus_text:
            focus_block = (
                "\n### FOCUS INSTRUCTIONS:\n"
                "In addition to the standard structure, pay particular attention to the following "
                "and ensure these aspects are clearly covered in your summary:\n"
                f"{focus_text}\n"
            )
        else:
            focus_block = ""

        model = get_model(summary_model_name)
        prompt = SUMMARY_PROMPT.format(
            word_count=word_count,
            focus_block=focus_block,
            notes=notes_input.strip(),
        )

        with st.spinner(f"Generating ~{word_count}-word summary..."):
            try:
                ph = st.empty()
                resp = generate_with_retry(model, prompt, stream=True)
                summary, _ = stream_and_collect(resp, ph)
                if not summary.strip():
                    raise ValueError("Model returned an empty response.")
                st.session_state["last_summary"] = summary
            except Exception as e:
                st.error(f"**Error:** {e}")

    # ── Output ─────────────────────────────────────────────────────────────────
    if "last_summary" in st.session_state:
        st.divider()
        summary = st.session_state["last_summary"]
        actual_words = len(summary.split())
        col_title, col_copy = st.columns([3, 1])
        with col_title:
            st.subheader(f"Summary ({actual_words:,} words)")
        with col_copy:
            copy_button(summary, "Copy Summary")
        st.markdown(summary)


# ── 7. MAIN ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SynthNotes Lite", layout="wide", page_icon="📝")

    nav = st.navigation([
        st.Page(page_generate, title="Generate Notes", icon=":material/edit_note:"),
        st.Page(page_summary,  title="Summary",        icon=":material/summarize:"),
    ])
    nav.run()


if __name__ == "__main__":
    main()
