# SynthNotes Pro · ASR test — working notes

**This is a fork of `SynthNotes-Pro`, created Sep 2026 to trial Google's dedicated
speech-to-text model without touching the production app.** The two are identical except
for the transcription options described under "The ASR variant" below. `SynthNotes-Pro`
remains the production app at <https://pro-note-maker.streamlit.app/> — do not assume a
change made here has been made there, or vice versa.

Streamlit app that turns a meeting recording into structured notes, then into an
intelligence brief, a summary, and an analysis Q&A. Auto-deployed from `main` of
`vaidyaritesh88/Gemini-Notes`. Single file: `app.py`, ~2,600 lines.

Owner is an Indian equity analyst. Inputs are expert-network calls, management meetings,
and internal research discussions, often in mixed Hindi/English.

---

## Pipeline

```
AUDIO file / live voice note / PDF / TXT
   │
   ▼
[1] TRANSCRIPTION  ── gemini-3.7-flash ── audio inputs only
    ffprobe duration → ffmpeg cuts 300s windows advancing 280s (20s OVERLAP),
    16 kHz mono WAV → Files API upload → transcribe each segment
    → segment N+1 receives the last 150 words of segment N and resumes from there
   │
   ▼
[2] REFINEMENT  ── gemini-3.7-flash ── OPTIONAL, UI toggle
    ≤4,000 words: one call │ longer: 4,000-word chunks, 400-word overlap, 3 in parallel
    corrects language, labels speakers, translates Hindi/Hinglish,
    deletes only verbal noise
    → shrink guard warns if output falls below 80% of input
   │
   ▼
[3] NOTES  ── gemini-3.1-pro-preview
    prompt = meeting-type prompt + WRITING_DISCIPLINE + user context box
    ≤4,000 words: one call │ longer: PROMPT_INITIAL then PROMPT_CONTINUATION
    with a rolling context package; heading-level dedup on merge
   │
   ▼
[4] INTELLIGENCE  ── gemini-3.7-flash
    notes chunked 4,000/400 → per-chunk extraction → INTEL_SYNTHESIS if >1 chunk
   │
   ├──► SUMMARY page   ── gemini-3.1-pro-preview  (~150w brief + summary at chosen length)
   └──► ANALYSE page   ── gemini-3.1-pro-preview  (Q&A over notes, suggested questions)

TRANSCRIBE page runs steps [1]–[2] only.
```

Four pages via `st.navigation`: `PAGE_PROCESS`, `PAGE_SUMMARY`, `PAGE_ANALYSE`,
`PAGE_TRANSCRIBE`.

---

## Model defaults

Set as `DEFAULT_*_MODEL` constants; selectboxes resolve them through
`default_model_index()` so reordering `MODELS` cannot silently repoint a default.

| Step | Model | Why |
|---|---|---|
| Transcription | `gemini-3.7-flash` | Stable, audio-native. `gemini-3.5-transcribe` selectable — see The ASR variant |
| Refinement | `gemini-3.7-flash` | Rewrites the whole transcript — never put a weak model here |
| Notes | `gemini-3.1-pro-preview` | Best instruction-following on a long rule-heavy prompt |
| Intelligence | `gemini-3.7-flash` | Classification over already-dense notes |
| Summary | `gemini-3.1-pro-preview` | Prose quality |
| Analysis | `gemini-3.1-pro-preview` | Reasoning over notes |

Roughly **$0.45 per one-hour meeting** end to end. Nothing in this pipeline is expensive
enough to justify trading quality for cost — do not "optimise" by downgrading a model.

Dropdown carries live models only. Removed Aug 2026: `gemini-2.0-flash-lite` and
`gemini-1.5-flash` (shut down by Google, would 404), `gemini-3-flash-preview`
(superseded), `gemini-3.5-flash` ($1.50/$9.00 for less quality than 3.7 Flash's
$0.75/$3.75).

---

## The ASR variant

`gemini-3.5-transcribe` (GA 26 Aug 2026, replaced Chirp 3) is offered as a transcription
option. **The default is still `gemini-3.7-flash`** — this fork exists to compare the two
on real recordings, not to switch.

It is not a drop-in model swap. Three things differ:

- **Different API.** `client.interactions.create(model=..., input=[{type,uri,mime_type}],
  generation_config={"transcription_config": ...})`, returning `.output_text`. Not
  `generate_content`. `transcribe_audio()` branches on `is_transcription_only_model()`.
- **It takes no instructions.** The LLM path stitches segment seams by handing the model
  the previous transcript's tail and asking it to resume. An ASR model cannot do that, so
  seams are joined afterwards by `_join_on_overlap()`, which matches the longest run of
  words ending one segment against the start of the next and trims it. It normalises away
  punctuation and speaker labels (diarization numbers speakers per request, so the same
  person can be "Speaker 1" then "Speaker 2"), and requires a 5-word match. **No match
  means no cut** — a duplicated sentence is a blemish, a dropped one is a defect.
- **Different limits.** 1 hour per request, 30 minutes with diarization or timestamps. So
  the ASR path uses 25-minute windows (`ASR_SEGMENT_SECONDS`) against the LLM path's five,
  taking an hour of audio from 11 seams to 2. `_segment_audio()` takes the window and
  overlap as arguments for this reason.

Configured for **verbatim mode with speaker diarization**. Verbatim rather than smart mode
because the refinement stage already strips disfluencies and translates — smart mode would
overlap it and make the two paths non-comparable. The context box is mined for
**custom vocabulary** by `_custom_vocabulary_from_context()`: comma/newline-separated
entries of five words or fewer, capped at 100.

Known gaps: speaker labels are numbered per request, so identities do not carry across
segments; the model does not translate, so Hindi/Hinglish still needs the refinement pass;
and `TRANSCRIPTION_ONLY_MODELS` must never leak into `MODELS`, or a dedicated ASR model
becomes selectable for notes or analysis, where it cannot work.

Cost, per hour of audio: `gemini-3.5-transcribe` $0.30 ($0.003/min in + $0.002/min out)
versus roughly $0.13 on 3.7 Flash today. Near parity once the 3.7 Flash promo ends
31 Dec 2026.

---

## The prompt design principle

**Read this before touching any note prompt.**

There is ONE canonical note prompt per meeting type — `EXPERT_MEETING_PROMPT`,
`MANAGEMENT_MEETING_PROMPT`, `INTERNAL_DISCUSSION_PROMPT` — plus a shared
`WRITING_DISCIPLINE` block appended to all three by `_build_base_prompt()`. There is
deliberately no concise/detailed switch and no verbosity setting.

The invariant that makes it work:

> **Coverage rules never mention length. Length rules never authorise dropping content.**

- Completeness lives in ZERO SKIPPING, PRIORITY #1–#4, and the tangent rule. None of
  these say anything about how long the output should be.
- Length lives entirely in `WRITING_DISCIPLINE` — one idea per bullet, no throat-clearing
  openers, no meta-commentary, no closing summary. None of these authorise omission.
- Where the two could collide — repetition — it is adjudicated explicitly: repeated
  *wording* is padding and may be tightened; a repeated *point* is signal and must appear
  every time it was made.

If you add a rule, work out which of the two categories it belongs to. A rule that
straddles both is a bug — that is precisely how the old Concise variant became lossy.

Two rules exist because a specific thing went wrong; do not "simplify" them away:

- **Tangents.** "Material that reads as a digression is frequently the most valuable
  content in the call... **You are not qualified to judge it off-topic.**" Anecdotes that
  look like asides are the whole reason for taking notes on expert calls.
- **Anti-tapering** (`PROMPT_CONTINUATION` item 6). Later chunks get under-processed in
  map-reduce. Stated as a *completeness* standard, never a word count — a length quota is
  satisfiable by padding.

Same reasoning governs `REFINEMENT_FIDELITY_RULE`: refinement may delete verbal noise
('um', stutters, false starts) and nothing else. *"Deleting a stutter is fine; deleting a
restated argument is not."* Refinement rewrites the entire transcript, so anything it
drops is invisible downstream — the notes stage never sees the original. Hence
`_check_refinement_shrink()`, which warns below `REFINEMENT_SHRINK_THRESHOLD` (0.80;
stripping filler legitimately costs 10–15%).

---

## Code map

| Area | Symbols |
|---|---|
| SDK adapter | `_ModelHandle`, `_StreamHandle`, `get_model()` |
| Retry / streaming | `generate_with_retry()`, `stream_and_collect()` |
| Cost tracking | `_record_usage()`, `compute_cost()`, `MODEL_PRICING`, `render_usage_panel()` |
| Audio | `_audio_duration_seconds()`, `_segment_audio()`, `transcribe_audio()` |
| Refinement | `REFINEMENT_FIDELITY_RULE`, `refine_transcript()`, `_check_refinement_shrink()` |
| Notes | three `*_PROMPT` constants, `WRITING_DISCIPLINE`, `_build_base_prompt()`, `generate_notes()` |
| Chunk wrappers | `PROMPT_INITIAL`, `PROMPT_CONTINUATION` |
| Intelligence | `INTEL_*_PROMPT`, `INTEL_SYNTHESIS_PROMPT`, `extract_intelligence()` |
| Summary | `SUMMARY_*_PROMPT`, `SUMMARY_REFINEMENT_PROMPT`, `generate_summary()` |
| Analysis | `ANALYSIS_PROMPT`, `NOTES_QA_PROMPT`, `QUESTION_SUGGESTION_PROMPT` |

Key constants: `CHUNK_WORD_SIZE` 4000, `CHUNK_WORD_OVERLAP` 400, `INTEL_CHUNK_SIZE` 4000,
`INTEL_OVERLAP` 400, `MAX_OUTPUT_TOKENS` 65536, `AUDIO_SEGMENT_SECONDS` 300,
`AUDIO_OVERLAP_SECONDS` 20, `MAX_PDF_MB` 25, `MAX_AUDIO_MB` 200.

---

## Gotchas

- **SDK.** Uses `google-genai`. The legacy `google-generativeai` was retired 30 Nov 2025
  and cannot reach Gemini 3.x. `_ModelHandle` preserves the old `.generate_content()`
  signature so call sites did not need rewriting — if you add a call, use `get_model()`,
  not the client directly.
- **Streaming responses are generators.** You cannot set attributes on them; that is what
  `_StreamHandle` is for. Do not "simplify" it away or cost tracking breaks silently.
- **ffmpeg + ffprobe** are system binaries, declared in the **repo-root** `packages.txt`,
  not in this folder. Cloud installs system packages only from the repo root. Without
  them, `_segment_audio()` returns `[]` and transcription falls back to single-shot —
  degraded, but it will not crash.
- **Root `requirements.txt` pins BOTH SDKs.** `google-genai` for this app,
  `google-generativeai` because the other five apps in this repo still import it. Do not
  remove either.
- **Notes run on a preview model.** `gemini-3.1-pro-preview` is the only Pro-class model
  above 2.5 — no Pro has reached GA as of Aug 2026. Preview endpoints change without
  notice. **If note quality regresses for no apparent reason, switch Notes to
  `gemini-2.5-pro` first** — that is the most likely cause and the cheapest test.
- **3.7 Flash promo pricing ends 31 Dec 2026** — input/output go $0.75/$3.75 → $1.50/$7.50.
  Update `MODEL_PRICING` then or the cost panel under-reports by half.
- **Audio input is billed above the text rate** on some models (2.5 Flash: $1.00/M vs
  $0.30/M). `compute_cost()` is modality-blind, so audio-heavy sessions read slightly low.
- **`saved_prompts_pro.json`** is written next to `app.py` at runtime. On Streamlit Cloud
  that filesystem is ephemeral — saved custom prompts do not survive a restart.
- **`GEMINI_API_KEY`** comes from env or Streamlit secrets. `_client` is `None` without it
  and every call raises a clear RuntimeError.

---

## History

**Aug 2026 overhaul.** Symptom: notes were missing key points from recordings. Root causes
found, in order of impact:

1. The note-style radio defaulted to **Concise**, and the Concise prompts were not shorter
   versions of the Detailed ones — they had dropped the ZERO SKIPPING rule, the
   one-bullet-per-point rule, multi-step explanations, and two whole nuance categories
   (comparisons, tangents). Fixed by deleting the concise/detailed split entirely.
2. Refinement ran on `gemini-2.5-flash-lite` with no anti-omission instruction — a weak
   model rewriting the full transcript, silently compressing it.
3. Audio was cut every 300s with **zero overlap**, so every seam could swallow a sentence.
4. Transcription ran on a preview endpoint that Google had superseded.
5. The app was on a retired SDK that cannot reach current models.

Also fixed on review: a self-contradiction between "remove nothing else" and the
tightening rules; a repetition loophole; a length quota in the continuation wrapper; and
voice notes being forced into a Q&A structure with no interviewer present.

All three note prompts now carry identical figure handling — `$`, `₹`, crore and lakh,
plus *"reproduce figures in the unit the speaker used — never convert, never round."*
Keep them in step: this rule drifted once already because the management prompt was
skipped for having the rupee symbol but not the rest of the rule.
