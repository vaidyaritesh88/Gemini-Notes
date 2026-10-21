# SynthNotes Pro (Claude) — working notes

Claude copy of SynthNotes Pro (`../SynthNotes-Pro`, the Gemini "Note Maker" at
<https://pro-note-maker.streamlit.app/>). Built 21-Oct-2026. **Identical prompts, chunking,
pages and pipeline** — only the model layer differs: every text step runs on Claude
(Fable 5.1 / Opus 5.5); transcription stays on Gemini because Claude has no audio input.
Deployed as a second Streamlit app from `SynthNotes-Pro-Claude/app.py` in
`vaidyaritesh88/Gemini-Notes`. Secrets: `ANTHROPIC_API_KEY` (required) and
`GEMINI_API_KEY` (audio only). Single file: `app.py`, ~2,600 lines.

**Intentional prompt differences from the Gemini app (21-Oct-2026, after comparing both on the
Rishi Kalia call):**
- `WRITING_DISCIPLINE`: "one idea per bullet" became "one point per bullet, written as one full
  sentence" plus "fold follow-ups into the question they follow". Opus took the old rule literally
  (12-word bullets, 86 headings for a 65-min call vs 45 from Gemini). Coverage rules are untouched.
- Intelligence prompts: every bullet starts with a bold headline, like the Gemini brief, which
  scanned better.
- `refine_transcript(..., vocabulary=)`: an optional names list that refinement may use to fix a
  misheard name, never to insert one.
- `transcribe_audio`: a segment outside 40-350 words a minute (`SPARSE_` / `RUNAWAY_WORDS_PER_MINUTE`)
  is retried up to `SPARSE_RETRIES` (2) times. Both failures were seen with no error: 17 words for
  5 minutes, and a 3,222-word loop for 4.8 minutes. A real Hinglish call ran at 200-240 words a minute.
- `transcribe_audio(..., cache_dir=)`: optional per-segment checkpoints (seg_NNN.txt) so a failed run resumes
  at the failed segment; used by the Call Pipeline. Uploads retry 4 times; generate_with_retry treats
  disconnects/DNS errors as transient.
- Tested 21-Oct: no Gemini model reliably transcribes "ICICI ne bhi" (it sounds like "kisi ne bhi");
  2.5 Pro and 3.1 Pro heard "AU" instead, which is worse. 3.7 Flash stays the transcription default.

**Keep the two apps in step.** A prompt fix in one belongs in the other; diff the two
`app.py` files — the only intended differences are the CONFIG block, the SDK adapter,
the transcription selectboxes (`AUDIO_MODELS`), the key check and help/caption text.

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
[2] REFINEMENT  ── claude-opus-5-5 ── OPTIONAL, UI toggle
    ≤4,000 words: one call │ longer: 4,000-word chunks, 400-word overlap, 3 in parallel
    corrects language, labels speakers, translates Hindi/Hinglish,
    deletes only verbal noise
    → shrink guard warns if output falls below 80% of input
   │
   ▼
[3] NOTES  ── claude-fable-5-1
    prompt = meeting-type prompt + WRITING_DISCIPLINE + user context box
    ≤4,000 words: one call │ longer: PROMPT_INITIAL then PROMPT_CONTINUATION
    with a rolling context package; heading-level dedup on merge
   │
   ▼
[4] INTELLIGENCE  ── claude-opus-5-5
    notes chunked 4,000/400 → per-chunk extraction → INTEL_SYNTHESIS if >1 chunk
   │
   ├──► SUMMARY page   ── claude-fable-5-1  (~150w brief + summary at chosen length)
   └──► ANALYSE page   ── claude-fable-5-1  (Q&A over notes, suggested questions)

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
| Transcription | `gemini-3.7-flash` | Claude has no audio input; `AUDIO_MODELS` list only |
| Refinement | `claude-opus-5-5` | Rewrites the whole transcript — never put a weak model here |
| Notes | `claude-fable-5-1` | Most capable model on a long rule-heavy prompt |
| Intelligence | `claude-opus-5-5` | Classification over already-dense notes |
| Summary | `claude-fable-5-1` | Prose quality |
| Analysis | `claude-fable-5-1` | Reasoning over notes |

Prices per 1M tokens (Oct 2026): Fable 5.1 $10 / $50, Opus 5.5 $4 / $20 — Fable is the
stronger and 2.5x dearer. Thinking tokens bill as output. Rough estimate ~$2–2.50 per
one-hour meeting with these defaults (~$1–1.20 all-Opus); replace with real cost-panel
figures once measured. Ritesh's API budget is a monthly figure (~$250, unconfirmed).

Claude call settings (in `_ModelHandle._claude_stream`): always streamed;
`max_tokens` = `CLAUDE_MAX_TOKENS` 128000 (thinking shares the budget); `thinking` omitted
(always on for both models — sending `disabled` or `budget_tokens` is a 400);
`output_config.effort` = `CLAUDE_EFFORT` "high" (Opus 5.5 defaults to "medium");
`fallbacks="default"` + beta `server-side-fallback-2026-07-01` re-runs a classifier
refusal on another model. A final refusal raises; a `max_tokens` stop shows a warning.
No assistant prefill on these models.

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

- **Two SDKs.** `anthropic` for every text step; `google-genai` for transcription. `get_model()`
  routes by model id (`claude-*` → Anthropic). Claude responses are wrapped in
  `SimpleNamespace` objects carrying `.text` and Gemini-style `usage_metadata` field names,
  so every call site and the cost panel work unchanged.
- **Gemini SDK.** Uses `google-genai`. The legacy `google-generativeai` was retired 30 Nov 2025
  and cannot reach Gemini 3.x. `_ModelHandle` preserves the old `.generate_content()`
  signature so call sites did not need rewriting — if you add a call, use `get_model()`,
  not the client directly.
- **Streaming responses are generators.** You cannot set attributes on them; that is what
  `_StreamHandle` is for. Do not "simplify" it away or cost tracking breaks silently.
- **ffmpeg comes from a pip wheel, not apt.** `_ffmpeg_exe()` resolves it via
  `imageio-ffmpeg` (in `requirements.txt`), falling back to a system `ffmpeg`. There is
  deliberately **no `packages.txt`** in this repo: Streamlit Cloud runs `apt-get` when
  one exists, and on 2026-09-07 its base image's Debian `bullseye-security` metadata
  expired, so apt exited non-zero and every deployment in the repo failed to build. Do
  not reintroduce `packages.txt` — it puts that failure mode back.
- **There is no ffprobe.** The wheel ships ffmpeg only, so `_audio_duration_seconds()`
  parses the `Duration:` line from ffmpeg's stderr. ffmpeg exits non-zero when given no
  output file; that is expected. On failure it returns 0.0 and transcription falls back
  to single-shot — degraded, but it will not crash.
- **Root `requirements.txt` pins BOTH SDKs.** `google-genai` for this app,
  `google-generativeai` because the other five apps in this repo still import it. Do not
  remove either.
- **If note quality regresses**, compare the same transcript in the Gemini app first; then
  check `CLAUDE_EFFORT` (raise to "xhigh") before changing models.
- **3.7 Flash promo pricing ends 31 Dec 2026** — input/output go $0.75/$3.75 → $1.50/$7.50.
  Update `MODEL_PRICING` then or the cost panel under-reports by half.
- **Audio input is billed above the text rate** on some models (2.5 Flash: $1.00/M vs
  $0.30/M). `compute_cost()` is modality-blind, so audio-heavy sessions read slightly low.
- **`saved_prompts_pro.json`** is written next to `app.py` at runtime. On Streamlit Cloud
  that filesystem is ephemeral — saved custom prompts do not survive a restart.
- **Keys** come from env or Streamlit secrets (`_secret()`). Without `ANTHROPIC_API_KEY`
  the app stops at `api_key_check()`; without `GEMINI_API_KEY` text/PDF still work and
  audio raises a clear RuntimeError.
- **Fable 5.1 needs 30-day data retention** on the Anthropic org (not available under zero
  data retention) — a 400 on every Fable call means the org's retention setting.
- **Offline test:** `python -I test_claude_adapter.py <abs path to app.py>` checks the
  adapter with the real SDK over a mock HTTP transport (request body and beta header,
  streaming, usage/cost, refusal, model routing). No key or network needed; run it after
  any change to the adapter or the anthropic SDK version.

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
