# SynthNotes Weekly — working notes

Streamlit app that turns one call's material (transcript / SynthNotes Pro notes / intelligence
brief) plus the analyst's own view into a call write-up in the analyst's voice, and turns a
week's call write-ups into the weekly. Single file: `app.py`. Sibling of `SynthNotes-Pro` in
`vaidyaritesh88/Gemini-Notes`; deploy as its own Streamlit Cloud app pointing at
`SynthNotes-Weekly/app.py`, with `GEMINI_API_KEY` in that app's secrets.

Why it exists (Sep 2026): the CIO asked for a write-up after every expert / management call
(who, when, context, key learnings), not just the Friday weekly. The analyst does not want to
hand over raw transcripts, because the weighting of what an expert said is the analyst's job.
So the tool produces the write-up from the analyst's side: facts from the call, over-statements
attributed or dropped, the analyst's weighting on top.

---

## Pipeline

```
CALL WRITE-UP page
  source = intelligence brief + notes + transcript (any subset, concatenated in that order)
  [1] DETAILED   DETAILED_PROMPT  -> 1,500-2,000 words of bullets (bold lead-in + sub-bullets)
                 over-statements: attribute ("management claims"), caveat if material, drop if empty
  [2] SUMMARY    SHORT_PROMPT     -> 350-450 words prose from [1] + analyst's view
                 over-statements: left out entirely
  assembled = header (title / type / who / date / context) + summary + "Details below:" + detailed
  auto-download .md; .docx export via python-docx

WEEKLY page
  inputs = write-ups sent from the Call page + uploaded files + paste boxes, + framing box
  [3] WEEKLY     WEEKLY_PROMPT    -> 750-850 words, one bold-titled section per company
```

Every stage runs `enforce_band()`: if the output is outside its band by more than
`BAND_TOLERANCE` (8%), one `ADJUST_PROMPT` pass expands or trims it against the source. Bands
are `DETAILED_BAND`, `SHORT_BAND`, `WEEKLY_BAND`. The band is enforced in code because a word
count in the instruction alone is not reliably honoured.

`REFINE_PROMPT` powers the Refine box on both pages; it always receives the source of truth
so a refinement cannot import new facts.

---

## The voice

`VOICE_GUIDE` and `STYLE_EXAMPLES` are the product. They were distilled from the weeklies in
`Janchor\Weekly` (Sep 2025 – Sep 2026): first person plural, no conviction language, explicit
attribution of anything that is a speaker's claim, hedged conclusions, opening paragraph that
ends in "Details below:", bullets with bold lead-ins, closing on what we will do next. The
prohibited-word list is in `VOICE_GUIDE`; the examples are the analyst's own sentences.

If output drifts from how the weeklies read, fix `VOICE_GUIDE` first, then the examples. Do
not add a "tone" selector; there is one voice.

**Over-statement policy differs by section, deliberately.** Detailed: keep the fact,
attribute the strength, caveat if material and unverifiable, drop if hollow. Summary and
Weekly: leave them out. The CIO reads the summary; the detail is there for audit.

---

## Model

`DEFAULT_MODEL` is `gemini-3.1-pro-preview` (preview endpoint; if quality regresses, try
`gemini-2.5-pro`). Temperature 0.4. A full call write-up is roughly 15-25k input tokens and
3k output, so a few cents; a weekly is less. `MODEL_PRICING` mirrors SynthNotes-Pro; 3.7 Flash
promo pricing ends 31 Dec 2026.

---

## Gotchas

- **API key** from `SynthNotes-Weekly/.env` (gitignored) locally, or Streamlit Secrets on
  the cloud. `_client` is `None` without it; every page stops with a clear message.
- **.docx reading** does not use python-docx (its lxml DLL is blocked on the analyst's
  Windows machine by an Application Control policy). `_docx_text()` parses `document.xml`
  with regex: paragraphs, bullet levels, bold runs. Good enough for weeklies and notes.
- **.docx writing** does use python-docx, guarded by try/except; on the analyst's machine
  the button is replaced by a caption, on Streamlit Cloud it works.
- **No `packages.txt`** in this repo, on purpose (see SynthNotes-Pro/CLAUDE.md).
- **Session state**: `call_out`, `weekly_out`, `weekly_calls` (the bucket that "Send to
  Weekly" appends to), `usage_log`. All lost on a Streamlit restart, which is why outputs
  auto-download on completion.
