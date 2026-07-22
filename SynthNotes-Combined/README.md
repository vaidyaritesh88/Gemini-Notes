# SynthNotes Combined

The two apps you actually used together in your transmission-sector research —
**SynthNotes MultiDocLean** (the boilerplate extractor) and **SynthNotes MultiDoc**
(the notes + synthesis engine) — joined into **one guided pipeline**.

Instead of running one app, downloading its output, and pasting it into the next
app by hand, this runs all three stages in a single click — and still lets you
jump in at any stage if you already have the earlier output.

## The three stages

1. **Extract** — strips the boilerplate out of annual reports & investor
   presentations, keeping only the analyst-relevant narrative. *(Lifted from
   MultiDocLean.)*
2. **Notes (Map)** — chunks the extracted content + transcripts and turns each
   chunk into structured notes following your instructions. *(From MultiDoc.)*
3. **Synthesise (Reduce)** — plans an outline, writes each section in parallel,
   and stitches them into one coherent equity-research note. *(From MultiDoc.)*

## Start from any stage

A "Start from" selector lets you begin at:

- **Start fresh** — raw annual reports + transcripts → runs all three stages.
- **I already have the extracted content** → skips Stage 1.
- **I already have the interim notes** → jumps straight to the final write-up
  (no tokens spent re-reading sources).

Every stage's output is offered as a download, so you can stop, inspect, and
resume later without paying to redo earlier stages.

## What it reuses vs. what's new

The extraction, map, synthesis, PDF export and cost-tracking code is lifted
**verbatim** from your existing MultiDocLean and MultiDoc apps, so output quality
matches what you already got. Only the interface and the stage-wiring are new.

One deliberate improvement over MultiDocLean: this app uses MultiDocLean **only
for extraction** and MultiDoc's plan-then-write for synthesis — it never runs
MultiDocLean's own synthesis stage, which was the part that re-sent the full
notes into every section and burned tokens.

## Inputs

Plain-text (`.txt`) versions of your documents, same as the originals expect.
By default the transcripts are passed **raw** to the notes stage (matching how
you worked); a checkbox lets you boilerplate-strip them too.

## Running it

### On Streamlit Community Cloud (recommended)

1. Copy this whole `SynthNotes-Combined/` folder into your `Gemini-Notes` GitHub
   repo (alongside `SynthNotes-Pro`, `SynthNotes-MultiDoc`, etc.) and push.
2. Go to [share.streamlit.io](https://share.streamlit.io), point a new app at
   `SynthNotes-Combined/app.py`.
3. Under **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. Deploy.

### Locally

```bash
cd SynthNotes-Combined
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
streamlit run app.py
```

If no key is found in secrets or environment, the app shows a box in the sidebar
to paste one for the session.

## Models & cost

Defaults mirror the originals: Flash Lite for extraction, Gemini 2.5 Flash for
notes, and **Gemini 3.0 Flash for the final synthesis** (the same model you ran
in MultiDoc/MultiDocLean). All selectable in the sidebar — switch synthesis to
Pro if you ever want maximum quality on a specific note. A per-stage token/cost
panel appears under the outputs after each run (approximate — check Google Cloud
billing for exact figures).
