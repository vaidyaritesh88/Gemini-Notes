"""SynthNotes Weekly — call write-ups and the weekly, in the analyst's own voice.

Two pages:
  Call Write-up  : transcript / notes / intelligence brief (+ analyst's view)  ->
                   short summary (350-450 words) on top of a detailed bullet
                   section (1500-2000 words), with the call header the CIO asked for
                   (who, when, context, key learnings).
  Weekly         : two to five call write-ups (+ the analyst's framing)  ->
                   a 750-850 word weekly in the house style.

Single file. Sibling of SynthNotes-Pro in the Gemini-Notes repo; same SDK adapter.
"""

import html as html_module
import io
import json
import os
import re
import time
import zipfile
from datetime import date, datetime
from typing import List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
load_dotenv()


# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    key = (key or "").strip().strip('"').strip("'")
    if "PASTE" in key.upper() or len(key) < 20:   # the .env placeholder, or junk
        return ""
    return key


_api_key = _load_api_key()
_client = genai.Client(api_key=_api_key) if _api_key else None

# Live models only. Writing quality matters more than cost here: a weekly is ~15k
# input tokens and ~1.5k output, so even the Pro model costs a few cents a run.
# gemini-2.5-pro was removed 12 Sep 2026: Google now returns 404 "no longer available to
# new users" for it. If 3.1 Pro preview regresses, 3.7 Flash is the fallback.
MODELS = {
    "Gemini 3.1 Pro (Best writing, preview)": "gemini-3.1-pro-preview",
    "Gemini 3.7 Flash (Fast)":                "gemini-3.7-flash",
    "Gemini 2.5 Flash (Cheapest)":            "gemini-2.5-flash",
}
DEFAULT_MODEL = "Gemini 3.1 Pro (Best writing, preview)"

MODEL_PRICING = {  # USD per 1M tokens (input, output)
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.7-flash":       (0.75,  3.75),
    "gemini-2.5-flash":       (0.30,  2.50),
}

MAX_OUTPUT_TOKENS = 16384

CALL_TYPES = ["Expert call", "Management meeting", "Dealer / channel check", "Internal discussion"]

# Word bands. The band is enforced in code (one adjustment pass), not just by instruction.
DETAILED_BAND = (1500, 2000)
SHORT_BAND    = (350, 450)
WEEKLY_BAND   = (750, 850)
BAND_TOLERANCE = 0.08   # ±8% before an adjustment pass is triggered


# ── 2. THE VOICE ───────────────────────────────────────────────────────────────
# Distilled from the analyst's weeklies (Sep 2025 – Sep 2026). Every prompt carries this
# block. If the output drifts from how the weeklies read, fix it here first.

VOICE_GUIDE = """### HOW THIS ANALYST WRITES — apply throughout

**Perspective and stance**
- First person plural throughout: "we met", "we spoke with", "our checks suggest", "we will keep tracking". Never "I".
- The register is measured and non-controversial. Findings are reported as what was heard, not as verdicts. Interest in a business is expressed softly ("it can be an interesting bank", "this is something we will keep track of"), never as a recommendation, a call to action, or a view on the stock or its price.
- High-conviction language is not used. Do NOT write: clearly, definitely, certainly, undoubtedly, obviously, game-changer, massive, huge, enormous, exceptional, exceptionally, undisputed, unmatched, transformational, revolutionary, must, will surely, no doubt, strongly, extremely, best-in-class, world-class, unprecedented, remarkable, remarkably, impressive, striking, alarming, disaster, collapse, surge/surging, skyrocket, plunge, crucial, critical, vital, paramount. No exclamation marks. No rhetorical questions.
- Preferred hedges, used naturally and not in every sentence: "seems", "appears", "it seems that", "possibly", "could", "should", "in some cases", "as of now", "for now", "initial evidence", "we are not sure if", "remains to be seen", "we will need to do more work to understand", "some more seasoning is needed", "difficult to estimate".
- Attribution is explicit whenever a statement is the speaker's claim rather than an established fact: "as per the management", "management claims", "management mentioned", "the expert felt", "the expert's view is", "dealers mentioned", "IR shared", "our checks suggest", "checks indicate". Any claim that could be self-serving (a target, a comparison with competitors, a promise of future improvement, a description of one's own strengths) is always attributed, never stated in the writer's voice.
- The analyst's own weighting appears as a short plain sentence, not an argument: "This is a very important outcome and we will monitor the roll-out", "we do not expect these launches to be an immediate needle-mover", "we will need to get full clarity if this method is sustainable over the longer term", "all the mentioned changes look positive but we will need to do checks to understand how many have been implemented on the ground".

**Structure habits**
- Opening paragraph(s): who we met or spoke to and how (broker-arranged group meeting, ex-employee via a network, phone calls with N dealers across named states), a one-to-three sentence recap of what the business is (mix, scale, positioning) for a reader who may not follow it, why we spoke to them, and then the headline learning in two to four sentences. The opening ends with a lead-in such as "Details below:", "Additional details below:", "Following are the KTAs:" or "Key takeaways below:".
- Body: bullets. Each top-level bullet opens with a **bold lead-in phrase and a colon**, followed by one to three plain sentences. Supporting specifics (figures, examples, names, timelines) go in indented sub-bullets.
- Closing paragraph: what remains open and what we will do next. "We will keep doing checks on X in the coming months", "We will do more work to understand the impact of Y", "We will update the team as we do more research in this area".

**Numbers and terms**
- Indian equity conventions: Rs, Rs75mn, Rs1.2tn, ₹250 crore where the source used crore, ~, %, bps, YoY, QoQ, CAGR, FY27, 1QFY27, 2HFY27, Apr'26, RoE, RoA, NIM, AUM, LAP, MSME, TAT, KTAs, IR, OEM, DSA, MFD.
- House unit is Rs mn/bn/tn (Rs75mn, Rs1.2tn). A figure given in crore or lakh may be converted exactly (Rs100 crore = Rs1bn, Rs10 lakh = Rs1mn) or kept as ₹ crore; never round, never change a percentage or a ratio. Use "~" for figures the source gave as approximate.
- Company short forms in brackets on first use: City Union Bank (CUBK), L&T Finance (LTF), Bajaj Finance (BAF).
- British/Indian spelling: favour, organisation, analyse, programme, centre.
- No em-dashes anywhere. Use a comma, brackets, or a new sentence instead.
"""

STYLE_EXAMPLES = """### EXAMPLES OF THE ANALYST'S OWN PROSE (match this register)
These show the register only. Do not reuse their sentences or phrases; every sentence you write must come from this call's material and the analyst's own words.

Opening of a dealer-check write-up:
"We recently went to Chennai to do checks on Cholamandalam. We focused on the LAP (Loan Against Property) and Home Loan business segments of Cholamandalam which account for ~32% of the loan book but are expected to grow much faster at 30%+ YoY vs high teens growth for the larger vehicle finance (~55% of loan book) business. As a result, it is important to understand the credit underwriting practices of Chola in these two segments to invest from the longer-term point of view. Based on our checks, it seems that Chola is following a very granular underwriting approach in these loan segments and penetrating deeper geographies. While it might seem from outside that Chola is lending to weaker segment customers, this understanding of the customer cash-flows and the collateral has ensured that Chola is able to grow in these segments without significant asset quality issues. Details are mentioned below:"

Opening of a management-meeting write-up:
"We recently met the management team of City Union Bank (CUBK) in a broker arranged group meeting. This is a regional bank with ~2/3rd of its loan book to MSME customers in Tamil Nadu and ~80-85% of the loan book is in the South Indian states. The reason for us to meet them was to understand the MSME asset quality and to understand the changes being brought in by the new MD & CEO. If the new management can improve loan growth trajectory above category growth while ensuring they maintain their asset quality and profitability, it can be an interesting bank. As per the management, the on-ground asset quality situation for MSME businesses remains solid with no material impact seen on the cash flows of these businesses due to the West Asia crisis. Additional details below:"

How a management claim is weighed:
"IR did share their plans of launching new bikes in the premium segment, but given their track-record, we do not expect these launches to be an immediate needle-mover for the company. IR mentioned changes in the production processes to improve the quality of their products at launch itself. This sounds interesting and should be monitored from longer term. There was not any material new information which would change our view on the business right now."

A body bullet:
"- **BAF can still offer new loans with flexi-loan like re-payment structure but without flexibility:** As per management, out of BAF's total loan book, ~19-20% is the flexi loan outstanding. Management claims that only half of the Flexi-loan customers utilise this flexibility. In a Flexi-loan, BAF allows the customer to repay only the interest component for the first two years and then the full EMI starts.
  - **New products will have slightly lower fees:** Flexi-loans had additional fees which BAF charged the customers for giving them the flexibility. There will be loss of these fees but as per our checks the new products are being built with a certain set of fees to partially off-set this loss."

Closings:
"All the mentioned changes look positive but we will need to do checks to understand how many of these changes have been properly implemented on the ground, how the old staff is adjusting to these changes vs their traditional methods of doing business and if any of these tech-related changes are diluting the underwriting standards."
"While the AI-led initiatives look interesting, whether they result in sustained benefit on credit cost and lower collections opex remains to be seen as the portfolios are not seasoned yet. We will continue to monitor the progress of the business going ahead."
"""


# ── 3. PROMPTS ─────────────────────────────────────────────────────────────────

DETAILED_PROMPT = """You are writing the DETAILED section of an analyst's call write-up for the CIO. The source material below is from one call: a transcript and/or structured notes and/or an intelligence brief (any of them may be present; they overlap). The analyst may also have given their own view on which points to weight.

Write ONLY the detailed section, as bullet points. Target {lo}-{hi} words.

**Format**
- Organise by topic in the order that makes the call easiest to follow, which is not necessarily the order of the conversation. Typically 5-9 topics.
- Each topic is a top-level bullet: "- **Topic lead-in phrase:** one or two plain sentences stating the point." followed by indented sub-bullets ("  - ") carrying the specifics: figures, examples, names, timelines, comparisons, the speaker's reasoning.
- No headers, no introduction, no closing paragraph, no "key takeaways" section, no "Details below:" line. The output starts directly with the first "- **" bullet and ends with the last sub-bullet.
- Use "-" for every bullet, never "*" or numbers.

**Fidelity**
- Everything must come from the source material. No inferences, no added context, no outside facts, nothing the speaker did not say.
- Keep every figure in the unit the speaker used. Keep the speaker's own qualifiers ("roughly", "in most cases", "typically", "at some dealers").
- Cover the whole call. Anecdotes and digressions are often the most useful part of a call; do not drop them as off-topic. A point the speaker returned to more than once is signal: keep it once in its most complete form and note that it was emphasised.

**Handling over-statements (important)**
- The source will contain claims stronger than the evidence behind them: sweeping generalisations, "always"/"never"/"everyone", "the best in the industry", promotional descriptions of the speaker's own company or product, round-number predictions, and adjectives such as massive/huge/unprecedented. Do not reproduce these in the writer's voice.
- Where there is a useful fact underneath, keep the fact and attribute the strength to the speaker: "management claims...", "the expert felt that...", "as per the dealer...".
- Where a claim is material to the thesis but cannot be verified from the call, keep it attributed and add a short caveat in the analyst's voice, for example "(this is the expert's view and would need to be cross-checked)" or "we would treat this as the management's expectation rather than a given".
- Where it is plainly an over-statement with no fact underneath, leave it out.
- Never caveat plain facts (numbers, dates, product names, process descriptions).

**The analyst's weighting**
- If the analyst has said which points they trust, which they discount and why, reflect that: trusted points get top-level bullets and fuller treatment; discounted points get a sub-bullet and a brief note in the analyst's voice ("we would treat this with caution given..."). Never contradict the analyst's stated view; never add a view the analyst has not expressed.
- Analyst-voice sentences (caveats, weighting, what we will do next) are at most one short sentence per topic, placed where they apply, and never the same sentence twice. Paraphrase the analyst's words into the flow; do not paste their note verbatim.

{voice}

---
CALL TYPE: {call_type}
CALL HEADER (who / when / why we spoke): {header}

ANALYST'S CONTEXT AND VIEW (may be empty):
{analyst_view}

---
SOURCE MATERIAL:
{source}
"""

SHORT_PROMPT = """Write the SUMMARY that sits at the top of the analyst's call write-up. Use only the detailed section below and the analyst's context and view. Target {lo}-{hi} words. Plain prose paragraphs; no bullets, no headers, no bold.

**Shape**
- Paragraph 1: who we spoke to and how, what the business or topic is in one or two sentences (for a reader who may not follow it), and why we spoke to them.
- Paragraphs 2-3: the three to five things that matter from this call, in the analyst's voice, with the key figures woven in. This is where the analyst's weighting shows: what we take from the call, what we would discount, what remains open. Attribute what is the speaker's claim; state plainly what is fact.
- Final paragraph (two or three sentences): what we will do next, keep tracking, or need to verify.
- Do NOT write any lead-in line such as "Details below:", "Additional details below:", "Key takeaways below" or "discussed below"; the app adds that itself. End on the final paragraph.
- Use the analyst's view as the weighting behind the paragraphs, paraphrased into the flow; do not paste the analyst's note verbatim and do not repeat the same analyst sentence twice.

**Over-statements**: in this summary, promotional or sweeping claims from the speaker are simply left out. Do not restate them even with a caveat. Report what was learnt.

**Do not** recommend, rate, or express a view on the stock. Do not use any of the words the voice guide prohibits. Hedge conclusions the way the examples do.

{voice}

{examples}

---
CALL TYPE: {call_type}
CALL HEADER (who / when / why we spoke): {header}

ANALYST'S CONTEXT AND VIEW (may be empty):
{analyst_view}

---
DETAILED SECTION (source of truth):
{detailed}
"""

WEEKLY_PROMPT = """You are writing this week's write-up for the CIO from {n} call write-up(s) produced during the week (below), plus the analyst's framing of the combined learning. Target {lo}-{hi} words in total (a hard ceiling; see the per-component limits below).

**Structure (this is how the analyst writes every week)**
{structure_block}
- Where two calls said different things, say so plainly and say which way we lean and why, without strong language ("our checks pointed in a similar direction", "the expert's view differs from management's on this and we will need to do more work").
- The combined learning across the calls goes in the analyst's voice, hedged, inside the opening or closing paragraphs. Do not add a separate "conclusion" or "summary" section.
- Budget the words across sections according to how much genuinely new information each call carried, not equally.
- The {lo}-{hi} word total is a ceiling. You cannot count words reliably, so obey these per-component limits instead, which add up to the ceiling:
{budget_block}
  Choose the bullets that carry the most; the CIO already has the detailed write-ups, so the weekly does not repeat every specific.

**Fidelity**: only what is in the call write-ups and the analyst's framing. No outside facts. No inference beyond what the analyst has stated. Over-statements from speakers are left out entirely or reduced to the attributed fact underneath. Keep figures exactly as given.

**Do not** recommend, rate, or express a view on the stock or its price. Do not use any of the words the voice guide prohibits.

{voice}

{examples}

---
ANALYST'S FRAMING FOR THE WEEK (may be empty):
{framing}

---
CALL WRITE-UPS:
{calls}
"""

WEEKLY_STRUCTURE_COMBINED = """- Write ONE combined write-up, not one section per call. The calls are treated as one body of checks on the same company or question.
- One bold title on its own line covering the whole piece, in the analyst's forms: "**Checks on X**", "**Discussion on X**", "**Checks on X and Y**".
- One opening paragraph: who we spoke to across all the calls (list them briefly: "we spoke with a branch manager, two DSAs and a former zonal head"), a one-to-three sentence recap of the business if the reader may not follow it, why we did these checks, then the combined learning in two to four sentences, ending with "Details below:" or "Key learnings below:".
- Then bullets organised by TOPIC, never by call. Each bullet brings together what the different sources said on that topic, with attribution inside the bullet ("the branch manager mentioned...", "the DSA felt...", "management claims...", "the former employee's view is..."), and notes where sources agreed or differed. A point made by several sources is stated once with the agreement noted, not repeated per source.
- Then one closing paragraph on what remains open and what we will do next.
- Never produce two parallel write-ups stitched together; never repeat the opening or closing per call."""

WEEKLY_STRUCTURE_PER_COMPANY = """- One section per company. Section title in bold on its own line, in one of these forms: "**Discussion on X**", "**Checks on X**", "**Meeting with X**", "**X: [what the checks were about]**".
- Each section: an opening paragraph (who we spoke to and how, a one-to-three sentence business recap if the reader may not follow the company, why we spoke, then the headline learning in two to four sentences, ending with "Details below:" or "Following are the KTAs:"); then 4-8 bullets with bold lead-ins and colons, with sub-bullets for the specifics; then a closing sentence or two on what remains open and what we will do next.
- Calls about the same company go into that company's single section, merged by topic with attribution, never as two sub-sections."""

WEEKLY_STRUCTURES = {
    "One combined write-up (calls on the same company or theme)": WEEKLY_STRUCTURE_COMBINED,
    "One section per company (calls on unrelated companies)":     WEEKLY_STRUCTURE_PER_COMPANY,
}

ADJUST_PROMPT = """The text below is {kind}. It is {count} words; it must end up at about {target} words (anywhere between {lo} and {hi} is acceptable). {direction}

Rules: keep the same sections, bullets and voice; do not add anything that is not in the source material; do not remove any figure, name, or attributed claim when trimming; when expanding, pull more specifics from the source material rather than adding filler or repeating points. Return the complete revised text, from its first line to its last, and nothing else.

{voice}

---
SOURCE MATERIAL (for reference only; the revised text may not go beyond it):
{source}

---
TEXT TO REVISE:
{text}
"""

REFINE_PROMPT = """Revise the text below according to the analyst's instruction. Only change what the instruction asks for. Keep the structure, the voice, and every fact not affected by the instruction. Do not add anything that is not in the source material unless the instruction itself supplies it (the analyst's own view or context counts as source). Return the complete revised text and nothing else.

{voice}

---
SOURCE MATERIAL (source of truth):
{source}

---
CURRENT TEXT:
{text}

---
INSTRUCTION:
{instruction}
"""


# ── 4. SDK ADAPTER, RETRY, USAGE ───────────────────────────────────────────────

def _model_id(display_name: str) -> str:
    return MODELS.get(display_name, MODELS[DEFAULT_MODEL])


def _record_usage(model_id: str, usage, stage: str) -> None:
    if usage is None:
        return
    try:
        inp = int(getattr(usage, "prompt_token_count", 0) or 0)
        out = int(getattr(usage, "candidates_token_count", 0) or 0)
    except Exception:
        return
    log = st.session_state.setdefault("usage_log", [])
    log.append({"stage": stage, "model": model_id, "input": inp, "output": out,
                "cost": compute_cost(inp, out, model_id)})


def compute_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    pin, pout = MODEL_PRICING.get(model_id, (0.0, 0.0))
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def generate(prompt: str, model_display: str, stage: str, placeholder=None,
             max_retries: int = 3) -> str:
    """One model call with retry on transient errors and on an incomplete response.

    Deliberately NOT streamed: streamed responses from the 3.1 Pro preview endpoint were
    observed to end silently part-way through long rewrites (no error, text stops
    mid-sentence). The non-streamed response for the same prompt was complete, and it
    carries a finish_reason we can check."""
    if _client is None:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    model_id = _model_id(model_display)
    config = types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.4)
    if placeholder is not None:
        placeholder.caption(f"{stage}: writing… (the Pro model thinks for a minute or two first)")
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _client.models.generate_content(model=model_id, contents=[prompt], config=config)
            _record_usage(model_id, getattr(resp, "usage_metadata", None), stage)
            text = (resp.text or "").strip()
            finish = ""
            try:
                finish = str(resp.candidates[0].finish_reason or "")
            except Exception:
                pass
            if text and ("STOP" in finish.upper() or not finish):
                if placeholder is not None:
                    placeholder.empty()
                return text
            last_err = RuntimeError(f"incomplete response from {model_id} (finish_reason={finish or 'none'})")
            if "MAX_TOKENS" in finish.upper() and text:
                # Better a long-but-complete-enough draft than nothing; caller's band check will act.
                if placeholder is not None:
                    placeholder.empty()
                return text
        except Exception as e:
            last_err = e
            err = str(e).lower()
            transient = any(k in err for k in
                            ["429", "503", "500", "deadline", "timeout", "unavailable", "resource_exhausted"])
            if not transient:
                raise
        time.sleep(2 ** (attempt + 1))
    raise last_err or RuntimeError("generation failed")


# ── 5. TEXT UTILITIES ──────────────────────────────────────────────────────────

def weekly_budget_block(n_sections: int, hi: int) -> str:
    """Per-section limits that sum to the ceiling. Models obey item limits far better
    than a total: '5 bullets of at most 40 words' lands; '850 words total' does not."""
    if n_sections <= 1:   # one combined write-up
        opening, closing, n_bul = int(hi * 0.22), int(hi * 0.07), 7
        bullet = (hi - opening - closing) // n_bul
        return (f"  - One write-up, at most {hi} words in total.\n"
                f"  - Opening paragraph at most {opening} words; {n_bul} topic bullets at most {bullet} words "
                f"each including any sub-bullet; closing at most {closing} words.\n"
                f"  - Sub-bullets are optional; at most one per bullet.")
    n = max(1, n_sections)
    per = hi // n
    opening = max(60, int(per * 0.30))
    closing = max(25, int(per * 0.10))
    n_bul = 5 if per >= 350 else 4
    bullet = max(25, (per - opening - closing) // n_bul)
    return (f"  - {n} section(s), at most {per} words each.\n"
            f"  - Per section: opening paragraph at most {opening} words; {n_bul} bullets at most "
            f"{bullet} words each including any sub-bullet; closing at most {closing} words.\n"
            f"  - Sub-bullets are optional; at most one per bullet.")


_SECTION_TITLE_RE = re.compile(r"^\s*\*\*[^*\n]{3,120}\*\*\s*$")


def section_word_counts(text: str) -> List[Tuple[str, int]]:
    """Split on bold title lines; returns (title, words) per section."""
    out, title, buf = [], "(preamble)", []
    for ln in (text or "").splitlines():
        if _SECTION_TITLE_RE.match(ln):
            if buf and any(b.strip() for b in buf):
                out.append((title, word_count("\n".join(buf))))
            title, buf = ln.strip().strip("*").strip(), []
        else:
            buf.append(ln)
    if buf and any(b.strip() for b in buf):
        out.append((title, word_count("\n".join(buf))))
    return out


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _looks_truncated(text: str) -> bool:
    """A revision that stops mid-sentence is worse than the original."""
    tail = (text or "").rstrip()
    return bool(tail) and tail[-1].isalnum()


def enforce_band(text: str, band: Tuple[int, int], kind: str, source: str,
                 model_display: str, status_write, max_passes: int = 2) -> str:
    """Bring the text into its word band with up to two targeted adjustment passes.
    Each pass asks for a specific delta (cut ~N words / add ~N words). A candidate is
    accepted only if it lands in the band; otherwise the candidate closest to the band's
    midpoint is kept, and a candidate that over-cuts below the floor or stops
    mid-sentence is never chosen over the original."""
    lo, hi = band
    mid = (lo + hi) // 2
    lo_ok, hi_ok = lo * (1 - BAND_TOLERANCE), hi * (1 + BAND_TOLERANCE)

    def in_band(n):
        return lo_ok <= n <= hi_ok

    best, best_n = text, word_count(text)
    if in_band(best_n):
        return best
    current = text
    for _ in range(max_passes):
        n = word_count(current)
        if n > hi:
            cut = n - mid
            secs = section_word_counts(current)
            if len(secs) > 1:
                per = mid // len(secs)
                sec_lines = "\n".join(f"  - '{t}': currently {w} words, rewrite to at most {per} words"
                                       for t, w in secs)
                direction = (f"Cut roughly {cut} words in total by rewriting each section to its limit:\n{sec_lines}\n"
                             f"Within a section, shorten the opening paragraph first, then drop sub-bullets, "
                             f"then merge bullets that make one point. Keep every section title and its "
                             f"closing sentence. Do not cut more than asked: the whole text must still be at "
                             f"least {lo} words.")
            else:
                direction = (f"Cut roughly {cut} words ({cut * 100 // max(n, 1)}% of the text) by tightening "
                             f"wording, dropping the least informative sub-bullets, and merging bullets that "
                             f"make one point. Keep every top-level bullet. Do not cut more than asked: the "
                             f"result must still be at least {lo} words.")
        else:
            add = mid - n
            direction = (f"Add roughly {add} words by pulling more specifics (figures, examples, reasoning) "
                         f"from the source material under the existing points. Do not add new sections.")
        status_write(f"{kind} is {n:,} words (target {lo}-{hi}); adjusting…")
        try:
            cand = generate(ADJUST_PROMPT.format(kind=kind, count=n, target=mid, lo=lo, hi=hi,
                                                 direction=direction, voice=VOICE_GUIDE,
                                                 source=source, text=current),
                            model_display, f"adjust {kind}")
        except Exception as e:  # an adjustment failure must never lose the draft
            status_write(f"adjustment failed ({e}); keeping the draft")
            break
        cn = word_count(cand)
        usable = cn >= lo * 0.85 and not _looks_truncated(cand) and cn > 50
        if usable and abs(cn - mid) < abs(best_n - mid):
            best, best_n = cand, cn
        if in_band(cn) and usable:
            break
        current = best  # next pass works from the best text so far, never from a bad candidate
    if best is not text:
        status_write(f"{kind}: {best_n:,} words after adjustment")
    return best


_LEADIN_RE = re.compile(r"^\s*\**\s*(additional\s+|further\s+|more\s+)?(details?|key\s+takeaways?|ktas?)\s*(are\s+)?(mentioned\s+|discussed\s+|given\s+)?(below)?\s*[:.]?\s*\**\s*$", re.I)


def clean_summary(text: str) -> str:
    """Drop any trailing 'Details below:'-type line the model wrote despite instructions."""
    lines = (text or "").rstrip().splitlines()
    while lines and (_LEADIN_RE.match(lines[-1]) or not lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def clean_detailed(text: str) -> str:
    """Normalise bullet markers and drop stray intro paragraphs before the first bullet."""
    out = []
    for ln in (text or "").splitlines():
        m = re.match(r"^(\s*)[*•]\s+", ln)
        if m:
            ln = m.group(1) + "- " + ln[m.end():]
        out.append(ln)
    first = next((i for i, ln in enumerate(out) if re.match(r"^\s*-\s+", ln)), None)
    if first is None:
        return "\n".join(out).strip()
    kept = [ln for ln in out[:first]
            if ln.strip() and len(ln.split()) <= 12 and not _LEADIN_RE.match(ln)]
    return "\n".join(kept + out[first:]).strip()


def _docx_text(data: bytes) -> str:
    """Minimal .docx reader (paragraphs, bullets with indent, bold runs). No lxml needed."""
    try:
        xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf8")
    except Exception:
        return ""
    lines = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", para, flags=re.S)
        prefix = ""
        if ppr and "<w:numPr>" in ppr.group(0):
            m = re.search(r'<w:ilvl w:val="(\d+)"', ppr.group(0))
            prefix = "  " * (int(m.group(1)) if m else 0) + "- "
        txt = ""
        for run in re.findall(r"<w:r[ >].*?</w:r>", para, flags=re.S):
            t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", run, flags=re.S))
            if not t:
                continue
            if re.search(r"<w:b/>|<w:b w:val=\"(1|true)\"", run):
                t = "**" + t + "**"
            txt += t
        txt = html_module.unescape(txt).replace("****", "").strip()
        if txt:
            lines.append(prefix + txt)
    return "\n".join(lines)


def _pdf_text(data: bytes) -> str:
    try:
        import PyPDF2  # optional
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        return f"[Could not read PDF: {e}]"


def read_upload(uploaded) -> str:
    """Text from a Streamlit UploadedFile: .txt/.md/.pdf/.docx."""
    if uploaded is None:
        return ""
    data = uploaded.getvalue()
    name = (uploaded.name or "").lower()
    if name.endswith(".pdf"):
        return _pdf_text(data)
    if name.endswith(".docx"):
        return _docx_text(data)
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return ""


def _sanitize(s: str, fallback: str = "untitled") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (s or "").strip()).strip("_")
    return s or fallback


def filename_for(name: str, kind: str, ext: str, on: Optional[date] = None) -> str:
    d = (on or datetime.now()).strftime("%Y%m%d")
    return f"{d}_{_sanitize(name)}_{_sanitize(kind, 'output')}.{ext.lstrip('.')}"


# ── Markdown → .docx (the weekly is a Word document) ───────────────────────────

def markdown_to_docx_bytes(md: str) -> Optional[bytes]:
    """Bold lines, **inline bold**, and '-' bullets with two-space indent levels.
    Uses python-docx if importable (it is on Streamlit Cloud); returns None otherwise."""
    try:
        import docx  # python-docx
        from docx.shared import Pt
    except Exception:
        return None
    d = docx.Document()
    style = d.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    def add_runs(par, text):
        for i, part in enumerate(re.split(r"(\*\*.+?\*\*)", text)):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                par.add_run(part[2:-2]).bold = True
            else:
                par.add_run(part)

    for raw in md.splitlines():
        if not raw.strip():
            continue
        m = re.match(r"^(\s*)[-*•]\s+(.*)$", raw)
        if m:
            level = min(len(m.group(1)) // 2, 2)
            style_name = "List Bullet" if level == 0 else f"List Bullet {level + 1}"
            try:
                p = d.add_paragraph(style=style_name)
            except KeyError:
                p = d.add_paragraph(style="List Bullet")
            add_runs(p, m.group(2))
        else:
            p = d.add_paragraph()
            add_runs(p, raw.strip())
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# ── UI helpers ─────────────────────────────────────────────────────────────────

def copy_button(text: str, label: str = "Copy to clipboard", key: str = ""):
    components.html(
        f"""
        <button onclick="doCopy()" style="background:#1a1a2e;color:#fff;border:none;padding:0.45rem 1.2rem;
            border-radius:0.3rem;cursor:pointer;font-size:0.875rem;width:100%;min-height:38px;">
          {html_module.escape(label)}
        </button>
        <script>
        function doCopy() {{
            var btn = document.querySelector('button');
            navigator.clipboard.writeText({json.dumps(text)}).then(() => {{
                btn.textContent = 'Copied';
                setTimeout(() => btn.textContent = {json.dumps(label)}, 2000);
            }}).catch(() => {{
                btn.textContent = 'Copy failed, select and copy manually';
                setTimeout(() => btn.textContent = {json.dumps(label)}, 2500);
            }});
        }}
        </script>
        """,
        height=46,
    )


def auto_download(files: List[Tuple[str, str, str]]) -> None:
    """Fire browser downloads for (filename, text, mime) tuples, staggered 600ms."""
    blocks = []
    for i, (fn, content, mime) in enumerate(files):
        blocks.append(
            f"setTimeout(function() {{ var b=new Blob([{json.dumps(content)}],{{type:{json.dumps(mime)}}});"
            f"var u=URL.createObjectURL(b); var a=document.createElement('a'); a.href=u; a.download={json.dumps(fn)};"
            f"document.body.appendChild(a); a.click(); setTimeout(function(){{URL.revokeObjectURL(u);"
            f"document.body.removeChild(a);}},200); }}, {i * 600});")
    components.html("<script>" + "\n".join(blocks) + "</script>", height=0)


def api_key_check():
    if not _api_key:
        st.error("**GEMINI_API_KEY not set.** Put it in `SynthNotes-Weekly/.env` "
                 "(one line: `GEMINI_API_KEY=...`) or in Streamlit Secrets.")
        st.stop()


def render_usage_panel():
    log = st.session_state.get("usage_log", [])
    if not log:
        return
    tin = sum(x["input"] for x in log)
    tout = sum(x["output"] for x in log)
    cost = sum(x["cost"] for x in log)
    with st.sidebar.expander(f"Usage this session · ${cost:.3f}", expanded=False):
        st.caption(f"{tin:,} input · {tout:,} output tokens over {len(log)} calls")
        for x in log[-8:]:
            st.caption(f"{x['stage']} · {x['input']:,}/{x['output']:,} · ${x['cost']:.3f}")


def sidebar_common() -> str:
    st.sidebar.markdown("### SynthNotes Weekly")
    model = st.sidebar.selectbox("Model", list(MODELS.keys()),
                                 index=list(MODELS.keys()).index(DEFAULT_MODEL))
    render_usage_panel()
    return model


def output_block(text: str, base_name: str, kind: str, key: str, on: Optional[date] = None):
    """Render markdown + copy + downloads for a finished output."""
    st.markdown(text.replace("$", "\\$"))   # display only: bare $ pairs render as LaTeX
    st.caption(f"{word_count(text):,} words")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        copy_button(text, key=key)
    with c2:
        st.download_button("Download .md", text, file_name=filename_for(base_name, kind, "md", on),
                           mime="text/markdown", key=f"{key}_md", use_container_width=True)
    with c3:
        st.download_button("Download .txt", text, file_name=filename_for(base_name, kind, "txt", on),
                           mime="text/plain", key=f"{key}_txt", use_container_width=True)
    with c4:
        docx_bytes = markdown_to_docx_bytes(text)
        if docx_bytes:
            st.download_button("Download .docx", docx_bytes,
                               file_name=filename_for(base_name, kind, "docx", on),
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               key=f"{key}_docx", use_container_width=True)
        else:
            st.caption("(.docx export needs python-docx)")


# ── 6. PAGE: CALL WRITE-UP ─────────────────────────────────────────────────────

def _source_input(label: str, key: str, hint: str) -> str:
    with st.expander(label, expanded=(key == "transcript")):
        up = st.file_uploader("Upload (.txt / .md / .pdf / .docx)", type=["txt", "md", "pdf", "docx"],
                              key=f"{key}_file", label_visibility="collapsed")
        pasted = st.text_area("or paste", height=160, key=f"{key}_text", placeholder=hint,
                              label_visibility="collapsed")
        text = read_upload(up) if up is not None else ""
        if pasted.strip():
            text = (text + "\n\n" + pasted) if text else pasted
        if text.strip():
            st.caption(f"{word_count(text):,} words loaded")
        return text.strip()


def build_source(transcript: str, notes: str, intel: str) -> str:
    parts = []
    if intel:
        parts.append("=== INTELLIGENCE BRIEF ===\n" + intel)
    if notes:
        parts.append("=== DETAILED NOTES ===\n" + notes)
    if transcript:
        parts.append("=== TRANSCRIPT ===\n" + transcript)
    return "\n\n".join(parts)


def page_call():
    api_key_check()
    model = sidebar_common()
    st.title("Call write-up")
    st.caption("Transcript, notes and/or intelligence brief from one call, plus your view, "
               "in: a short summary on top of a detailed bullet section, in your voice.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        title = st.text_input("Title", placeholder="Discussion on City Union Bank / Call with ex-Policybazaar Head of Health")
    with c2:
        when = st.date_input("Date of call", value=date.today())
    with c3:
        call_type = st.selectbox("Type", CALL_TYPES)
    who = st.text_input("Who we spoke to (name, role, affiliation; network if any)",
                        placeholder="MD & CEO and CFO, broker-arranged group meeting / ex-Regional Manager, via Third Bridge")
    why = st.text_input("Context: why we spoke to them",
                        placeholder="To understand MSME asset quality and the changes under the new MD & CEO")
    analyst_view = st.text_area(
        "Your view and weighting (optional but this is what makes it yours)", height=140,
        placeholder=("What you take from the call, what you would discount and why, what stays open. "
                     "e.g. 'The TAT improvement claims are credible, several dealers said the same. "
                     "The 20% market share target is aspirational; treat as management's hope. "
                     "Nothing here changes our view yet; we need on-ground checks on the new sales force.'"))

    st.markdown("**Source material** (at least one)")
    transcript = _source_input("Transcript", "transcript", "Paste the transcript…")
    notes = _source_input("Detailed notes (SynthNotes Pro output)", "notes", "Paste the Q&A notes…")
    intel = _source_input("Intelligence brief (SynthNotes Pro output)", "intel", "Paste the intelligence brief…")

    source = build_source(transcript, notes, intel)
    run = st.button("Write the call write-up", type="primary", disabled=not source, use_container_width=True)

    if run:
        if not title.strip():
            st.warning("Give the call a title.")
            st.stop()
        header = f"{title.strip()} | {who.strip() or 'n/a'} | {when.strftime('%d %b %Y')} | {why.strip() or 'n/a'}"
        status = st.status("Writing…", expanded=True)
        live = st.empty()
        try:
            status.write("Stage 1 of 2: detailed section")
            detailed = generate(DETAILED_PROMPT.format(
                lo=DETAILED_BAND[0], hi=DETAILED_BAND[1], voice=VOICE_GUIDE, call_type=call_type,
                header=header, analyst_view=analyst_view.strip() or "(none given)", source=source),
                model, "detailed", live)
            detailed = clean_detailed(enforce_band(detailed, DETAILED_BAND, "the detailed section", source, model, status.write))

            status.write("Stage 2 of 2: summary on top")
            short = generate(SHORT_PROMPT.format(
                lo=SHORT_BAND[0], hi=SHORT_BAND[1], voice=VOICE_GUIDE, examples=STYLE_EXAMPLES,
                call_type=call_type, header=header,
                analyst_view=analyst_view.strip() or "(none given)", detailed=detailed),
                model, "summary", live)
            short = clean_summary(enforce_band(short, SHORT_BAND, "the summary", detailed + "\n\n" + (analyst_view or ""),
                                               model, status.write))
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            st.stop()

        head = f"**{title.strip()}**  \n*{call_type} · {who.strip() or 'n/a'} · {when.strftime('%d %b %Y')}*"
        if why.strip():
            head += f"  \n*Context: {why.strip()}*"
        assembled = f"{head}\n\n{short}\n\nDetails below:\n\n{detailed}"
        st.session_state["call_out"] = {
            "title": title.strip(), "when": when, "type": call_type, "who": who.strip(),
            "short": short, "detailed": detailed, "assembled": assembled, "source": source,
            "analyst_view": analyst_view, "header": header,
        }
        st.session_state["pending_dl"] = [(filename_for(title, "call_writeup", "md", when), assembled, "text/markdown")]
        status.update(label=f"Done · summary {word_count(short)} words · detail {word_count(detailed)} words",
                      state="complete", expanded=False)
        st.rerun()

    out = st.session_state.get("call_out")
    if not out:
        return

    pending = st.session_state.pop("pending_dl", None)
    if pending:
        auto_download(pending)
        st.success("Saved to your downloads: " + ", ".join(f[0] for f in pending))

    st.divider()
    output_block(out["assembled"], out["title"], "call_writeup", "call", out["when"])

    with st.expander("Refine", expanded=False):
        target = st.radio("Apply to", ["Summary", "Detailed section"], horizontal=True)
        instr = st.text_area("Instruction", height=90, key="call_refine",
                             placeholder="e.g. Lead with the asset-quality point. Cut the branding bullet. "
                                         "Make it clearer that the target is management's, not ours.")
        if st.button("Apply", key="call_refine_btn") and instr.strip():
            live = st.empty()
            try:
                if target == "Summary":
                    src = out["detailed"] + "\n\nANALYST'S VIEW:\n" + (out["analyst_view"] or "")
                    out["short"] = clean_summary(generate(REFINE_PROMPT.format(voice=VOICE_GUIDE, source=src,
                                                                 text=out["short"], instruction=instr),
                                            model, "refine summary", live))
                else:
                    out["detailed"] = clean_detailed(generate(REFINE_PROMPT.format(voice=VOICE_GUIDE, source=out["source"],
                                                                    text=out["detailed"], instruction=instr),
                                               model, "refine detail", live))
                head = out["assembled"].split("\n\n", 1)[0]
                out["assembled"] = f"{head}\n\n{out['short']}\n\nDetails below:\n\n{out['detailed']}"
                st.session_state["call_out"] = out
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.button("Send this write-up to the Weekly page", use_container_width=True):
        bucket = st.session_state.setdefault("weekly_calls", [])
        bucket.append({"name": out["title"], "text": out["assembled"]})
        st.success(f"Added. The Weekly page now has {len(bucket)} write-up(s).")


# ── 7. PAGE: WEEKLY ────────────────────────────────────────────────────────────

def page_weekly():
    api_key_check()
    model = sidebar_common()
    st.title("Weekly")
    st.caption("Two to five call write-ups from the week, plus how you want to frame it, "
               "in: a 750-850 word weekly in your voice.")

    bucket = st.session_state.setdefault("weekly_calls", [])
    if bucket:
        st.markdown("**Write-ups sent from the Call page**")
        for i, c in enumerate(list(bucket)):
            cols = st.columns([6, 1])
            cols[0].caption(f"{i + 1}. {c['name']} · {word_count(c['text']):,} words")
            if cols[1].button("Remove", key=f"rm_{i}"):
                bucket.pop(i)
                st.rerun()

    ups = st.file_uploader("Add call write-ups as files (.txt / .md / .docx / .pdf)",
                           type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)
    n_paste = st.number_input("Paste boxes", min_value=0, max_value=5, value=0 if (bucket or ups) else 2)
    pasted = []
    for i in range(int(n_paste)):
        t = st.text_area(f"Call write-up {i + 1}", height=150, key=f"wk_paste_{i}")
        if t.strip():
            pasted.append({"name": f"Pasted {i + 1}", "text": t.strip()})

    structure = st.radio("Structure", list(WEEKLY_STRUCTURES.keys()), index=0, horizontal=False,
                         help="Default merges every call into one write-up organised by topic, with each "
                              "source attributed inside the bullets. Pick per-company only when the calls "
                              "are about unrelated companies.")
    framing = st.text_area(
        "How you want to frame the week (optional)", height=120,
        placeholder=("The combined learning, what to lead with, what to play down, anything the CIO asked about. "
                     "e.g. 'Lead with CUBK; the Star Health piece is short. Combined point: both are early in "
                     "process changes and we will need ground checks before forming a view.'"))

    calls = list(bucket) + [{"name": u.name, "text": read_upload(u)} for u in (ups or [])] + pasted
    calls = [c for c in calls if c["text"].strip()]
    st.caption(f"{len(calls)} write-up(s) · {sum(word_count(c['text']) for c in calls):,} words of input")

    run = st.button("Write the weekly", type="primary", disabled=not calls, use_container_width=True)
    if run:
        joined = "\n\n".join(f"=== CALL WRITE-UP {i + 1}: {c['name']} ===\n{c['text']}" for i, c in enumerate(calls))
        status = st.status("Writing the weekly…", expanded=True)
        live = st.empty()
        try:
            combined = structure.startswith("One combined")
            weekly = generate(WEEKLY_PROMPT.format(
                n=len(calls), lo=WEEKLY_BAND[0], hi=WEEKLY_BAND[1], voice=VOICE_GUIDE,
                examples=STYLE_EXAMPLES, framing=framing.strip() or "(none given)", calls=joined,
                structure_block=WEEKLY_STRUCTURES[structure],
                budget_block=weekly_budget_block(1 if combined else len(calls), WEEKLY_BAND[1])),
                model, "weekly", live)
            weekly = enforce_band(weekly, WEEKLY_BAND, "the weekly", joined + "\n\nFRAMING:\n" + framing,
                                  model, status.write)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            st.stop()
        st.session_state["weekly_out"] = {"text": weekly, "source": joined, "framing": framing}
        st.session_state["pending_wk_dl"] = [(filename_for("Ritesh", "Weekly_draft", "md"), weekly, "text/markdown")]
        status.update(label=f"Done · {word_count(weekly)} words", state="complete", expanded=False)
        st.rerun()

    out = st.session_state.get("weekly_out")
    if not out:
        return
    pending = st.session_state.pop("pending_wk_dl", None)
    if pending:
        auto_download(pending)
        st.success("Saved to your downloads: " + ", ".join(f[0] for f in pending))

    st.divider()
    output_block(out["text"], "Ritesh", "Weekly_draft", "weekly")

    with st.expander("Refine", expanded=False):
        instr = st.text_area("Instruction", height=90, key="wk_refine",
                             placeholder="e.g. Shorten the Star Health section to three bullets. "
                                         "Soften the line on BSE market share. Add that we will do dealer checks next week.")
        if st.button("Apply", key="wk_refine_btn") and instr.strip():
            live = st.empty()
            try:
                src = out["source"] + "\n\nANALYST'S FRAMING:\n" + (out["framing"] or "")
                out["text"] = generate(REFINE_PROMPT.format(voice=VOICE_GUIDE, source=src, text=out["text"],
                                                            instruction=instr), model, "refine weekly", live)
                st.session_state["weekly_out"] = out
                st.rerun()
            except Exception as e:
                st.error(str(e))


# ── 8. MAIN ────────────────────────────────────────────────────────────────────

PAGE_CALL   = st.Page(page_call,   title="Call write-up", icon=":material/edit_note:", default=True)
PAGE_WEEKLY = st.Page(page_weekly, title="Weekly",        icon=":material/calendar_view_week:")


def main():
    st.set_page_config(page_title="SynthNotes Weekly", layout="wide", page_icon="✎")
    st.navigation([PAGE_CALL, PAGE_WEEKLY]).run()


if __name__ == "__main__":
    main()
