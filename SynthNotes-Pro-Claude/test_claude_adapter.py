"""Offline test of SynthNotes-Pro-Claude's Claude adapter.

Real anthropic SDK + mock HTTP transport: checks the exact request body/headers the app
sends, and runs canned SSE streams through the app's own generate_with_retry /
stream_and_collect / usage tracking. No API key or network needed."""
import json, sys, importlib.util
import httpx, anthropic

APP = sys.argv[1]
captured = []


def sse(text, stop_reason="end_turn", in_tok=1200, out_tok=345):
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 1,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
    ]
    for piece in [text[i:i + 7] for i in range(0, len(text), 7)]:
        events.append(("content_block_delta", {"type": "content_block_delta", "index": 1,
                                               "delta": {"type": "text_delta", "text": piece}}))
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": out_tok}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


NEXT = {"text": "**Q1**\n- point one\n- point two", "stop": "end_turn"}


def handler(request: httpx.Request):
    captured.append({"headers": dict(request.headers), "body": json.loads(request.content)})
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content=sse(NEXT["text"], NEXT["stop"]))


spec = importlib.util.spec_from_file_location("app", APP)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
app._anthropic_key = "test"
app._claude = anthropic.Anthropic(api_key="test", max_retries=0,
                                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))
import streamlit as st

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# 1. Non-streaming call (as used by intelligence/refinement)
m = app.get_model("Claude Opus 5.5 (Best all-round)")
r = app.generate_with_retry(m, "Prompt A", stage="Intelligence")
check("non-stream text joined from text blocks only", r.text == NEXT["text"])
b, h = captured[-1]["body"], captured[-1]["headers"]
check("model id", b["model"] == "claude-opus-5-5")
check("max_tokens 128000", b["max_tokens"] == 128000)
check("effort high", b.get("output_config") == {"effort": "high"})
check("fallbacks default", b.get("fallbacks") == "default")
check("beta header", "server-side-fallback-2026-07-01" in h.get("anthropic-beta", ""))
check("no thinking param sent", "thinking" not in b)
check("streamed request", b.get("stream") is True)
check("prompt as user message", b["messages"] == [{"role": "user", "content": "Prompt A"}])
log = st.session_state["usage_log"]
check("usage recorded", log[-1] == {"model": "claude-opus-5-5", "stage": "Intelligence",
                                    "input_tokens": 1200, "output_tokens": 345})
check("cost uses Opus price", abs(app.compute_cost(1200, 345, "claude-opus-5-5") - (1200*4 + 345*20)/1e6) < 1e-12)

# 2. Streaming call (as used by notes)
m = app.get_model("Claude Fable 5.1 (Max quality)")
resp = app.generate_with_retry(m, "Prompt B", stream=True, stage="Notes")
text, tokens = app.stream_and_collect(resp)
check("stream text", text == NEXT["text"])
check("stream total tokens", tokens == 1545)
check("stream usage recorded", st.session_state["usage_log"][-1]["model"] == "claude-fable-5-1"
      and st.session_state["usage_log"][-1]["stage"] == "Notes")

# 3. Refusal raises a clear error
NEXT.update(text="", stop="refusal")
try:
    app.generate_with_retry(app.get_model("Claude Opus 5.5 (Best all-round)"), "x")
    check("refusal raises", False)
except RuntimeError as e:
    check("refusal raises", "declined" in str(e))

# 4. Transcription routes to Gemini list
check("audio model resolves to gemini", app.get_model("Gemini 3.7 Flash (Best all-round)").model_name == "gemini-3.7-flash")
check("defaults resolve", all(k in app.MODELS for k in [app.DEFAULT_NOTES_MODEL, app.DEFAULT_REFINE_MODEL,
      app.DEFAULT_INTEL_MODEL, app.DEFAULT_SUMMARY_MODEL, app.DEFAULT_ANALYSIS_MODEL])
      and app.DEFAULT_TRANSCRIPTION_MODEL in app.AUDIO_MODELS)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED")
