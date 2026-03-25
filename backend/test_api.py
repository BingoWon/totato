"""Integration tests for the FastAPI /api/predict endpoint."""

import json
import urllib.request

BASE = "http://localhost:8001"
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def api(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with _opener.open(req) as resp:
        return json.loads(resp.read())


def test_health():
    print("\n== GET /api/health ==")
    r = api("GET", "/api/health")
    check("status is ready", r.get("status") == "ready", r.get("status", ""))


def test_model():
    print("\n== GET /api/model ==")
    r = api("GET", "/api/model")
    check("has model_path", "model_path" in r, r.get("model_path", ""))
    params = r.get("total_parameters", 0)
    check("has total_parameters", params > 0, f"{params:,}")
    check("has bits_per_weight", r.get("bits_per_weight", 0) > 0, str(r.get("bits_per_weight")))
    has_chat = r.get("has_chat_template")
    check("has_chat_template reported", isinstance(has_chat, bool), str(has_chat))


def test_predict_basic():
    print("\n== POST /api/predict (basic) ==")
    r = api("POST", "/api/predict", {"text": "Hello!", "temperature": 1.0, "top_k": 20})
    dist = r["distribution"]
    check("returns distribution", "tokens" in dist, f"{len(dist['tokens'])} tokens")
    check("top_k=20 respected", len(dist["tokens"]) == 20, f"got {len(dist['tokens'])}")
    check("has sequence_length", dist["sequence_length"] > 0, str(dist["sequence_length"]))
    check("has vocab_size", dist["vocab_size"] > 10_000, f"{dist['vocab_size']:,}")

    top = dist["tokens"][0]
    required = ("token_id", "text", "probability", "logit", "rank")
    check("token has all fields", all(k in top for k in required))
    check("top rank is 1", top["rank"] == 1)
    check("probability in (0,1]", 0 < top["probability"] <= 1.0, f"{top['probability']:.4f}")


def test_predict_with_system_prompt():
    print("\n== POST /api/predict (system prompt) ==")
    r = api(
        "POST",
        "/api/predict",
        {
            "text": "What is 2+2?",
            "system_prompt": "You are a math tutor.",
            "temperature": 0.5,
            "top_k": 10,
        },
    )
    dist = r["distribution"]
    check("returns tokens", len(dist["tokens"]) > 0)
    check("sequence longer with sys prompt", dist["sequence_length"] > 5)


def test_predict_empty():
    print("\n== POST /api/predict (empty text) ==")
    r = api("POST", "/api/predict", {"text": ""})
    dist = r["distribution"]
    check("no tokens for empty text", len(dist["tokens"]) == 0)
    check("sequence_length is 0", dist["sequence_length"] == 0)


def test_predict_incremental():
    print("\n== POST /api/predict (incremental) ==")
    api("POST", "/api/reset")
    r1 = api("POST", "/api/predict", {"text": "The sky is", "top_k": 10})
    check("first has prefill_ms", r1["distribution"].get("prefill_ms") is not None)

    r2 = api("POST", "/api/predict", {"text": "The sky is blue", "top_k": 10})
    check("extension has step_ms", r2["distribution"].get("step_ms") is not None)


def test_predict_cached_logits():
    print("\n== POST /api/predict (cached logits) ==")
    api("POST", "/api/reset")
    api("POST", "/api/predict", {"text": "cache test", "temperature": 1.0, "top_k": 50})
    r2 = api("POST", "/api/predict", {"text": "cache test", "temperature": 0.5, "top_k": 100})
    check("cached flag", r2["distribution"].get("cached") is True)
    check("100 tokens returned", len(r2["distribution"]["tokens"]) == 100)


def test_reset():
    print("\n== POST /api/reset ==")
    r = api("POST", "/api/reset")
    check("reset returns ok", r.get("status") == "ok")


def test_tokenize():
    print("\n== POST /api/tokenize ==")
    r = api("POST", "/api/tokenize", {"text": "Hello world"})
    tokens = r["tokens"]
    check("returns tokens", len(tokens) > 0, f"{len(tokens)} tokens")
    reconstructed = "".join(t["text"] for t in tokens)
    check("reconstruction matches", reconstructed == "Hello world", repr(reconstructed))
    check("each has id and text", all("id" in t and "text" in t for t in tokens))

    r2 = api("POST", "/api/tokenize", {"text": ""})
    check("empty text returns []", r2["tokens"] == [])


if __name__ == "__main__":
    test_health()
    test_model()
    test_tokenize()
    test_predict_basic()
    test_predict_with_system_prompt()
    test_predict_empty()
    test_predict_incremental()
    test_predict_cached_logits()
    test_reset()

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 40}")
    raise SystemExit(1 if failed else 0)
