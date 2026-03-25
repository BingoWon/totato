"""
API integration test — requires the backend server running on localhost:8000.
Start server first:  python server.py
Then run:            python test_api.py
"""

import json
import sys
import urllib.request

BASE = "http://localhost:8001"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results: list[tuple[str, bool, str]] = []

# Bypass macOS system proxy for localhost
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(name: str, condition: bool, detail: str = ""):
    results.append((name, condition, detail))
    mark = PASS if condition else FAIL
    msg = f"  {mark} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


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
    print("\n── Health ──")
    r = api("GET", "/api/health")
    check("health returns ready", r.get("status") == "ready", r.get("status", ""))


def test_model():
    print("\n── Model Info ──")
    r = api("GET", "/api/model")
    check("has model_path", "model_path" in r, r.get("model_path", ""))
    params = r.get("total_parameters", 0)
    check("has total_parameters", params > 0, f"{params:,}")
    check("has bits_per_weight", r.get("bits_per_weight", 0) > 0, str(r.get("bits_per_weight")))


def test_init() -> dict:
    print("\n── POST /api/init ──")
    r = api("POST", "/api/init", {"prompt": "Hello, world!", "temperature": 1.0, "top_k": 20})
    dist = r["distribution"]
    check("returns distribution", "tokens" in dist, f"{len(dist['tokens'])} tokens")
    check("returns history", "history" in r, f"{len(r['history'])} entries")
    check("history empty on init", len(r["history"]) == 0)
    check("top_k=20 respected", len(dist["tokens"]) == 20, f"got {len(dist['tokens'])}")
    check("has sequence_length", dist["sequence_length"] > 0, str(dist["sequence_length"]))
    check("has vocab_size", dist["vocab_size"] > 100_000, f"{dist['vocab_size']:,}")

    top = dist["tokens"][0]
    required = ("token_id", "text", "probability", "logit", "rank")
    check("token has all fields", all(k in top for k in required))
    check("top rank is 1", top["rank"] == 1)
    check("probability in (0,1]", 0 < top["probability"] <= 1.0, f"{top['probability']:.4f}")
    return r


def test_step(init_resp: dict) -> dict:
    print("\n── POST /api/step ──")
    top = init_resp["distribution"]["tokens"][0]
    r = api(
        "POST",
        "/api/step",
        {
            "token_id": top["token_id"],
            "probability": top["probability"],
            "rank": top["rank"],
            "temperature": 1.0,
            "top_k": 20,
        },
    )
    dist = r["distribution"]
    check("returns distribution", len(dist["tokens"]) == 20)
    check("history grew by 1", len(r["history"]) == 1)
    check(
        "history token matches",
        r["history"][0]["token_id"] == top["token_id"],
        repr(r["history"][0]["text"]),
    )
    check("step_ms reported", dist.get("step_ms", 0) > 0, f"{dist.get('step_ms')} ms")
    return r


def test_multi_step():
    print("\n── Multi-Step Sequence ──")
    r = api("POST", "/api/init", {"prompt": "1+1=", "temperature": 0.01, "top_k": 10})
    texts = []
    for _ in range(5):
        top = r["distribution"]["tokens"][0]
        texts.append(top["text"])
        r = api(
            "POST",
            "/api/step",
            {
                "token_id": top["token_id"],
                "probability": top["probability"],
                "rank": top["rank"],
                "temperature": 0.01,
                "top_k": 10,
            },
        )
    check("5 greedy steps work", len(r["history"]) == 5)
    generated = "".join(texts)
    check("generated text non-empty", len(generated) > 0, repr(generated))


def test_reset():
    print("\n── POST /api/reset ──")
    r = api("POST", "/api/reset")
    check("reset returns ok", r.get("status") == "ok")

    r2 = api("POST", "/api/init", {"prompt": "Test", "temperature": 1.0, "top_k": 10})
    check("fresh session after reset", len(r2["history"]) == 0)


def main():
    print("=" * 60)
    print("  Token Explorer — API Integration Test")
    print(f"  Server: {BASE}")
    print("=" * 60)

    try:
        api("GET", "/api/health")
    except Exception as e:
        print(f"\n  {FAIL} Cannot connect to server at {BASE}")
        print("    Start it first: cd backend && python server.py")
        print(f"    Error: {e}")
        sys.exit(1)

    test_health()
    test_model()
    init_resp = test_init()
    test_step(init_resp)
    test_multi_step()
    test_reset()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(results)} total")
    print(f"{'=' * 60}")

    if failed:
        print("\nFailed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  {FAIL} {name}  ({detail})")
        sys.exit(1)
    else:
        print("\n  All tests passed!")


if __name__ == "__main__":
    main()
