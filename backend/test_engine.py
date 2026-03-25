"""
Comprehensive test suite for TokenEngine.
Tests the core MLX inference pipeline: load → prefill → step → distribution.
Run: python test_engine.py
"""

import math
import sys
import time

from engine import TokenEngine

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = ""):
    results.append((name, condition, detail))
    mark = PASS if condition else FAIL
    msg = f"  {mark} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def test_model_load(engine: TokenEngine):
    print("\n── Model Load ──")
    check("engine.loaded is True", engine.loaded)
    info = engine.model_info
    check("model_info has model_path", info.get("model_path") == MODEL)
    check(
        "total_parameters > 1B",
        info.get("total_parameters", 0) > 1_000_000_000,
        f"{info.get('total_parameters', 0):,}",
    )
    check(
        "bits_per_weight in range",
        2.0 < info.get("bits_per_weight", 0) < 16.1,
        f"{info.get('bits_per_weight')}",
    )
    check(
        "vocab_size > 10k",
        (info.get("vocab_size") or 0) > 10_000,
        f"{info.get('vocab_size'):,}" if info.get("vocab_size") else "None",
    )


def test_init_session(engine: TokenEngine) -> dict:
    print("\n── Init Session (Prefill) ──")
    prompt = "The capital of France is"
    dist = engine.init_session(prompt, temperature=1.0, top_k=50)

    tokens = dist["tokens"]
    check("distribution has tokens", len(tokens) > 0, f"{len(tokens)} tokens")
    check("top_k=50 respected", len(tokens) == 50, f"got {len(tokens)}")
    seq_len = dist["sequence_length"]
    check("sequence_length matches prompt", seq_len > 0, f"{seq_len}")
    check("vocab_size > 100k", dist["vocab_size"] > 100_000, f"{dist['vocab_size']:,}")
    check("prefill_ms reported", dist.get("prefill_ms", 0) > 0, f"{dist.get('prefill_ms')} ms")
    prefill_tps = dist.get("prefill_tps", 0)
    check("prefill_tps reported", prefill_tps > 0, f"{prefill_tps} tok/s")

    top = tokens[0]
    check("top token has token_id", isinstance(top["token_id"], int))
    has_text = isinstance(top["text"], str) and len(top["text"]) > 0
    check("top token has text", has_text, repr(top["text"]))
    check("top token has probability", 0 < top["probability"] <= 1.0, f"{top['probability']:.4f}")
    check("top token has logit", isinstance(top["logit"], float), f"{top['logit']:.2f}")
    check("top token rank is 1", top["rank"] == 1)

    probs = [t["probability"] for t in tokens]
    check("probabilities sorted descending", probs == sorted(probs, reverse=True))
    check("all probabilities valid", all(0 < p <= 1.0 for p in probs))

    prob_sum = sum(probs)
    check(
        "top-50 probability mass is substantial",
        prob_sum > 0.3,
        f"sum={prob_sum:.4f}",
    )

    # "Paris" should be the most likely next token
    top_texts = [t["text"].strip().lower() for t in tokens[:5]]
    check(
        "'Paris' in top-5 predictions",
        any("paris" in t for t in top_texts),
        f"top-5: {[t['text'] for t in tokens[:5]]}",
    )

    check("history is empty after init", len(engine.history) == 0)
    return dist


def test_step(engine: TokenEngine, init_dist: dict) -> dict:
    print("\n── Step (Single Token Decode) ──")
    top_token = init_dist["tokens"][0]

    dist = engine.step(
        token_id=top_token["token_id"],
        probability=top_token["probability"],
        rank=top_token["rank"],
        temperature=1.0,
        top_k=50,
    )

    tokens = dist["tokens"]
    check("step returns tokens", len(tokens) == 50, f"{len(tokens)}")
    check("step_ms reported", dist.get("step_ms", 0) > 0, f"{dist.get('step_ms')} ms")
    check(
        "sequence_length increased by 1",
        dist["sequence_length"] == init_dist["sequence_length"] + 1,
        f"{init_dist['sequence_length']} → {dist['sequence_length']}",
    )

    history = engine.history
    check("history has 1 entry", len(history) == 1)
    check(
        "history entry matches selected token",
        history[0]["token_id"] == top_token["token_id"],
        f"selected: {top_token['text']!r}",
    )

    probs = [t["probability"] for t in tokens]
    check("step probs sorted descending", probs == sorted(probs, reverse=True))
    return dist


def test_multi_step(engine: TokenEngine, prev_dist: dict):
    print("\n── Multi-Step (3 more tokens, always pick top-1) ──")
    dist = prev_dist
    for _i in range(3):
        top = dist["tokens"][0]
        dist = engine.step(top["token_id"], top["probability"], top["rank"], 1.0, 50)

    check("history has 4 entries total", len(engine.history) == 4, f"{len(engine.history)}")
    generated = "".join(h["text"] for h in engine.history)
    check("generated text is non-empty", len(generated) > 0, repr(generated))
    check(
        "sequence grows correctly",
        dist["sequence_length"] == prev_dist["sequence_length"] + 3,
    )


def test_temperature_effect(engine: TokenEngine):
    print("\n── Temperature Effect ──")
    engine.reset()
    prompt = "Once upon a time"

    dist_low = engine.init_session(prompt, temperature=0.01, top_k=50)
    probs_low = [t["probability"] for t in dist_low["tokens"]]
    entropy_low = -sum(p * math.log(p + 1e-12) for p in probs_low)

    engine.reset()
    dist_high = engine.init_session(prompt, temperature=2.0, top_k=50)
    probs_high = [t["probability"] for t in dist_high["tokens"]]
    entropy_high = -sum(p * math.log(p + 1e-12) for p in probs_high)

    check(
        "low temp has lower entropy (more peaked)",
        entropy_low < entropy_high,
        f"H(T=0.01)={entropy_low:.3f}, H(T=2.0)={entropy_high:.3f}",
    )
    check(
        "low temp top-1 probability > high temp top-1",
        probs_low[0] > probs_high[0],
        f"T=0.01: {probs_low[0]:.4f}, T=2.0: {probs_high[0]:.4f}",
    )


def test_non_greedy_selection(engine: TokenEngine):
    print("\n── Non-Greedy Token Selection ──")
    engine.reset()
    prompt = "The weather today is"
    dist = engine.init_session(prompt, temperature=1.0, top_k=50)

    # Select the 5th-ranked token instead of top-1
    fifth = dist["tokens"][4]
    dist2 = engine.step(fifth["token_id"], fifth["probability"], fifth["rank"], 1.0, 50)

    check(
        "can select non-top token (rank 5)",
        engine.history[-1]["rank"] == 5,
        f"selected: {fifth['text']!r} (rank {fifth['rank']})",
    )
    check("step after non-greedy returns valid dist", len(dist2["tokens"]) == 50)


def test_reset(engine: TokenEngine):
    print("\n── Reset ──")
    engine.reset()
    check("history cleared", len(engine.history) == 0)
    check("cache cleared", engine._cache is None)
    check("tokens cleared", len(engine._tokens) == 0)


def test_performance(engine: TokenEngine):
    print("\n── Performance Benchmark ──")
    engine.reset()
    prompt = "Explain quantum computing in simple terms."
    dist = engine.init_session(prompt, temperature=1.0, top_k=200)

    prefill_tps = dist.get("prefill_tps", 0)
    check("prefill > 20 tok/s", prefill_tps > 20, f"{prefill_tps} tok/s")

    step_times = []
    for _ in range(10):
        top = dist["tokens"][0]
        dist = engine.step(top["token_id"], top["probability"], top["rank"], 1.0, 200)
        step_times.append(dist["step_ms"])

    avg_ms = sum(step_times) / len(step_times)
    check("avg step < 200ms", avg_ms < 200, f"{avg_ms:.1f} ms")
    check(
        "step throughput > 5 tok/s",
        1000 / avg_ms > 5,
        f"{1000 / avg_ms:.1f} tok/s",
    )
    print(f"  [i] Step times: {[f'{t:.0f}ms' for t in step_times]}")


def main():
    print("=" * 60)
    print("  Token Explorer — Engine Test Suite")
    print("=" * 60)

    print(f"\nModel: {MODEL}")
    engine = TokenEngine(MODEL)

    print("\nLoading model (this takes a few seconds)...")
    t0 = time.perf_counter()
    engine.load_model()
    load_s = time.perf_counter() - t0
    print(f"Model loaded in {load_s:.1f}s")

    test_model_load(engine)
    init_dist = test_init_session(engine)
    step_dist = test_step(engine, init_dist)
    test_multi_step(engine, step_dist)
    test_temperature_effect(engine)
    test_non_greedy_selection(engine)
    test_reset(engine)
    test_performance(engine)

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
