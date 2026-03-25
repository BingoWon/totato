"""Unit tests for TokenEngine with the predict() API."""

import time

from test_helpers import check, summary

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_load(engine):
    print("\n== Model Loading ==")
    t0 = time.perf_counter()
    engine.load_model()
    elapsed = time.perf_counter() - t0
    check("model loads", engine.loaded)
    check("load < 30s", elapsed < 30, f"{elapsed:.1f}s")

    info = engine.model_info
    check("has model_path", "model_path" in info, info.get("model_path", ""))
    params = info.get("total_parameters", 0)
    check("total_parameters > 1B", params > 1_000_000_000, f"{params:,}")
    bpw = info.get("bits_per_weight", 0)
    check("bits_per_weight in range", 2.0 < bpw < 16.1, f"{bpw}")
    vs = info.get("vocab_size") or 0
    check("vocab_size > 10k", vs > 10_000, f"{vs:,}" if vs else "None")
    has_chat = info.get("has_chat_template", False)
    check("has_chat_template reported", isinstance(has_chat, bool), str(has_chat))


def test_predict_raw(engine):
    print("\n== Raw Completion Predict ==")
    dist = engine.predict("The capital of France is", top_k=50)
    tokens = dist["tokens"]
    check("has tokens", len(tokens) > 0, f"{len(tokens)} tokens")
    check("top_k=50 respected", len(tokens) == 50, f"got {len(tokens)}")
    seq_len = dist["sequence_length"]
    check("sequence_length > 0", seq_len > 0, f"{seq_len}")
    check("vocab_size > 100k", dist["vocab_size"] > 100_000, f"{dist['vocab_size']:,}")
    check("prefill_ms reported", dist.get("prefill_ms", 0) > 0, f"{dist.get('prefill_ms')} ms")
    prefill_tps = dist.get("prefill_tps", 0)
    check("prefill_tps reported", prefill_tps > 0, f"{prefill_tps} tok/s")

    top = tokens[0]
    check("top token has id", isinstance(top["id"], int))
    has_text = isinstance(top["text"], str) and len(top["text"]) > 0
    check("top token has text", has_text, repr(top["text"]))
    check(
        "top token has probability",
        0 < top["probability"] <= 1.0,
        f"{top['probability']:.4f}",
    )
    check("top token has logit", isinstance(top["logit"], float), f"{top['logit']:.2f}")
    check("top token rank is 1", top["rank"] == 1)

    probs = [t["probability"] for t in tokens]
    check("probabilities sorted desc", probs == sorted(probs, reverse=True))
    check("probabilities sum ~ 1", sum(probs) > 0.5, f"sum={sum(probs):.4f}")
    return dist


def test_incremental(engine):
    print("\n== Incremental Extension (KV Cache Reuse) ==")
    engine.reset()
    d1 = engine.predict("Hello", top_k=20)
    check("first predict has prefill_ms", d1.get("prefill_ms") is not None)

    d2 = engine.predict("Hello world", top_k=20)
    check("extension has step_ms", d2.get("step_ms") is not None)
    check("extension no prefill_ms", d2.get("prefill_ms") is None)
    check("sequence_length grew", d2["sequence_length"] > d1["sequence_length"])


def test_cache_reuse_same_text(engine):
    print("\n== Same Text Different Params (Cached Logits) ==")
    engine.reset()
    engine.predict("Testing cache", temperature=1.0, top_k=50)

    d2 = engine.predict("Testing cache", temperature=0.5, top_k=100)
    check("cached flag is True", d2.get("cached") is True)
    check("has 100 tokens", len(d2["tokens"]) == 100, f"got {len(d2['tokens'])}")


def test_edit_invalidates_cache(engine):
    print("\n== Text Edit Invalidates Cache ==")
    engine.reset()
    engine.predict("The quick brown fox", top_k=20)

    d2 = engine.predict("The quick red fox", top_k=20)
    check("edit triggers prefill", d2.get("prefill_ms") is not None)
    check("edit has no step_ms", d2.get("step_ms") is None)


def test_system_prompt(engine):
    print("\n== System Prompt (Chat Template) ==")
    engine.reset()
    dist = engine.predict(
        "What is 2+2?",
        system_prompt="You are a helpful math tutor.",
        top_k=50,
    )
    check("returns tokens", len(dist["tokens"]) > 0)
    check("has prefill_ms", dist.get("prefill_ms") is not None)
    seq_with_sys = dist["sequence_length"]
    check("sequence longer with sys prompt", seq_with_sys > 10, f"{seq_with_sys}")

    engine.reset()
    dist_no_sys = engine.predict("What is 2+2?", top_k=50)
    check(
        "sys prompt adds tokens",
        seq_with_sys > dist_no_sys["sequence_length"],
        f"{seq_with_sys} vs {dist_no_sys['sequence_length']}",
    )


def test_empty_text(engine):
    print("\n== Empty Text ==")
    dist = engine.predict("", top_k=50)
    check("empty text returns no tokens", len(dist["tokens"]) == 0)
    check("sequence_length is 0", dist["sequence_length"] == 0)


def test_temperature_effect(engine):
    print("\n== Temperature Effect ==")
    engine.reset()
    d_low = engine.predict("Once upon a time", temperature=0.01, top_k=10)
    engine.predict("Once upon a time", temperature=0.01, top_k=10)

    d_high = engine.predict("Once upon a time", temperature=2.0, top_k=10)
    top_low = d_low["tokens"][0]["probability"]
    top_high = d_high["tokens"][0]["probability"]
    check(
        "low temp -> sharper distribution",
        top_low > top_high,
        f"low={top_low:.4f} high={top_high:.4f}",
    )


def test_performance(engine):
    print("\n== Performance Benchmark ==")
    engine.reset()
    text = "The "
    step_times = []
    for _i in range(10):
        t0 = time.perf_counter()
        dist = engine.predict(text, top_k=20)
        step_times.append((time.perf_counter() - t0) * 1000)
        text += dist["tokens"][0]["text"]

    avg_ms = sum(step_times[1:]) / len(step_times[1:])
    check("avg step < 500ms", avg_ms < 500, f"{avg_ms:.1f} ms")
    check(
        "throughput > 2 tok/s",
        1000 / avg_ms > 2,
        f"{1000 / avg_ms:.1f} tok/s",
    )
    print(f"  [i] Step times: {[f'{t:.0f}ms' for t in step_times]}")
    print(f"  [i] Generated: {text!r}")


def test_tokenize(engine):
    print("\n== Tokenize ==")
    tokens = engine.tokenize("Hello world")
    check("returns tokens", len(tokens) > 0, f"{len(tokens)} tokens")
    reconstructed = "".join(t["text"] for t in tokens)
    check("reconstruction matches", reconstructed == "Hello world", repr(reconstructed))
    check("each token has id", all("id" in t for t in tokens))
    check("each token has text", all("text" in t for t in tokens))

    empty = engine.tokenize("")
    check("empty text returns []", empty == [])

    special = engine.tokenize("  \n\t")
    reconstructed_sp = "".join(t["text"] for t in special)
    check("whitespace reconstructs", reconstructed_sp == "  \n\t", repr(reconstructed_sp))


if __name__ == "__main__":
    from engine import TokenEngine

    eng = TokenEngine(MODEL)
    test_load(eng)
    test_predict_raw(eng)
    test_incremental(eng)
    test_cache_reuse_same_text(eng)
    test_edit_invalidates_cache(eng)
    test_system_prompt(eng)
    test_empty_text(eng)
    test_temperature_effect(eng)
    test_tokenize(eng)
    test_performance(eng)
    raise SystemExit(summary())
