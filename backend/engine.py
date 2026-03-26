import math
import threading
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import compute_bits_per_weight, get_total_parameters

PREFILL_CHUNK = 512


class Interrupted(Exception):
    pass


class TokenEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._cache = None
        self._cached_tokens: list[int] = []
        self._last_logits = None
        self._lock = threading.Lock()
        self._gen = 0
        self._gen_lock = threading.Lock()

    def load_model(self):
        self.model, self.tokenizer = load(self.model_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _next_gen(self) -> int:
        with self._gen_lock:
            self._gen += 1
            return self._gen

    def _is_current(self, gen: int) -> bool:
        return self._gen == gen

    def predict(
        self,
        text: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        top_k: int = 20,
    ) -> dict | None:
        gen = self._next_gen()

        with self._lock:
            if not self._is_current(gen):
                return None

            tokens = self._encode(text, system_prompt)
            if not tokens:
                return {
                    "tokens": [],
                    "sequence_length": 0,
                    "vocab_size": self.tokenizer.vocab_size,
                }

            common = self._common_prefix_len(tokens)

            if common == len(tokens) == len(self._cached_tokens) and self._last_logits is not None:
                dist = self._build_distribution(self._last_logits, temperature, top_k)
                dist["cached"] = True
                return dist

            can_extend = (
                self._cache is not None
                and common > 0
                and common == len(self._cached_tokens)
                and common < len(tokens)
            )

            t0 = time.perf_counter()

            if can_extend:
                remaining = tokens[common:]
            else:
                self._cache = make_prompt_cache(self.model)
                remaining = tokens

            try:
                self._prefill(remaining, gen)
            except Interrupted:
                self.reset()
                return None

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._cached_tokens = list(tokens)
            dist = self._build_distribution(self._last_logits, temperature, top_k)

            if can_extend:
                dist["step_ms"] = round(elapsed_ms, 1)
            else:
                dist["prefill_ms"] = round(elapsed_ms, 1)
                dist["prefill_tps"] = (
                    round(len(tokens) / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0
                )
            return dist

    def _encode(self, text: str, system_prompt: str | None) -> list[int]:
        if system_prompt:
            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            if text:
                messages.append({"role": "user", "content": text})
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self.tokenizer.encode(text) if text else []

    def _common_prefix_len(self, tokens: list[int]) -> int:
        n = min(len(tokens), len(self._cached_tokens))
        for i in range(n):
            if tokens[i] != self._cached_tokens[i]:
                return i
        return n

    def _prefill(self, tokens: list[int], gen: int):
        arr = mx.array(tokens)
        total = len(arr)
        pos = 0

        while total - pos > 1:
            if not self._is_current(gen):
                raise Interrupted
            n = min(PREFILL_CHUNK, total - pos - 1)
            self.model(arr[pos : pos + n][None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
            pos += n

        if not self._is_current(gen):
            raise Interrupted

        logits = self.model(arr[pos:][None], cache=self._cache)
        self._last_logits = logits[0, -1, :]
        mx.eval(self._last_logits)
        mx.eval([c.state for c in self._cache])

    def reset(self):
        self._cache = None
        self._cached_tokens = []
        self._last_logits = None

    def _build_distribution(self, logits: mx.array, temperature: float, top_k: int) -> dict:
        temp = max(temperature, 1e-7)
        probs = mx.softmax(logits / temp)
        k = min(top_k, probs.shape[0])

        indices = mx.argpartition(-probs, kth=k - 1)[:k]
        top_p = probs[indices]
        top_l = logits[indices]

        order = mx.argsort(-top_p)
        indices, top_p, top_l = indices[order], top_p[order], top_l[order]
        mx.eval(indices, top_p, top_l)

        return {
            "tokens": [
                {
                    "id": int(indices[i].item()),
                    "text": self.tokenizer.decode([int(indices[i].item())]),
                    "probability": float(top_p[i].item()),
                    "logit": float(top_l[i].item()),
                    "rank": i + 1,
                }
                for i in range(k)
            ],
            "sequence_length": len(self._cached_tokens),
            "vocab_size": int(probs.shape[0]),
        }

    def score(
        self,
        user_message: str,
        assistant_reply: str,
        system_prompt: str | None = None,
        top_k: int = 5,
    ) -> dict | None:
        gen = self._next_gen()

        with self._lock:
            if not self._is_current(gen):
                return None

            prompt_msgs: list[dict] = []
            if system_prompt:
                prompt_msgs.append({"role": "system", "content": system_prompt})
            prompt_msgs.append({"role": "user", "content": user_message})

            full_msgs = [*prompt_msgs, {"role": "assistant", "content": assistant_reply}]

            prompt_tokens = self.tokenizer.apply_chat_template(
                prompt_msgs, add_generation_prompt=True
            )
            full_tokens = self.tokenizer.apply_chat_template(full_msgs, add_generation_prompt=False)

            reply_start = len(prompt_tokens)
            if reply_start >= len(full_tokens):
                return {
                    "tokens": [],
                    "total_log_prob": 0,
                    "avg_log_prob": 0,
                    "perplexity": 1,
                    "prompt_length": reply_start,
                    "reply_length": 0,
                    "elapsed_ms": 0,
                }

            t0 = time.perf_counter()
            try:
                token_results = self._score_forward(full_tokens, reply_start, top_k, gen)
            except Interrupted:
                self.reset()
                return None
            elapsed_ms = (time.perf_counter() - t0) * 1000

            log_probs = [t["log_prob"] for t in token_results]
            total_lp = sum(log_probs)
            n = len(log_probs)
            avg_lp = total_lp / n if n else 0

            return {
                "tokens": token_results,
                "total_log_prob": round(total_lp, 4),
                "avg_log_prob": round(avg_lp, 4),
                "perplexity": round(math.exp(-avg_lp), 4) if n else 1,
                "prompt_length": reply_start,
                "reply_length": n,
                "elapsed_ms": round(elapsed_ms, 1),
            }

    def _score_forward(
        self, tokens: list[int], reply_start: int, top_k: int, gen: int
    ) -> list[dict]:
        arr = mx.array(tokens)
        total = len(arr)
        cache = make_prompt_cache(self.model)
        pos = 0
        results = []

        while pos < total:
            if not self._is_current(gen):
                raise Interrupted

            end = min(pos + PREFILL_CHUNK, total)
            logits = self.model(arr[pos:end][None], cache=cache)
            mx.eval([c.state for c in cache])

            for j in range(end - pos):
                target_idx = pos + j + 1
                if target_idx < reply_start or target_idx >= total:
                    continue

                logit_vec = logits[0, j, :]
                target_id = tokens[target_idx]
                probs = mx.softmax(logit_vec)
                target_prob = probs[target_id]
                rank = (probs > target_prob).sum()

                k = min(top_k, probs.shape[0])
                top_idx = mx.argpartition(-probs, kth=k - 1)[:k]
                top_p = probs[top_idx]
                order = mx.argsort(-top_p)
                top_idx, top_p = top_idx[order], top_p[order]
                mx.eval(target_prob, rank, top_idx, top_p)

                prob_val = float(target_prob.item())
                results.append(
                    {
                        "id": int(target_id),
                        "text": self.tokenizer.decode([int(target_id)]),
                        "probability": prob_val,
                        "log_prob": math.log(max(prob_val, 1e-30)),
                        "rank": int(rank.item()) + 1,
                        "alternatives": [
                            {
                                "id": int(top_idx[i].item()),
                                "text": self.tokenizer.decode([int(top_idx[i].item())]),
                                "probability": float(top_p[i].item()),
                            }
                            for i in range(k)
                        ],
                    }
                )

            pos = end

        return results

    def tokenize(self, text: str) -> list[dict]:
        if not text:
            return []
        ids = self.tokenizer.encode(text)
        simple = [self.tokenizer.decode([tid]) for tid in ids]
        if "".join(simple) == text:
            return [{"id": int(tid), "text": s} for tid, s in zip(ids, simple, strict=True)]
        result = []
        decoded = ""
        for i, tid in enumerate(ids):
            full = self.tokenizer.decode(ids[: i + 1])
            result.append({"id": int(tid), "text": full[len(decoded) :]})
            decoded = full
        return result

    @property
    def model_info(self) -> dict:
        if not self.loaded:
            return {}
        return {
            "model_path": self.model_path,
            "total_parameters": get_total_parameters(self.model),
            "bits_per_weight": round(compute_bits_per_weight(self.model), 2),
            "vocab_size": self.tokenizer.vocab_size,
            "has_chat_template": (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template is not None
            ),
        }
