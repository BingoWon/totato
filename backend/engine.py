import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import compute_bits_per_weight, get_total_parameters


class TokenEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._cache = None
        self._cached_tokens: list[int] = []
        self._last_logits = None

    def load_model(self):
        self.model, self.tokenizer = load(self.model_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(
        self,
        text: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        top_k: int = 200,
    ) -> dict:
        tokens = self._encode(text, system_prompt)
        if not tokens:
            return {
                "tokens": [],
                "sequence_length": 0,
                "vocab_size": self.tokenizer.vocab_size,
            }

        common = self._common_prefix_len(tokens)

        # Identical token sequence → reuse cached logits (fast path for param-only changes)
        if common == len(tokens) == len(self._cached_tokens) and self._last_logits is not None:
            dist = self._build_distribution(self._last_logits, temperature, top_k)
            dist["cached"] = True
            return dist

        # Strict extension of cached sequence → incremental decode
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

        self._prefill(remaining)
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

    def _prefill(self, tokens: list[int]):
        arr = mx.array(tokens)
        total = len(arr)
        pos = 0
        chunk = 2048

        while total - pos > 1:
            n = min(chunk, total - pos - 1)
            self.model(arr[pos : pos + n][None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
            pos += n

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
                    "token_id": int(indices[i].item()),
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

    def tokenize_text(self, text: str) -> list[dict]:
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
