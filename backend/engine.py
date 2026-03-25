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
        self._tokens: list[int] = []
        self._history: list[dict] = []

    def load_model(self):
        self.model, self.tokenizer = load(self.model_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def init_session(self, prompt: str, temperature: float, top_k: int) -> dict:
        self._cache = make_prompt_cache(self.model)
        self._tokens = self.tokenizer.encode(prompt)
        self._history = []

        prompt_arr = mx.array(self._tokens)
        t0 = time.perf_counter()

        total = len(prompt_arr)
        processed = 0
        prefill_size = 2048

        while total - processed > 1:
            n = min(prefill_size, total - processed - 1)
            self.model(prompt_arr[processed : processed + n][None], cache=self._cache)
            mx.eval([c.state for c in self._cache])
            processed += n

        logits = self.model(prompt_arr[processed:][None], cache=self._cache)
        next_logits = logits[0, -1, :]
        mx.eval(next_logits)
        mx.eval([c.state for c in self._cache])

        elapsed_ms = (time.perf_counter() - t0) * 1000
        dist = self._build_distribution(next_logits, temperature, top_k)
        dist["prefill_ms"] = round(elapsed_ms, 1)
        dist["prefill_tps"] = round(total / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0
        return dist

    def step(
        self, token_id: int, probability: float, rank: int, temperature: float, top_k: int
    ) -> dict:
        self._history.append(
            {
                "token_id": token_id,
                "text": self.tokenizer.decode([token_id]),
                "probability": probability,
                "rank": rank,
            }
        )
        self._tokens.append(token_id)

        t0 = time.perf_counter()
        logits = self.model(mx.array([[token_id]]), cache=self._cache)
        next_logits = logits[0, -1, :]
        mx.eval(next_logits)
        mx.eval([c.state for c in self._cache])
        step_ms = (time.perf_counter() - t0) * 1000

        dist = self._build_distribution(next_logits, temperature, top_k)
        dist["step_ms"] = round(step_ms, 1)
        return dist

    def reset(self):
        self._cache = None
        self._tokens = []
        self._history = []

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
            "sequence_length": len(self._tokens),
            "vocab_size": int(probs.shape[0]),
        }

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def model_info(self) -> dict:
        if not self.loaded:
            return {}
        return {
            "model_path": self.model_path,
            "total_parameters": get_total_parameters(self.model),
            "bits_per_weight": round(compute_bits_per_weight(self.model), 2),
            "vocab_size": self.tokenizer.vocab_size,
        }
