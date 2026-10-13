"""A fake LLM provider: deterministic answers, realistic token counts, latency and caching.

It is shaped like Hugging Face's InferenceClient (``chat_completion``), so
``agent_doctor.instrument`` records it like the real thing. Time passes on the active
clock, so under ``agent_doctor.testing.simulated_time`` no real time is spent.
"""

from __future__ import annotations

import re

from agent_doctor.testing import sleep

# name: (fixed latency s, s per input token, s per output token)
MODELS = {
    "demo-large": (0.35, 0.00025, 0.02),
    "demo-small": (0.08, 0.00005, 0.005),
}
CACHE_MIN_CHARS = 4000  # like real providers, only long shared prefixes are cached


class RateLimitError(Exception):
    """HTTP 429 from the provider."""

    status_code = 429


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _prompt_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class FakeProvider:
    """Shared across runs, like a real provider: holds the prompt cache."""

    def __init__(self) -> None:
        self.recent_prompts: list[str] = []

    def cached_chars(self, prompt: str) -> int:
        best = max((_common_prefix(prompt, p) for p in self.recent_prompts), default=0)
        self.recent_prompts = (self.recent_prompts + [prompt])[-20:]
        return best if best >= CACHE_MIN_CHARS else 0


class FakeInferenceClient:
    """One client per agent run. ``rate_limit_failures`` makes the first N calls fail with 429."""

    provider = "demo"

    def __init__(self, provider: FakeProvider, model: str = "demo-large", rate_limit_failures: int = 0):
        self._provider = provider
        self.model = model
        self._rate_limit_left = rate_limit_failures
        self._decide_calls = 0

    def chat_completion(self, messages, *, model=None, max_tokens=300, **kwargs):
        model = model or self.model
        fixed, per_in, per_out = MODELS[model]
        if self._rate_limit_left > 0:
            self._rate_limit_left -= 1
            sleep(0.05)
            raise RateLimitError("429 Too Many Requests: rate limit reached, retry later")

        prompt = _prompt_text(messages)
        cached_chars = self._provider.cached_chars(prompt)
        cached = len(prompt[:cached_chars]) // 4
        text = self._respond(messages, prompt)
        n_in, n_out = tokens(prompt), tokens(text)
        cached = min(cached, n_in)
        sleep(fixed + per_in * (n_in - cached) + per_out * n_out)
        usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}
        if cached:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return {
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": usage,
        }

    # -- the "reasoning" ----------------------------------------------------------
    def _respond(self, messages: list[dict], prompt: str) -> str:
        task = str(messages[-1]["content"])
        question = _find(r"Question: (.+)", prompt) or ""
        if task.startswith("TASK: route"):
            q = question.lower()
            if "refund" in q or "return" in q:
                return "refund"
            if "stock" in q or "available" in q:
                return "stock"
            return "order_status"
        if task.startswith("TASK: decide"):
            self._decide_calls += 1
            if "Always search again" in prompt and self._decide_calls <= 3:
                return "search_more"
            if "tracking" in task:
                return "need_tracking"
            return "done"
        if task.startswith("TASK: answer"):
            return answer_from_facts(question, prompt)
        return "OK."


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def answer_from_facts(question: str, context: str) -> str:
    """What a careful model would say given these facts; falls back to a vague guess."""
    q = question.lower()
    if "refund" in q or "return" in q:
        days = _find(r'"refund_days":\s*(\d+)', context) or _find(r"refund_days=(\d+)", context)
        if days:
            return f"You can get a full refund within {days} days of delivery."
        return "Refunds are usually possible within a few weeks."
    if "stock" in q or "available" in q:
        n = _find(r'"in_stock":\s*(\d+)', context) or _find(r"in_stock=(\d+)", context)
        if n is None:
            return "It should be available soon."
        return f"Yes, {n} in stock." if int(n) > 0 else "Sorry, it is out of stock right now."
    order = _find(r"#?(\d{4})", question) or "your order"
    eta = _find(r'"eta":\s*"([^"]+)"', context) or _find(r"eta=([A-Za-z]+)", context)
    if eta:
        return f"Order {order} has shipped and should arrive on {eta}."
    return f"Order {order} is on its way and should arrive in 3-5 days."
