"""Optional LLM backends for writing the fix plan: ``"provider:model"`` -> ``prompt -> text``.

The clients are created without ``instrument()``, so the plan-writing call is never saved
to your trace database as if it were one of your agent's runs.
"""

from __future__ import annotations

from agent_doctor.recommender.recommend import LLM

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}


def llm_from_spec(spec: str) -> LLM:
    """``anthropic[:model]``, ``openai:model`` or ``hf:model`` (Hugging Face Inference Providers).

    Credentials come from each SDK's usual environment variables (``ANTHROPIC_API_KEY``,
    ``OPENAI_API_KEY``, ``HF_TOKEN``).
    """
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip() or DEFAULT_MODELS.get(provider, "")
    if not model:
        raise ValueError(f"name a model for {provider!r}, e.g. {provider}:MODEL_ID")
    if provider == "anthropic":
        return _anthropic(model)
    if provider == "openai":
        return _openai(model)
    if provider in {"hf", "huggingface"}:
        return _huggingface(model)
    raise ValueError(f"unknown LLM provider {provider!r}; use anthropic, openai or hf")


def _need(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError:
        raise ImportError(f"pip install {extra} to write fix plans with this provider") from None


def _anthropic(model: str) -> LLM:
    anthropic = _need("anthropic", "anthropic")
    client = anthropic.Anthropic()

    def complete(prompt: str) -> str:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            # If the model declines, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined to write a fix plan")
        return "".join(b.text for b in response.content if b.type == "text").strip()

    return complete


def _openai(model: str) -> LLM:
    openai = _need("openai", "openai")
    client = openai.OpenAI()

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}])
        return (response.choices[0].message.content or "").strip()

    return complete


def _huggingface(model: str) -> LLM:
    hub = _need("huggingface_hub", '"agentdoctor[hf]"')
    client = hub.InferenceClient(model=model)

    def complete(prompt: str) -> str:
        response = client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=1200)
        return (response.choices[0].message.content or "").strip()

    return complete
