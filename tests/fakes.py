"""Fake LLM clients shaped like the real ones, so tests cost nothing."""

from __future__ import annotations

from typing import Any


def chat_response(text="ok", prompt=10, completion=5, model="demo-model", cached=None,
                  tool_calls=None, finish="stop") -> dict:
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion,
                             "total_tokens": prompt + completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": "{}"}}
            for i, n in enumerate(tool_calls)
        ]
    return {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage}


def stream_chunks(parts, prompt=10, completion=5, model="demo-model", with_usage=True) -> list:
    chunks = [{"model": model, "choices": [{"index": 0, "delta": {"content": p},
                                            "finish_reason": None}]} for p in parts]
    chunks[-1]["choices"][0]["finish_reason"] = "stop"
    if with_usage:
        chunks.append({"model": model, "choices": [],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})
    return chunks


class _Proxy:
    def __init__(self, client):
        self._client = client

    @property
    def completions(self):
        return _ProxyCompletions(self._client)


class _ProxyCompletions:
    def __init__(self, client):
        self._client = client

    @property
    def create(self):  # like huggingface_hub: resolved at access time
        return self._client.chat_completion


class FakeHFClient:
    """Mimics huggingface_hub.InferenceClient.chat_completion."""

    def __init__(self, responses, model="demo-model", provider=None):
        self.responses = list(responses)
        self.model = model
        self.provider = provider
        self.calls: list[dict] = []

    @property
    def chat(self):
        return _Proxy(self)

    def chat_completion(self, messages, *, model=None, stream=False, **kwargs):
        self.calls.append({"messages": messages, "model": model, "stream": stream, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return iter(item) if stream else item


class FakeAsyncHFClient(FakeHFClient):
    async def chat_completion(self, messages, *, model=None, stream=False, **kwargs):
        self.calls.append({"messages": messages, "model": model, "stream": stream})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if stream:
            async def gen():
                for c in item:
                    yield c
            return gen()
        return item


class _Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        return self.owner.responses.pop(0)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = _Chat(self)


class _Messages:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        return self.owner.responses.pop(0)


class FakeAnthropicClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.messages = _Messages(self)


def anthropic_response(text="hi", input_tokens=100, output_tokens=20, cache_read=0,
                       cache_write=0, model="claude-haiku-5-5", tool=None) -> dict:
    content = [{"type": "text", "text": text}]
    if tool:
        content.append({"type": "tool_use", "id": "t1", "name": tool, "input": {}})
    return {"model": model, "content": content, "stop_reason": "end_turn",
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                      "cache_read_input_tokens": cache_read,
                      "cache_creation_input_tokens": cache_write}}
