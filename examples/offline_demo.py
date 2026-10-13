"""Offline demo: record three runs of a small support agent with no network and no API key.

It uses a fake LLM client shaped like Hugging Face's InferenceClient, so it costs nothing.
The agent has the kinds of waste Agent Doctor is built to find: it repeats a tool call,
retries a flaky tool, and re-sends its whole history on every LLM call.

Run it, then look at what was recorded:

    python examples/offline_demo.py
    agent-doctor runs --prices examples/demo_pricing.yaml
    agent-doctor show <run id>
    agent-doctor cost <run id> --prices examples/demo_pricing.yaml
"""

from __future__ import annotations

import random
import time

from agent_doctor import instrument, trace_run, trace_tool


class FakeInferenceClient:
    """Returns canned answers; token counts grow with the prompt, like a real model."""

    model = "demo-model"
    provider = None

    def chat_completion(self, messages, *, model=None, max_tokens=200, **kwargs):
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        time.sleep(0.05 + prompt_chars / 200_000)  # longer prompts take longer
        last = str(messages[-1]["content"])
        if "Write the reply" in last:
            text = "Your order shipped yesterday and should arrive on Friday."
        else:
            text = "I should look up the order and its tracking."
        return {
            "model": model or self.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_chars // 4, "completion_tokens": len(text) // 4},
        }


client = instrument(FakeInferenceClient())

SYSTEM_PROMPT = "You are a support agent for an online shop. " + "Policy text. " * 300


@trace_tool
def search_orders(order_id: str) -> dict:
    time.sleep(0.12)
    return {"order_id": order_id, "status": "shipped", "carrier": "BlueDart"}


@trace_tool
def get_tracking(order_id: str, carrier: str) -> dict:
    time.sleep(0.15)
    if random.random() < 0.6:
        raise TimeoutError("carrier API timed out")
    return {"order_id": order_id, "eta": "Friday"}


@trace_tool(side_effects=True)
def send_reply(text: str) -> dict:
    time.sleep(0.02)
    return {"delivered": True}


@trace_run(agent="support-bot", agent_version="0.1.0", prompt_version="v1")
def handle(message: str) -> str:
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": message}]

    plan = client.chat_completion(messages=history)
    history.append(plan["choices"][0]["message"])

    order = search_orders("1234")
    history.append({"role": "user", "content": f"Tool result: {order}"})
    client.chat_completion(messages=history)

    order = search_orders("1234")  # planted waste: same call, same arguments

    tracking = None
    for _ in range(4):  # planted waste: retry with no backoff
        try:
            tracking = get_tracking(order_id="1234", carrier=order["carrier"])
            break
        except TimeoutError:
            continue
    history.append({"role": "user", "content": f"Tool result: {tracking}. Write the reply."})

    final = client.chat_completion(messages=history)  # planted waste: full history again
    text = final["choices"][0]["message"]["content"]
    send_reply(text)
    return text


if __name__ == "__main__":
    random.seed(7)
    for question in ["Where is my order #1234?", "Has order 1234 shipped?", "When will 1234 arrive?"]:
        print(f"> {question}\n  {handle(question)}")
    print("\nRecorded 3 runs. Next:\n"
          "  agent-doctor runs --prices examples/demo_pricing.yaml\n"
          "  agent-doctor cost <run id> --prices examples/demo_pricing.yaml")
