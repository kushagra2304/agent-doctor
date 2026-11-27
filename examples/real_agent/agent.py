"""A real tool-calling support agent on an open model (Hugging Face Inference Providers).

Unlike the bad agent, nothing is planted: the model decides which tools to call, in what
order and how often. Whatever waste Agent Doctor finds here is the model's own behaviour.
The tools are backed by small in-memory tables so the only external service is the LLM.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent_doctor import instrument, trace_run, trace_tool

MODEL = os.environ.get("AD_MODEL", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS = 8

SYSTEM_PROMPT = """You are ShopBot, the customer support assistant for an online store in India.
Answer the customer's question using the tools. Rules:
- Look facts up with tools; never guess order status, delivery dates, stock or policy.
- If a tool fails, say that the information is temporarily unavailable.
- When you have the answer, call send_reply exactly once with the final message for the
  customer, then reply with that same message.
- Keep replies to two or three friendly sentences."""

ORDERS = {
    "1234": {"status": "shipped", "carrier": "BlueDart", "items": ["running shoes"]},
    "5678": {"status": "shipped", "carrier": "Delhivery", "items": ["phone case", "charger"]},
    "9012": {"status": "packed", "carrier": "Ekart", "items": ["desk lamp"]},
}
ETAS = {"1234": "Friday", "5678": "Monday", "9012": "Wednesday"}
KB = {
    "refund": "Unused items can be returned within 30 days of delivery for a full refund.",
    "shipping": "Standard shipping takes 3-5 business days; express takes 1-2.",
    "warranty": "Electronics carry a 1-year manufacturer warranty.",
}
INVENTORY = {"SKU-RED-42": 12, "SKU-BLU-40": 0, "SKU-GRN-38": 3}


@trace_tool
def search_orders(order_id: str) -> dict:
    order = ORDERS.get(str(order_id).strip("# "))
    return {"order_id": order_id, "found": order is not None, **(order or {})}


@trace_tool
def get_tracking(order_id: str) -> dict:
    order_id = str(order_id).strip("# ")
    if order_id == "9012":  # this carrier's tracking API is down
        raise TimeoutError("Ekart tracking API timed out after 5 s")
    if order_id not in ETAS:
        return {"order_id": order_id, "error": "no tracking for this order"}
    return {"order_id": order_id, "eta": ETAS[order_id], "last_scan": "Pune sort centre"}


@trace_tool
def search_kb(query: str) -> dict:
    q = query.lower()
    hits = [{"topic": k, "text": v} for k, v in KB.items() if k in q or any(w in q for w in k.split())]
    return {"query": query, "results": hits or [{"topic": "none", "text": "No matching article."}]}


@trace_tool
def check_inventory(sku: str) -> dict:
    return {"sku": sku, "in_stock": INVENTORY.get(sku.upper(), 0)}


@trace_tool(side_effects=True)
def send_reply(customer_id: str, text: str) -> dict:
    return {"delivered": True, "customer_id": customer_id}


TOOLS = {f.__name__: f for f in (search_orders, get_tracking, search_kb, check_inventory, send_reply)}


def _schema(name: str, description: str, **params: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {p: {"type": "string", "description": d}
                                                         for p, d in params.items()},
                       "required": list(params)}}}


TOOL_SCHEMAS = [
    _schema("search_orders", "Look up an order: status, carrier and items.", order_id="The order number"),
    _schema("get_tracking", "Get the delivery estimate (ETA) for a shipped order.",
            order_id="The order number"),
    _schema("search_kb", "Search the help centre for store policies.", query="What to search for"),
    _schema("check_inventory", "How many units of a product are in stock.",
            sku="Product SKU, e.g. SKU-RED-42"),
    _schema("send_reply", "Send the final answer to the customer.", customer_id="The customer ID",
            text="The message to send"),
]


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _client():
    from huggingface_hub import InferenceClient

    return instrument(InferenceClient(model=MODEL), agent="shop-assistant")


_CLIENT = None


def client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _client()
    return _CLIENT


@trace_run(agent="shop-assistant", prompt_version="v1")
def handle(question: str, customer_id: str = "C-101") -> str:
    """Answer one customer question with a model-driven tool loop."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer {customer_id} asks: {question}"},
    ]
    answer = ""
    for _ in range(MAX_STEPS):
        resp = client().chat_completion(messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto",
                                        max_tokens=400, temperature=0)
        msg = _get(resp.choices[0], "message")
        calls = _get(msg, "tool_calls") or []
        answer = _get(msg, "content") or answer
        if not calls:
            break
        messages.append({"role": "assistant", "content": _get(msg, "content") or "", "tool_calls": [
            {"id": _get(c, "id"), "type": "function",
             "function": {"name": _get(_get(c, "function"), "name"),
                          "arguments": json.dumps(_args(c))}} for c in calls]})
        for c in calls:
            name, args = _get(_get(c, "function"), "name"), _args(c)
            try:
                result = TOOLS[name](**args) if name in TOOLS else {"error": f"unknown tool {name}"}
            except Exception as e:  # tell the model, like a real agent would
                result = {"error": f"{type(e).__name__}: {e}"}
            if name == "send_reply":
                answer = args.get("text", answer)
            messages.append({"role": "tool", "tool_call_id": _get(c, "id"), "name": name,
                             "content": json.dumps(result)})
    return answer.strip()


def _args(call: Any) -> dict[str, Any]:
    raw = _get(_get(call, "function"), "arguments")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except ValueError:
        return {}
