"""The runs that make up the fixture set: clean runs plus each planted problem, twice."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Faults


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    faults: Faults = field(default_factory=Faults)
    order_id: str | None = None
    sku: str | None = None
    customer_id: str = "C-101"
    description: str = ""


ORDER_QS = [("Where is my order #1234?", "1234"), ("When will order 5678 arrive?", "5678")]
REFUND_QS = ["Can I get a refund on my shoes?", "What is your return and refund window?"]
STOCK_QS = [("Is the red sneaker size 42 in stock?", "SKU-RED-42"),
            ("Is the green sneaker size 38 available?", "SKU-GRN-38")]


def _pairs(name, faults, kind, description):
    out = []
    for i in range(2):
        cust = ["C-101", "C-202"][i]
        if kind == "order":
            q, oid = ORDER_QS[i]
            out.append(Scenario(f"{name}-{i + 1}", q, faults, order_id=oid, customer_id=cust,
                                description=description))
        elif kind == "refund":
            out.append(Scenario(f"{name}-{i + 1}", REFUND_QS[i], faults, customer_id=cust,
                                description=description))
        else:
            q, sku = STOCK_QS[i]
            out.append(Scenario(f"{name}-{i + 1}", q, faults, sku=sku, customer_id=cust,
                                description=description))
    return out


SCENARIOS: list[Scenario] = [
    *_pairs("clean-order", Faults(), "order", "No planted problems."),
    *_pairs("clean-refund", Faults(), "refund", "No planted problems."),
    *_pairs("clean-stock", Faults(), "stock", "No planted problems."),
    *_pairs("duplicate-lookup", Faults(duplicate_lookup=True), "order",
            "Looks the order up twice with the same arguments."),
    *_pairs("tracking-retry-loop", Faults(tracking_failures=3), "order",
            "Tracking fails three times and is retried immediately, then works."),
    *_pairs("tracking-down", Faults(tracking_down=True), "order",
            "Tracking never works; the agent retries, then answers with a guess."),
    *_pairs("context-bloat", Faults(context_bloat=True), "order",
            "Sends the whole history, raw tool JSON included, on every call."),
    *_pairs("near-duplicate-search", Faults(near_duplicate_search=True), "refund",
            "Runs the same knowledge-base search twice in different words."),
    *_pairs("unused-tool", Faults(unused_tool=True), "refund",
            "Fetches the customer profile and never uses it."),
    *_pairs("sequential-lookups", Faults(sequential_independent=True), "refund",
            "Runs three independent lookups one after another instead of in parallel."),
    *_pairs("search-loop", Faults(search_loop=True), "refund",
            "A prompt line makes the agent re-search the same article three times."),
    *_pairs("slow-inventory", Faults(slow_inventory=True), "stock",
            "The inventory check takes 4.5 seconds."),
    *_pairs("big-model-routing", Faults(big_model_routing=True), "stock",
            "Uses the large model for a one-word routing decision."),
    *_pairs("no-prompt-cache", Faults(no_prompt_cache=True), "order",
            "A timestamp at the top of the system prompt breaks prompt caching."),
    *_pairs("llm-rate-limit", Faults(llm_rate_limit=3), "stock",
            "The first three LLM calls hit rate limits and are retried with no backoff."),
    *_pairs("everything-wrong", Faults(duplicate_lookup=True, tracking_failures=2,
                                       context_bloat=True, big_model_routing=True,
                                       no_prompt_cache=True), "order",
            "Several problems at once."),
]
