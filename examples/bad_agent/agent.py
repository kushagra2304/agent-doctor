"""A customer-support agent with switchable, deliberately planted problems.

Each ``Faults`` flag plants one kind of waste or unreliability that Agent Doctor's
analyzers must find. When a fault fires, the agent writes down which spans prove it
(the "planted" list), which becomes the answer key for the analyzer tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_doctor import instrument
from agent_doctor._util import now_ms
from agent_doctor.schema import FindingType, Run
from agent_doctor.sdk import RunRecorder
from agent_doctor.store import TraceStore

from .fake_llm import FakeInferenceClient, FakeProvider, RateLimitError, answer_from_facts
from .tools import INVENTORY, ORDERS, REFUND_ARTICLE, ServiceDown, ToolFaults, Tools

AGENT_NAME = "support-bot"
LARGE, SMALL = "demo-large", "demo-small"

_RULES = [
    "Greet the customer by name when you know it.",
    "Never promise a delivery date you have not checked.",
    "Quote refund windows exactly as the policy states them.",
    "Do not share internal order notes with the customer.",
    "Keep replies under three sentences unless asked for detail.",
    "Escalate threats, legal questions and safety issues to a human.",
    "Use the customer's language and a friendly, plain tone.",
    "If a tool fails, say so honestly instead of guessing.",
]
SYSTEM_PROMPT = "You are ShopBot, the support agent for an online store in India.\n" + "\n".join(
    f"Rule {i + 1}.{j + 1}: {rule}" for i in range(8) for j, rule in enumerate(_RULES)
)


@dataclass
class Faults:
    duplicate_lookup: bool = False       # search_orders twice with the same arguments
    near_duplicate_search: bool = False  # the same knowledge-base search, reworded
    unused_tool: bool = False            # fetch the customer profile, never use it
    tracking_failures: int = 0           # tracking API fails N times, retried with no backoff
    tracking_down: bool = False          # tracking API always fails; the agent answers anyway
    search_loop: bool = False            # prompt makes the agent re-search with no new results
    sequential_independent: bool = False # three independent lookups run one after another
    slow_inventory: bool = False         # inventory check takes 4.5 s
    context_bloat: bool = False          # full history, raw tool JSON included, sent every call
    big_model_routing: bool = False      # the large model classifies the question
    no_prompt_cache: bool = False        # a timestamp at the top of the prompt breaks caching
    llm_rate_limit: int = 0              # first N LLM calls hit 429, retried with no backoff

    def active(self) -> list[str]:
        return [k for k, v in self.__dict__.items() if v]


@dataclass
class Planted:
    type: FindingType
    kind: str
    detail: str
    evidence_span_ids: list[str]

    def as_dict(self) -> dict:
        return {"type": self.type.value, "kind": self.kind, "detail": self.detail,
                "evidence_span_ids": self.evidence_span_ids}


@dataclass
class AgentResult:
    answer: str
    run: Run
    planted: list[Planted] = field(default_factory=list)


def _compact(tool: str, out: dict) -> str:
    if tool == "search_orders":
        return f"order {out['order_id']} status={out.get('status')} carrier={out.get('carrier')}"
    if tool == "get_tracking":
        return f"eta={out['eta']}"
    if tool in ("search_kb", "read_article"):
        art = out["results"][0] if "results" in out else out
        return f"refund_days={art['refund_days']}"
    if tool == "get_customer":
        return f"customer={out['name']} tier={out['tier']}"
    if tool == "get_promotions":
        return f"promo={out['promo_code']}"
    if tool == "check_inventory":
        return f"in_stock={out['in_stock']}"
    return json.dumps(out)


class SupportAgent:
    def __init__(self, provider: FakeProvider, faults: Faults | None = None,
                 store: TraceStore | None = None, agent_version: str = "0.3.0"):
        self.provider = provider
        self.f = faults or Faults()
        self.store = store
        self.agent_version = agent_version

    # -- one run ------------------------------------------------------------------
    def run(self, question: str, *, customer_id: str = "C-101", order_id: str | None = None,
            sku: str | None = None, tags: dict | None = None) -> AgentResult:
        f = self.f
        self.client = instrument(
            FakeInferenceClient(self.provider, rate_limit_failures=f.llm_rate_limit),
            agent=AGENT_NAME,
        )
        self.tools = Tools(ToolFaults(f.tracking_failures, f.tracking_down, f.slow_inventory))
        self.question = question
        self.history: list[dict] = [{"role": "user", "content": f"Question: {question}"}]
        self.facts: dict[str, str] = {}
        self.planted: list[Planted] = []
        self.llm_ids: list[str] = []
        self._swallowed: list[str] | None = None

        with RunRecorder(AGENT_NAME, task_input=question, agent_version=self.agent_version,
                         prompt_version="v1-loop" if f.search_loop else "v1",
                         tags=tags, store=self.store, save=self.store is not None) as rec:
            self.rec = rec
            intent = self._route()
            if intent == "order_status":
                self._order_status(order_id or "1234")
            elif intent == "refund":
                self._refund(customer_id)
            else:
                self._stock(sku or "SKU-RED-42")
            answer = self._llm("TASK: answer the customer", LARGE)
            answer_span = self.llm_ids[-1]
            self.tools.send_reply(customer_id, answer)
            rec.final_output = answer

            if self._swallowed:
                self._plant(FindingType.ERROR, "swallowed_tool_failure",
                            "every get_tracking call failed, yet the agent answered with a guessed date",
                            self._swallowed + [answer_span])
            if f.context_bloat:
                self._plant(FindingType.CONTEXT_BLOAT, "context_bloat",
                            "full history with raw tool JSON is re-sent on every LLM call",
                            list(self.llm_ids))
            if f.no_prompt_cache:
                self._plant(FindingType.COST, "prompt_cache_miss",
                            "a timestamp at the start of the system prompt prevents prompt caching",
                            list(self.llm_ids))
        assert rec.run is not None
        return AgentResult(answer, rec.run, self.planted)

    # -- helpers ------------------------------------------------------------------
    def _last(self) -> str:
        return self.rec.spans[-1].span_id

    def _plant(self, type_: FindingType, kind: str, detail: str, evidence: list[str]) -> None:
        self.planted.append(Planted(type_, kind, detail, evidence))

    def _system(self) -> str:
        text = SYSTEM_PROMPT
        if self.f.search_loop:
            text += "\nAlways search again to be thorough before answering."
        if self.f.no_prompt_cache:
            stamp = datetime.fromtimestamp(now_ms() / 1000, tz=timezone.utc).isoformat()
            text = f"Current time: {stamp}\n" + text
        return text

    def _observe(self, tool: str, out: dict) -> None:
        if self.f.context_bloat:
            self.history.append({"role": "user", "content": f"Tool result {tool}: {json.dumps(out)}"})
        else:
            self.facts[tool] = _compact(tool, out)

    def _llm(self, task: str, model: str) -> str:
        system = {"role": "system", "content": self._system()}
        if self.f.context_bloat:
            self.history.append({"role": "user", "content": task})
            messages = [system, *self.history]
        else:
            facts = "Facts: " + "; ".join(self.facts.values()) if self.facts else "Facts: none"
            messages = [system, {"role": "user", "content": f"Question: {self.question}"},
                        {"role": "user", "content": f"{task}\n{facts}"}]
        failed: list[str] = []
        for _ in range(6):  # retries immediately: no backoff
            try:
                resp = self.client.chat_completion(messages=messages, model=model, max_tokens=200)
                break
            except RateLimitError:
                failed.append(self._last())
        else:
            raise RuntimeError("LLM unavailable")
        self.llm_ids.append(self._last())
        if len(failed) >= 2:
            self._plant(FindingType.RETRY_LOOP, "llm_retry_without_backoff",
                        f"{len(failed)} rate-limited LLM calls retried immediately with no backoff",
                        failed + [self._last()])
        text = resp["choices"][0]["message"]["content"]
        if self.f.context_bloat:
            self.history.append({"role": "assistant", "content": text})
        return text

    # -- flows --------------------------------------------------------------------
    def _route(self) -> str:
        model = LARGE if self.f.big_model_routing else SMALL
        intent = self._llm("TASK: route the question to order_status, refund or stock", model).strip()
        if self.f.big_model_routing:
            self._plant(FindingType.COST, "expensive_model_for_simple_step",
                        f"the large model ({LARGE}) is used for a one-word routing decision",
                        [self.llm_ids[-1]])
        return intent

    def _order_status(self, order_id: str) -> None:
        order = self.tools.search_orders(order_id)
        first = self._last()
        self._observe("search_orders", order)
        if self.f.duplicate_lookup:
            order = self.tools.search_orders(order_id)
            self._plant(FindingType.REDUNDANT_TOOL_CALL, "exact_duplicate_tool_call",
                        f"search_orders called twice with order_id={order_id}", [first, self._last()])
        self._llm("TASK: decide whether tracking is needed", LARGE)

        attempts: list[str] = []
        for _ in range(4):
            try:
                tracking = self.tools.get_tracking(order_id=order_id, carrier=order["carrier"])
                attempts.append(self._last())
                self._observe("get_tracking", tracking)
                break
            except ServiceDown:
                attempts.append(self._last())
        if len(attempts) >= 3:
            ok = not self.f.tracking_down
            self._plant(FindingType.RETRY_LOOP, "tool_retry_loop",
                        f"get_tracking retried {len(attempts) - 1} times with no backoff"
                        + ("" if ok else ", and never succeeded"), attempts)
        if self.f.tracking_down:
            self._swallowed = attempts

    def _refund(self, customer_id: str) -> None:
        if self.f.sequential_independent:
            ids = []
            self._observe("get_customer", self.tools.get_customer(customer_id))
            ids.append(self._last())
            self._observe("get_promotions", self.tools.get_promotions("en-IN"))
            ids.append(self._last())
            self._observe("search_kb", self.tools.search_kb("refund policy"))
            ids.append(self._last())
            self._plant(FindingType.PARALLELIZABLE_CALLS, "sequential_independent_calls",
                        "get_customer, get_promotions and search_kb do not depend on each other "
                        "but run one after another", ids)
        else:
            self._observe("search_kb", self.tools.search_kb("refund policy"))
        first_kb = self._last()
        if self.f.near_duplicate_search:
            self._observe("search_kb", self.tools.search_kb("what is the refund policy for orders"))
            self._plant(FindingType.REDUNDANT_TOOL_CALL, "near_duplicate_tool_call",
                        "search_kb called twice with reworded queries that return the same article",
                        [first_kb, self._last()])

        loop_ids: list[str] = []
        suffixes = ["details", "exceptions", "fine print"]
        for i in range(4):
            decision = self._llm("TASK: decide whether more research is needed", LARGE)
            loop_ids.append(self.llm_ids[-1])
            if decision != "search_more":
                break
            self._observe("search_kb", self.tools.search_kb(f"refund policy {suffixes[i]}"))
            loop_ids.append(self._last())
            self._observe("read_article", self.tools.read_article("KB-12"))
            loop_ids.append(self._last())
        if self.f.search_loop:
            self._plant(FindingType.RETRY_LOOP, "no_progress_loop",
                        "the agent searched and re-read the same article 3 times with no new facts",
                        loop_ids)

        if self.f.unused_tool:
            self.tools.get_customer(customer_id)  # fetched "just in case", never used
            self._plant(FindingType.UNUSED_TOOL_OUTPUT, "unused_tool_output",
                        "get_customer's result never reaches any later LLM call", [self._last()])

    def _stock(self, sku: str) -> None:
        self._observe("check_inventory", self.tools.check_inventory(sku))
        if self.f.slow_inventory:
            self._plant(FindingType.LATENCY, "slow_tool",
                        "check_inventory takes 4.5 s, over half of the run", [self._last()])


def expected_answer(question: str, order_id: str | None = None, sku: str | None = None) -> str:
    """The correct answer, built from the true data rather than from what the agent saw."""
    facts = [f"refund_days={REFUND_ARTICLE['refund_days']}"]
    if order_id:
        facts.append(f"eta={ORDERS[order_id]['eta']}")
    if sku:
        facts.append(f"in_stock={INVENTORY.get(sku, 0)}")
    return answer_from_facts(question, "; ".join(facts))
