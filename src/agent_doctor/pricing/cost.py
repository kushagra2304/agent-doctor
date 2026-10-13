"""Cost of a run: per span, per model and in total."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_doctor.pricing.table import PriceTable
from agent_doctor.schema import Run, SpanKind


class SpanCost(BaseModel):
    span_id: str
    kind: str
    name: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = Field(default=None, description="None when the model has no price.")
    price_match: str | None = Field(default=None, description="The pricing pattern that matched.")


class ModelCost(BaseModel):
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None


class CostReport(BaseModel):
    run_id: str
    total_cost_usd: float = Field(description="Sum of all priced spans (LLM and tool costs).")
    llm_cost_usd: float
    tool_cost_usd: float = Field(description="From tool spans with attributes['cost_usd'].")
    cache_savings_usd: float = Field(description="Saved by cached input tokens vs full price.")
    fully_priced: bool
    unpriced_models: list[str]
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    most_expensive_span_id: str | None
    spans: list[SpanCost]
    by_model: list[ModelCost]

    def projected_cost_usd(self, runs: int) -> float:
        """What ``runs`` runs like this one would cost."""
        return self.total_cost_usd * runs


def estimate_cost(run: Run, prices: PriceTable | None = None) -> CostReport:
    prices = prices if prices is not None else PriceTable.load()
    span_costs: list[SpanCost] = []
    by_model: dict[str, ModelCost] = {}
    unpriced: set[str] = set()
    llm_total = tool_total = savings = 0.0

    for span in run.ordered_spans:
        if span.kind is SpanKind.LLM:
            model = span.model or span.name
            entry = prices.find(model, span.provider)
            cost = None
            if entry is not None:
                cost = entry.cost_usd(span.input_tokens, span.output_tokens, span.cached_tokens)
                inp, cached_rate, _ = entry.rates(span.input_tokens)
                savings += span.cached_tokens * (inp - cached_rate) / 1_000_000
                llm_total += cost
            else:
                unpriced.add(model)
            span_costs.append(
                SpanCost(
                    span_id=span.span_id, kind=span.kind.value, name=span.name, model=model,
                    input_tokens=span.input_tokens, output_tokens=span.output_tokens,
                    cached_tokens=span.cached_tokens, cost_usd=cost,
                    price_match=entry.match if entry else None,
                )
            )
            mc = by_model.setdefault(model, ModelCost(model=model, cost_usd=0.0))
            mc.calls += 1
            mc.input_tokens += span.input_tokens
            mc.output_tokens += span.output_tokens
            mc.cached_tokens += span.cached_tokens
            mc.cost_usd = None if cost is None or mc.cost_usd is None else mc.cost_usd + cost
        else:
            raw = span.attributes.get("cost_usd")
            cost = float(raw) if isinstance(raw, int | float) and raw >= 0 else None
            if cost is not None:
                tool_total += cost
            span_costs.append(
                SpanCost(span_id=span.span_id, kind=span.kind.value, name=span.name, cost_usd=cost)
            )

    priced = [s for s in span_costs if s.cost_usd]
    most = max(priced, key=lambda s: s.cost_usd or 0.0).span_id if priced else None
    return CostReport(
        run_id=run.run_id,
        total_cost_usd=llm_total + tool_total,
        llm_cost_usd=llm_total,
        tool_cost_usd=tool_total,
        cache_savings_usd=savings,
        fully_priced=not unpriced,
        unpriced_models=sorted(unpriced),
        input_tokens=run.total_input_tokens,
        output_tokens=run.total_output_tokens,
        cached_tokens=run.total_cached_tokens,
        most_expensive_span_id=most,
        spans=span_costs,
        by_model=sorted(by_model.values(), key=lambda m: -(m.cost_usd or 0.0)),
    )
