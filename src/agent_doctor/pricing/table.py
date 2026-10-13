"""Load model prices and match model names to them."""

from __future__ import annotations

import fnmatch
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_doctor import config


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace(" ", "-").replace("_", "-")


class TierPrices(BaseModel):
    model_config = ConfigDict(extra="forbid")
    above_input_tokens: int = Field(gt=0)
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cached_input: float | None = Field(default=None, ge=0)


class ModelPrice(BaseModel):
    """Prices for one model pattern, USD per 1M tokens."""

    model_config = ConfigDict(extra="forbid")
    match: str
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cached_input: float | None = Field(default=None, ge=0, description="Defaults to input.")
    provider: str | None = None
    long_context: TierPrices | None = None
    as_of: str | None = None
    source: str | None = None
    origin: str = Field(default="built-in", description="Which file this entry came from.")

    def rates(self, input_tokens: int) -> tuple[float, float, float]:
        """(input, cached_input, output) per 1M tokens for a call of this size."""
        tier = self.long_context
        if tier is not None and input_tokens > tier.above_input_tokens:
            cached = tier.cached_input if tier.cached_input is not None else tier.input
            return tier.input, cached, tier.output
        cached = self.cached_input if self.cached_input is not None else self.input
        return self.input, cached, self.output

    def cost_usd(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        inp, cached, out = self.rates(input_tokens)
        uncached = max(input_tokens - cached_tokens, 0)
        return (uncached * inp + cached_tokens * cached + output_tokens * out) / 1_000_000


class PriceTable:
    """Ordered price entries. Earlier entries (user files) take priority over later ones."""

    def __init__(self, entries: list[ModelPrice] | None = None):
        self.entries: list[ModelPrice] = list(entries or [])

    # -- loading ----------------------------------------------------------------
    @staticmethod
    def parse(data: dict[str, Any], origin: str) -> list[ModelPrice]:
        if not isinstance(data, dict) or "models" not in data:
            raise ValueError(f"{origin}: expected a mapping with a 'models' list")
        unit = data.get("unit", "per_million_tokens")
        if unit != "per_million_tokens":
            raise ValueError(f"{origin}: only unit 'per_million_tokens' is supported, got {unit!r}")
        return [ModelPrice(**{**m, "origin": origin}) for m in data["models"] or []]

    @classmethod
    def from_file(cls, path: str | Path) -> PriceTable:
        text = Path(path).read_text(encoding="utf-8")
        return cls(cls.parse(yaml.safe_load(text) or {}, origin=str(path)))

    @classmethod
    def builtin(cls) -> PriceTable:
        text = resources.files("agent_doctor.pricing").joinpath("pricing.yaml").read_text("utf-8")
        return cls(cls.parse(yaml.safe_load(text), origin="built-in"))

    @classmethod
    def load(cls, extra_file: str | Path | None = None) -> PriceTable:
        """``extra_file`` first, then your user pricing file, then the built-in table."""
        entries: list[ModelPrice] = []
        if extra_file is not None:
            entries += cls.from_file(extra_file).entries
        user = config.user_pricing_path()
        if user.is_file():
            entries += cls.from_file(user).entries
        entries += cls.builtin().entries
        return cls(entries)

    # -- matching ---------------------------------------------------------------
    def find(self, model: str | None, provider: str | None = None) -> ModelPrice | None:
        if not model:
            return None
        full = normalize_model_name(model)
        candidates = [full]
        if "/" in full:
            candidates.append(full.rsplit("/", 1)[1])

        best: tuple[int, int, int, ModelPrice] | None = None  # (exact, specificity, -order)
        for order, entry in enumerate(self.entries):
            if entry.provider and provider and entry.provider.lower() != provider.lower():
                continue
            pattern = normalize_model_name(entry.match)
            for name in candidates:
                if name == pattern:
                    key = (1, len(pattern), -order, entry)
                elif fnmatch.fnmatchcase(name, pattern):
                    key = (0, len(pattern.replace("*", "")), -order, entry)
                else:
                    continue
                if best is None or key[:3] > best[:3]:
                    best = key
        return best[3] if best else None

    def __len__(self) -> int:
        return len(self.entries)
