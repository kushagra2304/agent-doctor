"""Remove personal data and secrets from runs before they are stored or shared.

>>> from agent_doctor.ingest import Redactor
>>> clean = Redactor().run(run)

Turn it on for everything you record with ``AGENT_DOCTOR_REDACT=1``. Tool argument hashes
are kept from the original values, so duplicate-call detection still works on redacted runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agent_doctor.schema import Run


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def _phone(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[PHONE]" if 10 <= len(digits) <= 15 else m.group(0)


_CALLBACKS = {"CARD": _card, "PHONE": _phone}


# Order matters: specific secrets before generic patterns.
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("PRIVATE_KEY", r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"),
    ("JWT", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("API_KEY", r"\b(?:sk-ant-[A-Za-z0-9_-]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}"
                r"|gh[pousr]_[A-Za-z0-9]{36,}|xox[abpr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})"),
    ("AWS_KEY", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("TOKEN", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("CARD", r"\b(?:\d[ -]?){12,18}\d\b"),
    ("PHONE", r"(?<![\w.+])\+?\(?\d[\d .()-]{7,}\d(?!\w)"),
    ("IP", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
]

SECRET_KEYS = frozenset({"password", "passwd", "secret", "api_key", "apikey", "access_token",
                         "refresh_token", "authorization", "auth", "token", "cookie", "set-cookie",
                         "private_key", "client_secret"})


@dataclass
class Redactor:
    """Replaces matches with ``[NAME]`` and masks values under secret-looking keys.

    ``patterns`` are (name, regex) pairs; ``extra`` adds to the defaults; ``keys`` are dict
    keys (case-insensitive) whose values are replaced entirely.
    """

    patterns: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    extra: Iterable[tuple[str, str]] = ()
    keys: frozenset[str] = SECRET_KEYS

    def __post_init__(self) -> None:
        self._compiled = [(name, re.compile(rx, re.S)) for name, rx in [*self.patterns, *self.extra]]

    def text(self, s: str) -> str:
        for name, rx in self._compiled:
            s = rx.sub(_CALLBACKS.get(name, f"[{name}]"), s)
        return s

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, dict):
            return {k: "[REDACTED]" if str(k).lower() in self.keys and v[k] not in (None, "")
                    else self.value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        return v

    def run(self, run: Run) -> Run:
        """A redacted copy of the run (the original is not changed)."""
        data = run.model_dump(mode="json")
        for key in ("task_input", "final_output", "tags"):
            data[key] = self.value(data[key])
        for span in data["spans"]:
            for key in ("input_messages", "output_text", "tool_args", "tool_output", "error_message",
                        "attributes"):
                span[key] = self.value(span[key])
        return Run.model_validate(data)
