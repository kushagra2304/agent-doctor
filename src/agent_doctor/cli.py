"""Command-line interface.

Phase 0 ships one command, ``validate``. Phase 1 moves this to Typer and adds
``runs`` and ``cost``.
"""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError

from agent_doctor import __version__
from agent_doctor.schema import Run


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        run = Run.from_json_file(args.path)
    except FileNotFoundError:
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2
    except ValidationError as e:
        print(f"invalid trace: {args.path}\n{e}", file=sys.stderr)
        return 1
    print(json.dumps(run.summary(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-doctor",
        description="Find out why your AI agent is expensive, slow or unreliable.",
    )
    parser.add_argument("--version", action="version", version=f"agent-doctor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Check that a trace file matches the schema.")
    p.add_argument("path", help="Path to a run JSON file.")
    p.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
