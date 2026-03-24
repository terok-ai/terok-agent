# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-agent."""

from __future__ import annotations

import argparse
import sys


def _cmd_agents(args: argparse.Namespace) -> None:
    """List registered agents (and optionally tools)."""
    from .registry import _load_bundled_agents, _load_user_agents, get_registry

    reg = get_registry()
    names = reg.all_names if args.all else reg.agent_names

    if not names:
        print("No agents registered.", file=sys.stderr)
        return

    # Read raw YAML for kind metadata
    raw = _load_bundled_agents()
    raw.update(_load_user_agents())

    rows: list[tuple[str, str, str]] = []
    for name in sorted(names):
        p = reg.providers.get(name)
        auth = reg.auth_providers.get(name)
        label = p.label if p else (auth.label if auth else name)
        kind = raw.get(name, {}).get("kind", "native")
        rows.append((name, label, kind))

    w_name = max(len("NAME"), max(len(r[0]) for r in rows))
    w_label = max(len("LABEL"), max(len(r[1]) for r in rows))

    print(f"{'NAME':<{w_name}}  {'LABEL':<{w_label}}  TYPE")
    for name, label, kind in rows:
        print(f"{name:<{w_name}}  {label:<{w_label}}  {kind}")


def main() -> None:
    """Run the terok-agent CLI."""
    parser = argparse.ArgumentParser(prog="terok-agent", description="Single-agent task runner")
    sub = parser.add_subparsers()

    # agents
    agents_p = sub.add_parser("agents", help="List registered agents")
    agents_p.add_argument("--all", action="store_true", help="Include tools (gh, glab)")
    agents_p.set_defaults(func=_cmd_agents)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
