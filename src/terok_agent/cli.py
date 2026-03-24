# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for terok-agent.

Built from the command registry in :mod:`terok_agent.commands`.
No command logic lives here — just argument wiring and dispatch.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _meta_version

from .commands import COMMANDS, ArgDef, CommandDef

try:
    __version__ = _meta_version("terok-agent")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _wire_command(sub: argparse._SubParsersAction, cmd: CommandDef) -> None:
    """Add a :class:`CommandDef` to an argparse subparser group."""
    p = sub.add_parser(cmd.name, help=cmd.help)
    for arg in cmd.args:
        kwargs: dict = {}
        if arg.help:
            kwargs["help"] = arg.help
        if arg.type is not None:
            kwargs["type"] = arg.type
        if arg.default is not None:
            kwargs["default"] = arg.default
        if arg.action is not None:
            kwargs["action"] = arg.action
        if arg.dest is not None:
            kwargs["dest"] = arg.dest
        if arg.nargs is not None:
            kwargs["nargs"] = arg.nargs
        p.add_argument(arg.name, **kwargs)
    p.set_defaults(_cmd=cmd)


def _arg_key(arg: ArgDef) -> str:
    """Derive the kwarg name for an argument definition."""
    return arg.dest or arg.name.lstrip("-").replace("-", "_")


def _dispatch(args: argparse.Namespace) -> None:
    """Extract handler kwargs from parsed args and call the handler."""
    cmd: CommandDef = args._cmd
    if cmd.handler is None:
        raise SystemExit(f"Command '{cmd.name}' has no handler")
    kwargs = {_arg_key(arg): getattr(args, _arg_key(arg), arg.default) for arg in cmd.args}
    cmd.handler(**kwargs)


def main() -> None:
    """Run the terok-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="terok-agent",
        description="Single-agent task runner for hardened Podman containers",
    )
    parser.add_argument("--version", action="version", version=f"terok-agent {__version__}")
    sub = parser.add_subparsers()

    for cmd in COMMANDS:
        _wire_command(sub, cmd)

    args = parser.parse_args()
    if hasattr(args, "_cmd"):
        _dispatch(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
