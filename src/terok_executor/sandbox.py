# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Bring the sandbox layer up to a state an agent container can use.

The sandbox package installs shield / vault / gate / clearance but
leaves the vault's ``routes.json`` for its consumer to populate —
routes are generated from terok-executor's agent roster, which
sandbox (correctly) doesn't know about.  This module ties the two
halves together so every caller that wants a functional runtime
reaches for one entry point.

:func:`ensure_sandbox_ready` is the composable hook every frontend
should call: ``terok-executor setup``, ``terok-executor run`` when
the preflight needs to self-heal, and — in PR #6 — ``terok setup``
as it sheds its own inline orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terok_sandbox import SandboxConfig


def ensure_sandbox_ready(
    *,
    root: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    no_gate: bool = False,
    no_clearance: bool = False,
    cfg: SandboxConfig | None = None,
) -> None:
    """Generate vault routes, then run the sandbox install aggregator.

    Order matters: the aggregator's vault phase calls ``install_systemd_units``
    which enables+starts the vault unit; the vault reads ``routes.json``
    on startup, so the routes file must already exist.  Without the
    pre-step, the vault starts empty and agents fail to fetch
    credentials — the operator then has to remember to run
    ``terok-executor vault routes`` separately.

    The ``no_*`` flags mirror the sandbox aggregator's shape so this
    helper is a drop-in replacement for a direct ``_handle_sandbox_setup``
    call in any frontend that's routing vault traffic through the
    executor's agent roster.
    """
    from terok_sandbox.commands import _handle_sandbox_setup

    from terok_executor.roster.loader import ensure_vault_routes

    if not no_vault:
        ensure_vault_routes(cfg=cfg)
    _handle_sandbox_setup(
        root=root,
        no_shield=no_shield,
        no_vault=no_vault,
        no_gate=no_gate,
        no_clearance=no_clearance,
        cfg=cfg,
    )


def uninstall_sandbox_services(
    *,
    root: bool = False,
    no_shield: bool = False,
    no_vault: bool = False,
    no_gate: bool = False,
    no_clearance: bool = False,
) -> None:
    """Tear down the sandbox stack — routes.json is left on disk.

    Operators re-installing after an uninstall keep their roster-
    derived routes, so the next install doesn't regenerate them from
    scratch unless the roster itself changed.  Symmetric to
    :func:`ensure_sandbox_ready` for the ``--no-*`` opt-out flags.
    """
    from terok_sandbox.commands import _handle_sandbox_uninstall

    _handle_sandbox_uninstall(
        root=root,
        no_shield=no_shield,
        no_vault=no_vault,
        no_gate=no_gate,
        no_clearance=no_clearance,
    )
