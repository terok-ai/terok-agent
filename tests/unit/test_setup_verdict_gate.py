# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Cheap stamp-based gate that runs before live preflight on ``run`` / ``run-tool``.

Pins the script-readable exit-code contract from epic terok-ai/terok#685
phase 4: ``3 = setup needed`` for FIRST_RUN / STALE_AFTER_UPDATE /
STAMP_CORRUPT, ``4 = downgrade detected`` for STALE_AFTER_DOWNGRADE,
no exit on OK.  ``--no-preflight`` waives the gate as a single escape
hatch.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from terok_sandbox import SetupVerdict

from terok_executor.commands import _setup_verdict_or_exit

# ── Verdict → exit-code mapping ───────────────────────────────────────


@pytest.mark.parametrize(
    ("verdict", "expected_code", "expected_fragment"),
    [
        pytest.param(
            SetupVerdict.FIRST_RUN,
            3,
            "no setup stamp found",
            id="first-run-exits-3",
        ),
        pytest.param(
            SetupVerdict.STALE_AFTER_UPDATE,
            3,
            "package versions changed",
            id="stale-after-update-exits-3",
        ),
        pytest.param(
            SetupVerdict.STAMP_CORRUPT,
            3,
            "stamp is unreadable",
            id="stamp-corrupt-exits-3",
        ),
    ],
)
def test_setup_needed_verdicts_all_exit_three(
    verdict: SetupVerdict,
    expected_code: int,
    expected_fragment: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The three "setup needed" verdicts collapse to exit 3 with a fix hint."""
    with patch("terok_sandbox.needs_setup", return_value=verdict):
        with pytest.raises(SystemExit) as excinfo:
            _setup_verdict_or_exit(skip=False)
    assert excinfo.value.code == expected_code
    err = capsys.readouterr().err
    assert expected_fragment in err
    # Every "setup needed" path points at the canonical fix.
    assert "terok-executor setup" in err


def test_downgrade_exits_four_with_named_packages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """STALE_AFTER_DOWNGRADE refuses with exit 4, names the offending packages."""
    with (
        patch("terok_sandbox.needs_setup", return_value=SetupVerdict.STALE_AFTER_DOWNGRADE),
        patch(
            "terok_executor.commands._name_downgraded_packages",
            return_value=["terok-sandbox 0.0.97 → 0.0.95"],
        ),
    ):
        with pytest.raises(SystemExit) as excinfo:
            _setup_verdict_or_exit(skip=False)
    assert excinfo.value.code == 4
    err = capsys.readouterr().err
    assert "downgrade detected" in err
    assert "terok-sandbox 0.0.97 → 0.0.95" in err
    # Downgrade message points at the deliberate override path, not the easy fix.
    assert "rm" in err and "stamp" in err


def test_downgrade_falls_back_to_generic_when_diff_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the stamp can't be re-read for diffing, surface a generic refusal — never crash."""
    with (
        patch("terok_sandbox.needs_setup", return_value=SetupVerdict.STALE_AFTER_DOWNGRADE),
        patch("terok_executor.commands._name_downgraded_packages", return_value=[]),
    ):
        with pytest.raises(SystemExit) as excinfo:
            _setup_verdict_or_exit(skip=False)
    assert excinfo.value.code == 4
    assert "one or more packages" in capsys.readouterr().err


def test_ok_verdict_returns_silently() -> None:
    """OK is the happy path — no print, no exit, control flows back to preflight."""
    with patch("terok_sandbox.needs_setup", return_value=SetupVerdict.OK):
        # No SystemExit raised; helper returns None.
        assert _setup_verdict_or_exit(skip=False) is None


# ── Escape hatch ──────────────────────────────────────────────────────


def test_skip_bypasses_gate_without_calling_needs_setup() -> None:
    """``--no-preflight`` waives the gate too — same knob, same scope.

    The user's escape hatch is one flag, not two.  Crucially we don't
    even call ``needs_setup`` when skipping — pinning that no I/O
    happens on the bypass path.
    """
    with patch("terok_sandbox.needs_setup") as mock_needs_setup:
        assert _setup_verdict_or_exit(skip=True) is None
        mock_needs_setup.assert_not_called()


# ── _name_downgraded_packages helper ──────────────────────────────────


def test_name_downgraded_packages_lists_each_offender(tmp_path) -> None:
    """Helper compares stamped vs installed and names every package that regressed."""
    from terok_executor.commands import _name_downgraded_packages

    stamp = tmp_path / "setup.stamp"

    def fake_read(_path):
        return {"terok-sandbox": "0.0.97", "terok-executor": "0.0.115", "terok-shield": "0.6.31"}

    def fake_installed():
        # sandbox went backwards, executor stayed put, shield disappeared entirely.
        return {"terok-sandbox": "0.0.95", "terok-executor": "0.0.115"}

    out = _name_downgraded_packages(stamp, fake_read, fake_installed)
    assert "terok-sandbox 0.0.97 → 0.0.95" in out
    assert "terok-shield (uninstalled)" in out
    # Equal versions don't get listed.
    assert not any("terok-executor" in entry for entry in out)


def test_name_downgraded_packages_swallows_read_error(tmp_path) -> None:
    """A racing setup overwriting the stamp can't crash the diagnostic helper."""
    from terok_executor.commands import _name_downgraded_packages

    def boom(_path):
        raise RuntimeError("stamp went away mid-diff")

    out = _name_downgraded_packages(tmp_path / "x", boom, lambda: {})
    assert out == []
