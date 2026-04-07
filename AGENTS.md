# Agent Instructions

## Verification

Run `make check` before declaring work done — it covers lint, unit
tests, module boundaries (tach), security, docstring coverage, dead
code, and SPDX compliance.  Skip tests that need podman or docker;
those run on a dedicated test machine.

## Code style

- Domain-first docstrings, public entry points above private helpers,
  top-down reading order.
- SPDX copyright: author name "Jiri Vyskocil".  Add a new
  `SPDX-FileCopyrightText` line only for a previously unlisted
  contributor making a substantive change.

## Docs

- Markdown files under `docs/` are lowercase by convention; root-level
  files (`README.md`, `AGENTS.md`) are not.
