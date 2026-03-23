# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Vendored utility functions — standalone, no terok-agent domain deps."""

from ._fs import ensure_dir, ensure_dir_writable
from ._podman import podman_userns_args
from ._yaml import load as yaml_load

__all__ = [
    "ensure_dir",
    "ensure_dir_writable",
    "podman_userns_args",
    "yaml_load",
]
