# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Generate roster-reference and routes-schema pages from the Pydantic models.

Runs during ``mkdocs build`` via the mkdocs-gen-files plugin.  Introspects
[`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml] (the agent YAML
input contract) and [`VaultRouteEntry`][terok_executor.roster.schema.VaultRouteEntry]
(the generated ``routes.json`` output contract) to produce:

- A markdown roster-reference page with field tables and an annotated YAML example
- ``schemas/agent.schema.json`` for editor autocompletion of agent YAMLs
- ``schemas/routes.schema.json`` for sandbox-side validation of the generated routes file

Every ``Field(description=...)`` in those models is the **single source of truth**.
"""

from __future__ import annotations

import io
import json

import mkdocs_gen_files
from mkdocs_terok.config_reference import (
    render_json_schema,
    render_model_tables,
    render_yaml_example,
)
from pydantic import TypeAdapter

from terok_executor.roster.schema import RawAgentYaml, VaultRouteEntry

_MD_RULE = "---\n\n"


def _generate() -> str:
    """Build the full roster-reference.md content."""
    buf = io.StringIO()
    buf.write("# Agent Roster Reference\n\n")
    buf.write(
        "This page is **auto-generated** from the Pydantic schema in "
        "[`roster.schema`][terok_executor.roster.schema].  Every field listed "
        "here is validated at load time — unknown keys are rejected, catching "
        "typos before they silently fall back to defaults.\n\n"
    )
    buf.write(
        "**JSON Schema files** (for editor autocompletion and validation):\n\n"
        "[:material-download: agent.schema.json](schemas/agent.schema.json){: .md-button }\n"
        "[:material-download: routes.schema.json](schemas/routes.schema.json){: .md-button }\n\n"
    )

    buf.write(_MD_RULE)
    buf.write("## Agent YAML\n\n")
    buf.write(
        "Each file under ``resources/agents/*.yaml`` (and any user override "
        "in ``~/.config/terok/agent/agents/*.yaml``) is parsed into "
        "[`RawAgentYaml`][terok_executor.roster.schema.RawAgentYaml] before "
        "being projected onto the runtime types in "
        "[`roster.types`][terok_executor.roster.types].\n\n"
        'All sections use ``extra="forbid"`` — typos like ``headles:`` or '
        "``prommpt_flag:`` raise a precise error rather than silently using "
        "defaults.\n\n"
    )
    buf.write(render_model_tables(RawAgentYaml))

    buf.write("### Full example\n\n")
    buf.write('```yaml title="claude.yaml"\n')
    buf.write(render_yaml_example(RawAgentYaml))
    buf.write("```\n\n")

    buf.write(_MD_RULE)
    buf.write("## Generated routes.json\n\n")
    buf.write(
        "[`AgentRoster.generate_routes_json()`][terok_executor.roster.loader.AgentRoster.generate_routes_json] "
        "produces the ``routes.json`` file consumed by the sandbox vault server.  "
        "Each entry conforms to "
        "[`VaultRouteEntry`][terok_executor.roster.schema.VaultRouteEntry].  "
        "The full file is a top-level ``{provider_name: VaultRouteEntry}`` object; "
        "empty optional fields are dropped from the serialized output.\n\n"
    )
    buf.write(render_model_tables(VaultRouteEntry))

    return buf.getvalue()


_routes_adapter: TypeAdapter[dict[str, VaultRouteEntry]] = TypeAdapter(dict[str, VaultRouteEntry])


with mkdocs_gen_files.open("roster-reference.md", "w") as f:
    f.write(_generate())

with mkdocs_gen_files.open("schemas/agent.schema.json", "w") as f:
    f.write(render_json_schema(RawAgentYaml, title="terok-executor agent YAML"))

with mkdocs_gen_files.open("schemas/routes.schema.json", "w") as f:
    schema = _routes_adapter.json_schema()
    schema.setdefault("title", "terok-executor generated routes.json")
    f.write(json.dumps(schema, indent=2))
