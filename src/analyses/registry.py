"""Saved analyses: a config row, never generated code.

Handoff §3.6 asks for an agent factory. This is the v1 of it, and the v1 is
deliberately level 1 only: a saved analysis is a *question plus a selection over
existing tools*. It is not a new tool, not new orchestration, and not generated
code. A registry that could add capability would need its own evaluation for
every entry, and there is no way to build that honestly at this scale.

The load-bearing rule is the one on `AnalysisSpec.golden_case_ids`: **saving
requires at least one golden case**. An agent factory that produces unevaluated
agents defeats the entire point of the project — it manufactures things that
look like diagnostics and have never been scored. The validator enforces the
field is non-empty; this module enforces that the ids actually exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.state import AnalysisSpec
from src.config import REPO_ROOT
from src.tools import tool_names

__all__ = ["REGISTRY_PATH", "AnalysisRegistry"]

REGISTRY_PATH = REPO_ROOT / "config" / "analyses.json"


@dataclass
class AnalysisRegistry:
    """Saved analyses, validated against the tool set and the golden set."""

    path: Path = REGISTRY_PATH
    known_cases: frozenset[str] = frozenset()

    @classmethod
    def with_golden_cases(cls, path: Path | None = None) -> AnalysisRegistry:
        """Build a registry that checks ids against the real golden set."""
        from eval.golden import build_golden_set

        return cls(
            path=path or REGISTRY_PATH,
            known_cases=frozenset(case.id for case in build_golden_set()),
        )

    # ------------------------------------------------------------------
    def load(self) -> list[AnalysisSpec]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text())
        return [AnalysisSpec.model_validate(row) for row in payload.get("analyses", [])]

    def get(self, analysis_id: str) -> AnalysisSpec | None:
        return next((a for a in self.load() if a.id == analysis_id), None)

    def save(self, spec: AnalysisSpec) -> AnalysisSpec:
        """Validate and persist. Raises rather than saving something unscored."""
        self.validate(spec)
        existing = [a for a in self.load() if a.id != spec.id]
        rows = sorted([*existing, spec], key=lambda a: a.id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"analyses": [a.model_dump(mode="json") for a in rows]}, indent=2
            )
        )
        return spec

    def delete(self, analysis_id: str) -> bool:
        rows = [a for a in self.load() if a.id != analysis_id]
        if len(rows) == len(self.load()):
            return False
        self.path.write_text(
            json.dumps(
                {"analyses": [a.model_dump(mode="json") for a in rows]}, indent=2
            )
        )
        return True

    # ------------------------------------------------------------------
    def validate(self, spec: AnalysisSpec) -> None:
        """Everything that must be true before an analysis can be saved.

        The model's own validators catch the empty cases; these catch the ones
        that need the rest of the system to know about — a tool that does not
        exist, a golden case that does not exist, a question with a slot
        nothing will ever fill.
        """
        unknown_tools = [t for t in spec.allowed_tools if t not in tool_names()]
        if unknown_tools:
            raise ValueError(
                f"analysis {spec.id!r} allows tools that do not exist: "
                + ", ".join(sorted(unknown_tools))
            )
        if self.known_cases:
            missing = [c for c in spec.golden_case_ids if c not in self.known_cases]
            if missing:
                raise ValueError(
                    f"analysis {spec.id!r} cites golden cases that do not exist: "
                    + ", ".join(sorted(missing))
                    + ". An analysis evaluated against nothing is an unevaluated "
                    "analysis wearing a number."
                )

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": a.id,
                "name": a.name,
                "scope": a.scope,
                "trigger": a.trigger,
                "tools": len(a.allowed_tools),
                "golden_cases": len(a.golden_case_ids),
            }
            for a in self.load()
        ]
