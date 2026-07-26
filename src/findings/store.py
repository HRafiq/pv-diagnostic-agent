"""The findings store: an append-only log, plus the lifecycle over it.

Without lifecycle you get the same three findings shouted at you every morning
until you stop reading them, and a monitoring system nobody reads is worse than
none — it provides cover. So the store's real job is not storage, it is knowing
that the thing it saw today is the thing it saw yesterday.

**Identity is `(scope, cause-or-candidate-set)`, never the text.** Two runs of
the same investigation produce different prose, and matching on wording would
open a new finding every sweep. An unsettled finding is identified by the *set*
of causes still standing, so an investigation that narrows four candidates to
two is a genuinely different finding and correctly opens a new one.

**Nothing is ever mutated or deleted.** Every state change appends a new record
with a new `observed_at`, and the current view is a fold over the log. That is
what makes "what changed since yesterday" answerable at all, and it means a
finding that was wrong stays visible rather than vanishing.

Times come from the injected clock. There is no wall clock in this file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.clock import Clock
from src.findings.models import Finding, Lifecycle

__all__ = ["FindingRecord", "FindingsStore", "identity_of"]


def identity_of(finding: Finding) -> str:
    """The key two sightings of the same problem share.

    Scope plus what the finding concluded — the cause when it settled, the
    sorted set of survivors when it did not. Deliberately not the title or the
    summary: those are written afresh on every run and would make every sweep
    look like a new problem.
    """
    if finding.settled:
        answer = finding.cause or "(unnamed)"
    else:
        answer = "|".join(sorted(c.cause for c in finding.candidate_causes))
    return f"{finding.scope}::{answer}"


@dataclass(frozen=True)
class FindingRecord:
    """One sighting: a finding as it stood at one moment of simulated time."""

    identity: str
    observed_at: datetime
    lifecycle: Lifecycle
    finding: Finding
    note: str = ""
    misses: int = 0
    """Consecutive sweeps this finding was not seen in.

    Carried on the record rather than derived, because the log only knows about
    sweeps that *wrote* something. A sweep where nothing fired leaves no trace,
    so counting distinct timestamps would mean a finding is only aged when some
    other finding happens to be active — and a resolved string fault would sit
    open forever on a quiet plant.
    """

    @property
    def is_bookkeeping(self) -> bool:
        """A record that only advanced the miss count, changing nothing else."""
        return self.misses > 0 and self.lifecycle != "resolved"

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "observed_at": self.observed_at.isoformat(),
            "lifecycle": self.lifecycle,
            "note": self.note,
            "misses": self.misses,
            "finding": self.finding.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FindingRecord:
        return cls(
            identity=str(payload["identity"]),
            observed_at=datetime.fromisoformat(str(payload["observed_at"])),
            lifecycle=str(payload["lifecycle"]),  # type: ignore[arg-type]
            finding=Finding.model_validate(payload["finding"]),
            note=str(payload.get("note", "")),
            misses=int(str(payload.get("misses", 0) or 0)),
        )


class FindingsStore:
    """Append-only JSONL, with the lifecycle folded over it on read.

    Args:
        path: The log file. Created on first write.
        clock: Injected time source. Every record is stamped with *simulated*
            time, so a replayed 2017 sweep dates its findings in 2017.
        resolve_after_sweeps: How many consecutive sweeps a finding must go
            unseen before it is marked resolved. One is too eager — a single
            cloudy day can hide a real string fault from a detector — and this
            is the knob that decides whether the interface cries wolf or goes
            quiet on something real.
    """

    def __init__(
        self,
        path: Path | str,
        clock: Clock,
        resolve_after_sweeps: int = 3,
    ) -> None:
        self._path = Path(path)
        self._clock = clock
        self._resolve_after = max(1, resolve_after_sweeps)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def records(self) -> Iterator[FindingRecord]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield FindingRecord.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(
                        f"{self._path}:{lineno} is not a valid finding record"
                    ) from exc

    def current(self) -> list[FindingRecord]:
        """The latest record for each identity — the fold over the log."""
        latest: dict[str, FindingRecord] = {}
        for record in self.records():
            latest[record.identity] = record
        return list(latest.values())

    def open_findings(self) -> list[FindingRecord]:
        """Everything a plant manager still has to act on, ranked."""
        live = [
            r
            for r in self.current()
            if r.lifecycle in ("new", "ongoing", "acknowledged")
        ]
        return rank_by_energy(live)

    def history(self, identity: str) -> list[FindingRecord]:
        return [r for r in self.records() if r.identity == identity]

    def since(
        self, when: datetime, include_bookkeeping: bool = False
    ) -> list[FindingRecord]:
        """What changed after `when` — the "what's new since yesterday" query.

        Ageing records are excluded by default. "Still not seen, second sweep"
        is real history and belongs in the log, but it is not news, and a
        what's-new list padded with it is one nobody reads.
        """
        return [
            r
            for r in self.records()
            if r.observed_at > when and (include_bookkeeping or not r.is_bookkeeping)
        ]

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def observe(
        self, findings: Iterable[Finding], note: str = ""
    ) -> list[FindingRecord]:
        """Record one sweep's findings and age everything it did not see.

        Returns only the records this call wrote, so a caller can report what
        changed without diffing the whole store.
        """
        now = self._clock.now()
        before = {r.identity: r for r in self.current()}
        written: list[FindingRecord] = []
        seen: set[str] = set()

        for finding in findings:
            identity = identity_of(finding)
            seen.add(identity)
            previous = before.get(identity)
            if previous is None:
                lifecycle: Lifecycle = "new"
            elif previous.lifecycle == "suppressed":
                # A suppressed finding stays suppressed until a human clears it.
                # Re-raising it on the next sweep is exactly how an operator
                # learns to ignore the whole interface.
                lifecycle = "suppressed"
            elif previous.lifecycle == "acknowledged":
                lifecycle = "acknowledged"
            else:
                lifecycle = "ongoing"
            written.append(self._append(identity, now, lifecycle, finding, note))

        # Age out anything not seen this sweep.
        for identity, previous in before.items():
            if identity in seen or previous.lifecycle in ("resolved", "suppressed"):
                continue
            missed = previous.misses + 1
            if missed >= self._resolve_after:
                written.append(
                    self._append(
                        identity,
                        now,
                        "resolved",
                        previous.finding,
                        f"not seen in {missed} consecutive sweeps",
                    )
                )
            else:
                # Not yet resolved, but the miss has to be recorded or the
                # count restarts every sweep and nothing ever closes.
                self._append(
                    identity,
                    now,
                    previous.lifecycle,
                    previous.finding,
                    f"not seen in {missed} consecutive sweeps",
                    misses=missed,
                )
        return written

    def set_lifecycle(
        self, identity: str, lifecycle: Lifecycle, note: str = ""
    ) -> FindingRecord | None:
        """Acknowledge or suppress a finding. Appends; never edits."""
        latest = {r.identity: r for r in self.current()}.get(identity)
        if latest is None:
            return None
        return self._append(
            identity, self._clock.now(), lifecycle, latest.finding, note
        )

    # ------------------------------------------------------------------
    def _append(
        self,
        identity: str,
        when: datetime,
        lifecycle: Lifecycle,
        finding: Finding,
        note: str,
        misses: int = 0,
    ) -> FindingRecord:
        stamped = finding.model_copy(update={"lifecycle": lifecycle})
        record = FindingRecord(identity, when, lifecycle, stamped, note, misses)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            handle.flush()
        return record


def rank_by_energy(records: list[FindingRecord]) -> list[FindingRecord]:
    """Most energy at stake first, with verified figures ahead of unverified.

    A finding whose cause is unsettled carries an unverified energy figure, and
    an unverified number must not outrank a verified one of the same size — it
    is a smaller claim and the ordering should say so.
    """
    return sorted(
        records,
        key=lambda r: (
            r.finding.energy_verified,
            r.finding.energy_at_stake_kwh,
        ),
        reverse=True,
    )
