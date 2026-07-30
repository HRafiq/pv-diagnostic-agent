"""A network failure and an archive gap are not the same fact.

PVDAQ has genuine holes — a decade-long archive of real loggers does. A day
that 404s is *data*: the day does not exist, and the quality panel draws it as
a gap. A day that could not be fetched because DNS died is not data; nothing
was learned about that day at all.

The ingest used to record both as "missing". A flaky resolver during a download
therefore truncated four months off the end of a record, the command printed a
success summary, and the loss surfaced much later as an unrelated crash inside
an evaluation run — by which point the shortened record looked like the
dataset. These tests pin the distinction and the refusal to write a record
short by network failure.
"""

from __future__ import annotations

import urllib.error
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import src.data.ingest as ingest
from src.data.ingest import PartialDownload, _read_day
from src.data.sources import SystemMetadata


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry backoff is real seconds. Tests assert the count, not the wait."""
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://example/x", code=code, msg="x", hdrs=None, fp=None
    )


# ---------------------------------------------------------------------------
# The distinction itself
# ---------------------------------------------------------------------------
def test_a_404_is_an_archive_gap_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fetch(_: str) -> bytes:
        nonlocal calls
        calls += 1
        raise _http_error(404)

    monkeypatch.setattr(ingest, "fetch_bytes", fetch)

    _, payload, status = _read_day(4902, date(2016, 5, 1))
    assert (payload, status) == (None, "gap")
    assert calls == 1, "a day the archive does not have will not appear on retry"


def test_a_dns_failure_is_retried_and_then_reported_as_a_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fetch(_: str) -> bytes:
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(ingest, "fetch_bytes", fetch)

    _, payload, status = _read_day(4902, date(2016, 5, 1))
    assert (payload, status) == (None, "failed")
    assert calls == ingest._FETCH_RETRIES


def test_a_transient_failure_recovers_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fetch(_: str) -> bytes:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise TimeoutError("slow")
        return b"payload"

    monkeypatch.setattr(ingest, "fetch_bytes", fetch)

    _, payload, status = _read_day(4902, date(2016, 5, 1))
    assert (payload, status) == (b"payload", "ok")


def test_a_server_error_is_not_mistaken_for_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 says the server is unwell, not that the day is absent."""
    monkeypatch.setattr(
        ingest, "fetch_bytes", lambda _: (_ for _ in ()).throw(_http_error(503))
    )

    _, payload, status = _read_day(4902, date(2016, 5, 1))
    assert status == "failed"
    assert payload is None


# ---------------------------------------------------------------------------
# What the ingest does with them
# ---------------------------------------------------------------------------
def _meta() -> SystemMetadata:
    return SystemMetadata(
        system_id=4902,
        name="Test",
        latitude=39.13,
        longitude=0.0,
        altitude_m=138.0,
        location="nowhere",
        dc_capacity_kw=100.0,
        tilt_deg=20.0,
        azimuth_deg=180.0,
        tracking=False,
        module_model="Test Module",
        module_quantity=350,
        modules_per_string=50,
        strings=7,
        inverter_model="Test Inverter 95kW",
        raw={},
    )


def test_a_partial_download_refuses_to_write_anything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The record must not reach disk, because nothing downstream can tell a
    short record from a complete one."""
    monkeypatch.setattr(
        ingest,
        "fetch_bytes",
        lambda _: (_ for _ in ()).throw(urllib.error.URLError("dns")),
    )

    with pytest.raises(PartialDownload) as caught:
        ingest.ingest_system(_meta(), years=[2016], out_dir=tmp_path)

    message = str(caught.value)
    assert "network failure" in message
    assert "--allow-partial" in message
    assert not list(tmp_path.glob("*.parquet")), "a partial record was written"
    assert not list(tmp_path.glob("*manifest.json"))


def test_a_total_network_failure_does_not_blame_the_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every day failing must not be reported as 'no PVDAQ data for this
    system' — that is a claim about the archive, and it would send someone to
    check the wrong thing."""
    monkeypatch.setattr(
        ingest,
        "fetch_bytes",
        lambda _: (_ for _ in ()).throw(urllib.error.URLError("dns")),
    )

    with pytest.raises(PartialDownload):
        ingest.ingest_system(_meta(), years=[2016], out_dir=tmp_path)


def test_an_archive_gap_alone_does_not_stop_the_ingest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """404s are a normal property of the dataset. Only fetch failures block."""
    monkeypatch.setattr(
        ingest, "fetch_bytes", lambda _: (_ for _ in ()).throw(_http_error(404))
    )

    # No PartialDownload: nothing failed to fetch. It falls through to the
    # empty-record error instead, which is the accurate complaint here.
    with pytest.raises(RuntimeError) as caught:
        ingest.ingest_system(_meta(), years=[2016], out_dir=tmp_path)
    assert not isinstance(caught.value, PartialDownload)
    assert "no PVDAQ data" in str(caught.value)


def test_allow_partial_is_required_to_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With the flag the ingest proceeds past the guard rather than raising."""
    monkeypatch.setattr(
        ingest,
        "fetch_bytes",
        lambda _: (_ for _ in ()).throw(urllib.error.URLError("dns")),
    )

    with pytest.raises(RuntimeError) as caught:
        ingest.ingest_system(
            _meta(), years=[2016], out_dir=tmp_path, allow_partial=True
        )
    # It got past the PartialDownload guard and failed later, on having no rows.
    assert not isinstance(caught.value, PartialDownload)


def test_the_manifest_separates_gaps_from_fetch_failures() -> None:
    """Only one of the two belongs in a reproducibility record.

    `missing_days` describes the dataset and is the same for everyone.
    `fetch_failures` describes one download on one machine.
    """
    source = Path(ingest.__file__).read_text(encoding="utf-8")
    assert '"missing_days": missing,' in source
    assert '"fetch_failures": failed,' in source


def test_ingest_result_carries_both_counts() -> None:
    fields: dict[str, Any] = ingest.IngestResult.__dataclass_fields__
    assert "missing_days" in fields
    assert "fetch_failures" in fields
