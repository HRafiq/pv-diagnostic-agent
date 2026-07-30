"""The LLM boundary: caching, budgets, and refusing to run without a key.

The determinism claim in CLAUDE.md is scoped, and the scope is enforced here.
Physics and tools are bitwise reproducible; the model path is not, so a
regression run relies on `CachedClient` to compare like with like. If the cache
key ever stopped including the prompt, two different questions would return the
same answer and a whole evaluation would silently be measuring one run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.llm import (
    BudgetExceeded,
    CachedClient,
    LLMResponse,
    ScriptedClient,
    build_client,
)
from src.config import load_models_config


# ===========================================================================
# ScriptedClient
# ===========================================================================
def test_scripted_replies_are_consumed_in_order() -> None:
    client = ScriptedClient(replies={"router": [{"action": "stop"}, {"action": "go"}]})
    assert client.complete("router", "s", "u").parsed == {"action": "stop"}
    assert client.complete("router", "s", "u").parsed == {"action": "go"}


def test_a_dict_reply_is_returned_both_parsed_and_as_text() -> None:
    client = ScriptedClient(replies={"planner": [{"scope": "plant"}]})
    response = client.complete("planner", "s", "u", schema={})
    assert response.parsed == {"scope": "plant"}
    assert json.loads(response.text) == {"scope": "plant"}


def test_a_string_reply_has_no_parsed_payload() -> None:
    client = ScriptedClient(replies={"planner": ["just prose"]})
    response = client.complete("planner", "s", "u")
    assert response.parsed is None and response.text == "just prose"


def test_a_missing_reply_raises_rather_than_returning_something_plausible() -> None:
    client = ScriptedClient(replies={"planner": []})
    with pytest.raises(AssertionError, match="no reply queued"):
        client.complete("router", "s", "u")


def test_the_default_reply_covers_every_node() -> None:
    client = ScriptedClient(replies={}, default={"action": "stop"})
    assert client.complete("router", "s", "u").parsed == {"action": "stop"}
    assert client.complete("critic", "s", "u").parsed == {"action": "stop"}


def test_calls_are_recorded_for_assertions() -> None:
    client = ScriptedClient(replies={}, default="ok")
    client.complete("planner", "system text", "user text")
    assert client.calls == [("planner", "system text", "user text")]


# ===========================================================================
# CachedClient
# ===========================================================================
class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, node: str, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"reply {self.calls}",
            parsed={"n": self.calls},
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.02,
            latency_ms=900,
            model="test-model",
        )


def test_an_identical_prompt_hits_the_cache(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedClient(inner=inner, path=tmp_path)

    first = client.complete("planner", "s", "u")
    second = client.complete("planner", "s", "u")

    assert inner.calls == 1
    assert client.hits == 1 and client.misses == 1
    assert second.text == first.text
    assert second.cached is True and first.cached is False


def test_a_different_prompt_misses(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedClient(inner=inner, path=tmp_path)
    client.complete("planner", "s", "question one")
    client.complete("planner", "s", "question two")
    assert inner.calls == 2 and client.hits == 0


def test_the_node_is_part_of_the_key(tmp_path: Path) -> None:
    """Different nodes use different models. Sharing a cache entry between them
    would serve a router's answer to the synthesiser."""
    inner = CountingClient()
    client = CachedClient(inner=inner, path=tmp_path)
    client.complete("planner", "s", "u")
    client.complete("critic", "s", "u")
    assert inner.calls == 2


def test_the_schema_is_part_of_the_key(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedClient(inner=inner, path=tmp_path)
    client.complete("planner", "s", "u", schema={"type": "object"})
    client.complete("planner", "s", "u", schema={"type": "string"})
    assert inner.calls == 2


def test_the_cache_survives_a_new_process(tmp_path: Path) -> None:
    """The whole point: a re-run of an evaluation compares like with like."""
    first = CachedClient(inner=CountingClient(), path=tmp_path)
    first.complete("planner", "s", "u")

    inner = CountingClient()
    second = CachedClient(inner=inner, path=tmp_path)
    response = second.complete("planner", "s", "u")

    assert inner.calls == 0
    assert second.hits == 1
    assert response.cached is True


def test_a_cached_response_reports_itself_as_cached(tmp_path: Path) -> None:
    """Cost figures stay honest: a cached run is not evidence about price."""
    client = CachedClient(inner=CountingClient(), path=tmp_path)
    client.complete("planner", "s", "u")
    replayed = client.complete("planner", "s", "u")
    assert replayed.cached
    assert replayed.cost_usd == 0.02  # what it cost when it was really paid for


def test_caching_can_be_switched_off(tmp_path: Path) -> None:
    inner = CountingClient()
    client = CachedClient(inner=inner, path=tmp_path, enabled=False)
    client.complete("planner", "s", "u")
    client.complete("planner", "s", "u")
    assert inner.calls == 2
    assert not list(tmp_path.glob("*.json"))


# ===========================================================================
# Budgets and construction
# ===========================================================================
def test_no_api_key_fails_loudly() -> None:
    """Falling back to a stub would produce an evaluation run that looks
    complete and means nothing."""
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        build_client(load_models_config(), api_key=None)


def test_the_call_cap_is_enforced_before_the_call_is_made() -> None:
    from src.agent.llm import AnthropicClient

    models = load_models_config()
    client = AnthropicClient.__new__(AnthropicClient)
    object.__setattr__(client, "models", models)
    object.__setattr__(client, "calls", models.limits.max_llm_calls_per_investigation)
    object.__setattr__(client, "spend_usd", 0.0)

    with pytest.raises(BudgetExceeded, match="not terminating"):
        client._check_budget()


def test_the_cost_cap_is_enforced() -> None:
    from src.agent.llm import AnthropicClient

    models = load_models_config()
    client = AnthropicClient.__new__(AnthropicClient)
    object.__setattr__(client, "models", models)
    object.__setattr__(client, "calls", 3)
    object.__setattr__(
        client, "spend_usd", models.limits.max_cost_usd_per_investigation
    )

    with pytest.raises(BudgetExceeded, match="cap after 3 calls"):
        client._check_budget()


def test_pricing_covers_every_configured_model() -> None:
    """A model with no price entry would silently report a run as costing $0."""
    models = load_models_config()
    for profile in models.profiles.values():
        for node in profile.values():
            assert node.model in models.pricing


def test_cost_is_computed_per_million_tokens() -> None:
    models = load_models_config()
    price = models.pricing["claude-sonnet-5"]
    cost = models.cost_usd("claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(price.input + price.output)


def test_response_tokens_sum_input_and_output() -> None:
    assert LLMResponse(text="", input_tokens=100, output_tokens=25).tokens == 125
