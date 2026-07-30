from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    ModelConfig,
    ModelsConfig,
    build_clock,
    load_models_config,
    load_site_defaults,
)


def test_site_defaults_load_and_validate() -> None:
    defaults = load_site_defaults()
    site = defaults.site("dkasc_alice_springs")
    assert site.latitude < 0  # southern hemisphere
    assert site.timezone == "Australia/Darwin"
    # Desert siting: albedo above the generic 0.20 default.
    assert site.albedo > 0.20


def test_unknown_site_lists_what_is_configured() -> None:
    with pytest.raises(KeyError, match="configured sites"):
        load_site_defaults().site("nope")


def test_thresholds_are_present_and_sane() -> None:
    t = load_site_defaults().thresholds
    # Below this POA the PR denominator gets small and the ratio blows up.
    assert t["pr_min_poa_w_m2"] >= 100.0
    assert 0.0 < t["pr_deficit_open"] < 0.5
    assert 0.9 < t["clipping_frac_of_ac_rating"] <= 1.0


def test_models_config_loads_all_four_nodes() -> None:
    models = load_models_config()
    for node in ("planner", "router", "synthesizer", "critic"):
        assert models.node(node).max_tokens > 0


def test_no_temperature_field_exists() -> None:
    # temperature/top_p/top_k were removed on Opus 5, Sonnet 5, Opus 4.7/4.8 —
    # sending one is a 400. The schema must not tempt anyone into adding it.
    for field in ("temperature", "top_p", "top_k"):
        assert field not in ModelConfig.model_fields


def test_effort_is_rejected_on_models_that_do_not_support_it() -> None:
    with pytest.raises(ValidationError, match="rejects the `effort` parameter"):
        ModelConfig(model="claude-haiku-4-5", effort="high", max_tokens=1000)


def test_every_configured_model_has_a_price() -> None:
    # Otherwise cost accounting silently reports $0 and the cost target in
    # eval §5.6 becomes unfalsifiable.
    models = load_models_config()
    for profile in models.profiles.values():
        for node in profile.values():
            assert node.model in models.pricing


def test_cost_arithmetic() -> None:
    models = load_models_config()
    # 1M input + 1M output on Sonnet 5 list pricing = $3 + $15.
    assert models.cost_usd("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(
        18.0
    )


def test_profile_missing_a_node_is_rejected() -> None:
    with pytest.raises(ValidationError, match="missing node config"):
        ModelsConfig.model_validate(
            {
                "active_profile": "p",
                "pricing": {"claude-sonnet-5": {"input": 3.0, "output": 15.0}},
                "profiles": {
                    "p": {"planner": {"model": "claude-sonnet-5", "max_tokens": 100}}
                },
                "limits": {
                    "max_llm_calls_per_investigation": 1,
                    "max_cost_usd_per_investigation": 1.0,
                    "max_planner_critic_cycles": 1,
                    "max_tools_per_cycle": 1,
                    "request_timeout_seconds": 1,
                },
                "response_cache": {"enabled": False, "path": "x"},
            }
        )


def test_clock_is_built_from_config() -> None:
    clock = build_clock()
    assert clock.now() == clock.start
    assert clock.now().tzinfo is not None


def test_critic_cycle_cap_matches_the_handoff() -> None:
    assert load_models_config().limits.max_planner_critic_cycles == 4


def test_the_measurement_budget_lives_in_exactly_one_place() -> None:
    """It was hardcoded in `loop_plain`, `loop_graph` *and* `eval/runner`.

    Three copies of one budget that could disagree silently. Both loops and the
    runner now default to `None` and read the limit, so raising it is a config
    change rather than three code edits with two chances to forget.
    """
    import inspect

    from eval.runner import run_agent_engine
    from src.agent.loop_graph import investigate_with_graph
    from src.agent.loop_plain import investigate

    for fn in (investigate, investigate_with_graph, run_agent_engine):
        default = inspect.signature(fn).parameters["max_tools_per_cycle"].default
        assert default is None, f"{fn.__name__} carries its own copy of the budget"

    assert load_models_config().limits.max_tools_per_cycle > 0


def test_the_circuit_breakers_sit_above_what_the_configuration_allows() -> None:
    """A breaker below the operating point does not save money — it kills runs.

    `BudgetExceeded` aborts the whole evaluation by design, because it normally
    means the loop is not terminating. So a per-investigation cap set below what
    four cycles legitimately cost converts an ordinary expensive case into a dead
    run, which is what the old $1.00 did: one measured cycle is $0.65.

    Checked as arithmetic rather than as literals, so raising the cycle cap or
    the measurement budget forces the breakers to be revisited.
    """
    limits = load_models_config().limits

    # One router turn per measurement, one per knowledge look-up, one to stop;
    # plus planner, synthesiser and critic. Two look-ups per cycle is typical.
    calls_per_cycle = limits.max_tools_per_cycle + 3 + 3
    allowed = calls_per_cycle * limits.max_planner_critic_cycles
    assert limits.max_llm_calls_per_investigation >= allowed, (
        f"{limits.max_planner_critic_cycles} cycles need about {allowed} calls; "
        f"the breaker at {limits.max_llm_calls_per_investigation} would fire first"
    )

    # Measured: $0.65 for one cycle at eight measurements, dominated by the
    # three Sonnet nodes rather than by the Haiku router.
    measured_cost_per_cycle = 0.70
    assert limits.max_cost_usd_per_investigation >= (
        measured_cost_per_cycle * limits.max_planner_critic_cycles
    ), "the cost breaker sits below what the cycle allowance legitimately costs"
