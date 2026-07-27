"""The pinned SDK must accept what the agent actually sends.

This is the test the project most needed and did not have. Every request
parameter in `src/agent/llm.py` was correct for the current Claude API, and the
pinned SDK was `anthropic==0.62.0` — fifteen versions before `output_config`
existed on `messages.create`. Nothing failed, because nothing had ever called
it: no API key was available while the project was built, so the entire LLM
path had never executed once. The first real run died on
`TypeError: Messages.create() got an unexpected keyword argument 'output_config'`
thirty seconds in, with an API key already spent on nothing.

The whole suite ran green throughout. `ScriptedClient` covers the orchestration
without a network, which is the right design — but it means the one thing a
scripted client cannot check is whether the *real* client's payload is valid.

So this file checks the payload against the installed SDK's own signature and
parameter types. It needs no key, makes no request, and costs nothing.
"""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from src.agent.llm import request_kwargs, sanitize_schema, violations
from src.agent.nodes.critic import CRITIC_SCHEMA
from src.agent.nodes.planner import PLAN_SCHEMA
from src.agent.nodes.router import ROUTE_SCHEMA
from src.agent.nodes.synthesizer import SYNTHESIS_SCHEMA
from src.config import ModelConfig, load_models_config

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _create_signature() -> inspect.Signature:
    import anthropic

    return inspect.signature(
        anthropic.Anthropic(api_key="not-a-real-key").messages.create
    )


def _sdk_effort_levels() -> set[str]:
    """The effort levels the installed SDK types, read from the SDK itself.

    Read rather than restated so a level the API adds or drops shows up here on
    the next dependency bump instead of as a 400 on the first call.
    """
    from anthropic.types.output_config_param import OutputConfigParam

    hint = get_type_hints(OutputConfigParam).get("effort")
    levels = {
        value
        for member in get_args(hint)
        for value in get_args(member)
        if isinstance(value, str)
    }
    assert levels, "could not read effort levels from the SDK's OutputConfigParam"
    return levels


def _configured_nodes() -> list[tuple[str, str, ModelConfig]]:
    """Every (profile, node) pair in config/models.yaml.

    Profiles exist so the cost/quality experiment is a config change rather than
    a code change — which means a profile nobody has run yet can carry a
    parameter the SDK rejects. They are all checked.
    """
    models = load_models_config()
    pairs: list[tuple[str, str, ModelConfig]] = []
    for profile in sorted(models.profiles):
        for node in ("planner", "router", "synthesizer", "critic"):
            pairs.append((profile, node, models.node(node, profile)))
    return pairs


NODES = _configured_nodes()
NODE_IDS = [f"{profile}:{node}" for profile, node, _ in NODES]


@pytest.mark.parametrize(("profile", "node", "cfg"), NODES, ids=NODE_IDS)
def test_every_parameter_is_accepted_by_the_pinned_sdk(
    profile: str, node: str, cfg: ModelConfig
) -> None:
    """The exact failure that shipped: a parameter the pinned SDK does not have."""
    accepted = set(_create_signature().parameters)
    sent = set(request_kwargs(cfg, "system", "user", SCHEMA))

    unknown = sorted(sent - accepted)
    assert not unknown, (
        f"{profile}:{node} sends {unknown}, which the installed anthropic SDK "
        f"does not accept. Either the pin in pyproject.toml is behind the API "
        f"(`output_config` needs >= 0.77.0) or src/agent/llm.py is sending a "
        f"parameter that no longer exists."
    )


def test_output_config_is_a_real_parameter_not_passthrough() -> None:
    """`output_config` must be typed, not swallowed by `**kwargs` or extra_body.

    A signature check alone would pass on an SDK that accepted anything. This
    pins that the SDK models the parameter, which is what makes the check above
    mean something.
    """
    parameters = _create_signature().parameters
    assert "output_config" in parameters

    kinds = {p.kind for p in parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds, (
        "messages.create accepts **kwargs, so an unknown-parameter check cannot "
        "detect a stale pin — tighten this test against the request model instead"
    )


def test_effort_and_format_are_nested_not_top_level() -> None:
    """Both live inside `output_config`. Sending either at top level is a 400."""
    cfg = ModelConfig(
        model="claude-sonnet-5",
        effort="high",
        thinking="adaptive",
        max_tokens=8000,
        cache_system_prompt=True,
    )
    kwargs = request_kwargs(cfg, "system", "user", SCHEMA)

    assert "effort" not in kwargs
    assert "format" not in kwargs
    assert kwargs["output_config"]["effort"] == "high"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"


def test_the_removed_sampling_parameters_are_never_sent() -> None:
    """`temperature`, `top_p` and `top_k` return a 400 on every model this
    project uses. A config that reintroduced one would fail on the first call."""
    for profile, node, cfg in NODES:
        kwargs = request_kwargs(cfg, "system", "user", None)
        banned = {"temperature", "top_p", "top_k"} & set(kwargs)
        assert not banned, f"{profile}:{node} sends removed parameters {banned}"


@pytest.mark.parametrize(("profile", "node", "cfg"), NODES, ids=NODE_IDS)
def test_the_configured_effort_level_is_one_the_sdk_knows(
    profile: str, node: str, cfg: ModelConfig
) -> None:
    """A typo in an effort level is a 400 on the first call, not a config error.

    `router` deliberately carries `effort: null` because Haiku 4.5 rejects the
    parameter; that is a valid configuration and is skipped rather than failed.
    """
    if not cfg.effort:
        pytest.skip("this node sends no effort level")

    assert cfg.effort in _sdk_effort_levels(), (
        f"{profile}:{node} sets effort={cfg.effort!r}, which is not one of "
        f"{sorted(_sdk_effort_levels())}"
    )


def test_a_schemaless_call_sends_no_format() -> None:
    """Nodes that return prose must not be constrained to a schema."""
    cfg = ModelConfig(
        model="claude-sonnet-5",
        effort="high",
        thinking="adaptive",
        max_tokens=8000,
        cache_system_prompt=True,
    )
    kwargs = request_kwargs(cfg, "system", "user", None)
    assert "format" not in kwargs.get("output_config", {})
    assert kwargs["output_config"]["effort"] == "high"


def test_thinking_is_adaptive_or_disabled_never_a_token_budget() -> None:
    """`budget_tokens` was removed on these models and returns a 400."""
    for _, _, cfg in NODES:
        thinking = request_kwargs(cfg, "s", "u", None).get("thinking")
        if thinking is None:
            continue
        assert thinking.get("type") in {"adaptive", "disabled"}
        assert "budget_tokens" not in thinking


def test_unused_parameter_names_are_not_silently_accepted() -> None:
    """Guard the guard: if this ever passes, the signature check is worthless."""
    accepted = set(_create_signature().parameters)
    assert "definitely_not_a_real_parameter" not in accepted
    assert get_origin(dict[str, Any]) is dict  # keep the typing import honest


# ---------------------------------------------------------------------------
# Schema validity — the second thing the first real run found
# ---------------------------------------------------------------------------
# Structured outputs accept a subset of JSON Schema. `minItems: 2` on the
# planner's hypotheses list returned:
#   400 output_config.format.schema: For 'array' type, 'minItems' values other
#   than 0 or 1 are not supported (got: [2, 5])
# Same root cause as the stale pin above — a request that had never been made.
# `sanitize_schema` moves those constraints into the field description and
# `violations` re-checks them against the reply, so the requirement survives
# with its enforcement point moved.
ALL_SCHEMAS: dict[str, dict[str, Any]] = {
    "planner": PLAN_SCHEMA,
    "router": ROUTE_SCHEMA,
    "synthesizer": SYNTHESIS_SCHEMA,
    "critic": CRITIC_SCHEMA,
}

# Rejected by the structured-output API. `minItems` is special: 0 and 1 are
# accepted, anything else is not.
REJECTED_KEYWORDS = {
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "multipleOf",
    "pattern",
}


def _walk(node: Any, path: str = "") -> list[tuple[str, str, Any]]:
    """Every (path, keyword, value) pair in a schema."""
    found: list[tuple[str, str, Any]] = []
    if isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_walk(item, f"{path}[{index}]"))
    elif isinstance(node, dict):
        for key, value in node.items():
            found.append((path or "(root)", key, value))
            found.extend(_walk(value, f"{path}.{key}"))
    return found


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_the_sanitised_schema_carries_nothing_the_api_rejects(name: str) -> None:
    clean = sanitize_schema(ALL_SCHEMAS[name])

    offences = [
        f"{path}.{keyword} = {value!r}"
        for path, keyword, value in _walk(clean)
        if keyword in REJECTED_KEYWORDS
        or (keyword == "minItems" and value not in (0, 1))
    ]
    assert not offences, (
        f"the {name} schema still carries constraints structured outputs "
        f"rejects, so the first call returns a 400: {offences}"
    )


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_sanitising_preserves_the_shape(name: str) -> None:
    """Only constraints are removed — never a property, type, or enum."""
    original, clean = ALL_SCHEMAS[name], sanitize_schema(ALL_SCHEMAS[name])

    def shape(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: shape(value)
                for key, value in sorted(node.items())
                if key not in REJECTED_KEYWORDS
                and key not in {"minItems", "description"}
            }
        if isinstance(node, list):
            return [shape(item) for item in node]
        return node

    assert shape(original) == shape(clean)


def test_a_removed_constraint_is_still_stated_to_the_model() -> None:
    """Dropping it from the wire must not drop it from the instructions."""
    clean = sanitize_schema(PLAN_SCHEMA)
    description = clean["properties"]["hypotheses"]["description"]
    assert "At least 2" in description
    assert "At most 8" in description


def test_a_supported_min_items_stays_on_the_wire() -> None:
    """0 and 1 are accepted, so they keep being machine-enforced."""
    clean = sanitize_schema(PLAN_SCHEMA)
    assert clean["properties"]["planned_tools"]["minItems"] == 1


def test_the_removed_constraints_are_re_checked_against_the_reply() -> None:
    """The point of moving them: they are still enforced, just here."""
    one_cause = {
        "scope": "array",
        "opening_reasoning": "...",
        "hypotheses": [{"id": "H1", "cause": "soiling"}],
        "planned_tools": ["compute_temp_corrected_pr"],
    }
    broken = violations(PLAN_SCHEMA, one_cause)
    assert broken and "at least 2" in broken[0]

    two_causes = {**one_cause, "hypotheses": [{"id": "H1"}, {"id": "H2"}]}
    assert violations(PLAN_SCHEMA, two_causes) == []


def test_a_confidence_outside_the_unit_interval_is_caught() -> None:
    assert violations(SYNTHESIS_SCHEMA, {"confidence": 1.4})
    assert violations(SYNTHESIS_SCHEMA, {"confidence": -0.1})
    assert violations(SYNTHESIS_SCHEMA, {"confidence": 0.78}) == []


def test_booleans_are_not_range_checked_as_numbers() -> None:
    """`bool` subclasses `int`; a naive check would compare True against a
    minimum and produce nonsense."""
    schema = {"properties": {"flag": {"type": "boolean", "minimum": 5}}}
    assert violations(schema, {"flag": True}) == []
