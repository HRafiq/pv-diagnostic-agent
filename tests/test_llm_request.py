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
from types import SimpleNamespace
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


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_every_object_is_closed(name: str) -> None:
    """`additionalProperties: false` is required on every object.

    The keyword-rejection test above checks what a schema must *not* carry.
    This checks what it must carry — the distinction that let a second 400
    through after the first was fixed:

      400 For 'object' type, 'additionalProperties: true' is not supported.

    The router's tool-argument object was deliberately open, which is not
    expressible here. It is built from the tool registry instead.
    """
    clean = sanitize_schema(ALL_SCHEMAS[name])

    open_objects = [
        f"{path} (additionalProperties={value!r})"
        for path, value in _objects_with_bad_additional_properties(clean)
    ]
    assert not open_objects, (
        f"the {name} schema has objects the API will reject: {open_objects}"
    )


def _objects_with_bad_additional_properties(
    node: Any, path: str = "(root)"
) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            value = node.get("additionalProperties", "<missing>")
            if value is not False:
                found.append((path, value))
        for key, value in node.items():
            found.extend(
                _objects_with_bad_additional_properties(value, f"{path}.{key}")
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(
                _objects_with_bad_additional_properties(item, f"{path}[{index}]")
            )
    return found


def test_sanitising_closes_an_object_the_author_left_open() -> None:
    """Enforced structurally, not left to each schema author to remember."""
    assert (
        sanitize_schema({"type": "object", "additionalProperties": True})[
            "additionalProperties"
        ]
        is False
    )
    assert (
        sanitize_schema({"type": "object", "properties": {}})["additionalProperties"]
        is False
    )


def test_the_router_argument_vocabulary_matches_the_tools() -> None:
    """The closed args object must name every argument a tool can take.

    If they drift, the router cannot express a call the registry accepts —
    a silent capability loss rather than an error.
    """
    from src.tools import REGISTRY

    declared = set(ROUTE_SCHEMA["properties"]["args"]["properties"])
    real = {
        field for spec in REGISTRY.values() for field in spec.args_model.model_fields
    }
    assert declared == real, (
        f"router args schema and tool registry disagree: "
        f"only in schema {sorted(declared - real)}, "
        f"only in tools {sorted(real - declared)}"
    )


def test_a_null_argument_means_not_supplied() -> None:
    """Every arg is required-and-nullable to satisfy the object rules, so the
    nulls must be dropped before a tool sees them and overrides its defaults."""
    from src.agent.nodes.router import _clean_args

    assert _clean_args({"start": "2017-05-16", "end": None, "min_run": 8}) == {
        "start": "2017-05-16",
        "min_run": 8,
    }
    assert _clean_args(None) == {}
    assert _clean_args("not a dict") == {}


def test_a_malformed_request_is_recognised_as_systemic() -> None:
    """A 400 repeats identically for every case; a rate limit does not."""
    import anthropic
    import httpx

    from src.agent.llm import is_systemic_request_error

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    bad = anthropic.BadRequestError(
        "schema rejected",
        response=httpx.Response(400, request=request),
        body=None,
    )
    limited = anthropic.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=request),
        body=None,
    )

    assert is_systemic_request_error(bad)
    assert not is_systemic_request_error(limited)
    assert not is_systemic_request_error(ValueError("a bad reply"))

    # Wrapped by a node before it reaches the runner.
    try:
        try:
            raise bad
        except anthropic.BadRequestError as exc:
            raise RuntimeError("planner failed") from exc
    except RuntimeError as wrapped:
        assert is_systemic_request_error(wrapped)


# ---------------------------------------------------------------------------
# Why a reply ended
# ---------------------------------------------------------------------------
# The first working run failed on:
#   ValueError: synthesizer was asked for structured output but returned
#   unparseable text: '{"settled": true, "category": "not_the_plant", ...
# The JSON was not malformed — it was cut off mid-sentence. `max_tokens` bounds
# thinking and reply together, and 8000 was not enough for a synthesis at
# effort=high. The message blamed the model for a budget problem, which is the
# kind of error that sends someone to rewrite a prompt for an afternoon.
class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Usage:
    input_tokens = 100
    output_tokens = 200


class _Reply:
    def __init__(
        self, text: str, stop_reason: str, category: str | None = None
    ) -> None:
        self.content = [_Block(text)]
        self.usage = _Usage()
        self.stop_reason = stop_reason
        self.stop_details = SimpleNamespace(category=category) if category else None


class _Stream:
    """Stands in for the SDK's streaming context manager.

    The client streams rather than calling `create` — the SDK refuses a
    non-streaming request whose max_tokens could run past ten minutes, which is
    every node now that the ceilings are raised. A double that stubs `create`
    would pass while the real path was broken, so this mirrors the shape the
    code actually uses.
    """

    def __init__(self, reply: _Reply) -> None:
        self._reply = reply

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> _Reply:
        return self._reply


def _client_returning(reply: _Reply) -> Any:
    from src.agent.llm import AnthropicClient

    client = AnthropicClient(models=load_models_config(), api_key="not-a-real-key")
    client._client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kw: _Stream(reply))
    )
    return client


def test_the_client_streams_rather_than_blocking() -> None:
    """Pinned because the failure mode is silent until a real call.

    A non-streaming request whose max_tokens could exceed ten minutes raises
    `ValueError: Streaming is required ...` *before sending anything* — so
    raising the token ceilings broke every node at once, and a test double
    stubbing `create` would not have noticed.
    """
    from src.agent.llm import AnthropicClient

    used: list[str] = []
    client = AnthropicClient(models=load_models_config(), api_key="not-a-real-key")
    client._client = SimpleNamespace(
        messages=SimpleNamespace(
            stream=lambda **kw: (
                used.append("stream"),
                _Stream(_Reply('{"a": 1}', "end_turn")),
            )[1],
            create=lambda **kw: pytest.fail("blocking create must not be used"),
        )
    )
    client.complete("synthesizer", "sys", "user", None)
    assert used == ["stream"]


def test_the_streaming_guard_counts_as_systemic() -> None:
    """It is raised for every case identically, so the run must stop, not
    reproduce it forty-three times."""
    from src.agent.llm import is_systemic_request_error

    guard = ValueError(
        "Streaming is required for operations that may take longer than 10 minutes."
    )
    assert is_systemic_request_error(guard)
    assert not is_systemic_request_error(ValueError("a reply broke its schema"))

    try:
        try:
            raise guard
        except ValueError as exc:
            raise RuntimeError("planner failed") from exc
    except RuntimeError as wrapped:
        assert is_systemic_request_error(wrapped)


def test_a_truncated_reply_says_it_was_truncated() -> None:
    from src.agent.llm import TruncatedReply

    client = _client_returning(_Reply('{"settled": true, "cause": "stri', "max_tokens"))

    with pytest.raises(TruncatedReply) as caught:
        client.complete("synthesizer", "sys", "user", SCHEMA)

    message = str(caught.value)
    assert "token cap before finishing" in message
    assert "max_tokens covers thinking" in message
    assert "config/models.yaml" in message


def test_a_refusal_is_not_reported_as_bad_json() -> None:
    from src.agent.llm import RefusedReply

    client = _client_returning(_Reply("", "refusal", category="cyber"))

    with pytest.raises(RefusedReply) as caught:
        client.complete("synthesizer", "sys", "user", SCHEMA)
    assert "safety classifiers" in str(caught.value)
    assert "cyber" in str(caught.value)


def test_genuinely_unparseable_text_still_reports_as_such() -> None:
    """The truncation check must not swallow a real formatting failure."""
    client = _client_returning(_Reply("I'm afraid I can't do that.", "end_turn"))

    with pytest.raises(ValueError, match="unparseable text"):
        client.complete("synthesizer", "sys", "user", SCHEMA)


def test_a_complete_reply_is_returned_normally() -> None:
    client = _client_returning(_Reply('{"answer": "ok"}', "end_turn"))
    response = client.complete("synthesizer", "sys", "user", SCHEMA)
    assert response.parsed == {"answer": "ok"}


@pytest.mark.parametrize(("profile", "node", "cfg"), NODES, ids=NODE_IDS)
def test_every_node_has_room_for_its_reply(
    profile: str, node: str, cfg: ModelConfig
) -> None:
    """max_tokens bounds thinking *and* the reply on these models.

    A ceiling costs nothing when unused — billing tracks tokens generated — so
    a tight one buys no saving and risks a truncated answer. These floors are
    set from the first real run, where the synthesiser truncated at 8000.
    """
    floors = {"planner": 12000, "router": 1500, "synthesizer": 24000, "critic": 12000}
    assert cfg.max_tokens >= floors[node], (
        f"{profile}:{node} caps output at {cfg.max_tokens}, below the "
        f"{floors[node]} this node needs for thinking plus a full reply"
    )
