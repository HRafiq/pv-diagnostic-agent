"""The LLM boundary — one seam, three implementations.

Everything that talks to a model goes through `LLMClient`. That seam exists for
three reasons, and only the first is obvious:

1. **Testability.** The whole planner/router/synthesiser/critic loop is exercised
   against `ScriptedClient` with no network and no API key, so the orchestration
   is regression-tested independently of any model's mood.
2. **Reproducibility.** `CachedClient` keys responses on `(model, prompt_hash)`,
   so a re-run of the evaluation compares like with like. Without it, Tab 5's
   "which cases changed verdict since last run" would be reading sampling noise
   and calling it a regression.
3. **Cost control.** Every call is counted and priced against `config/models.yaml`
   before it is made, so a runaway loop fails loudly instead of quietly.

There is no `temperature` here. It was removed on current Claude models and
returns a 400; reasoning depth is set with `effort` instead (DECISION 0004).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.config import ModelConfig, ModelsConfig
from src.determinism import stable_hash

__all__ = [
    "AnthropicClient",
    "BudgetExceeded",
    "CachedClient",
    "LLMClient",
    "LLMResponse",
    "RefusedReply",
    "ScriptedClient",
    "TruncatedReply",
    "is_systemic_request_error",
    "request_kwargs",
    "sanitize_schema",
    "violations",
]


class BudgetExceeded(RuntimeError):
    """Raised when an investigation would exceed its configured cost or call cap."""


# Transient failures are retried in place: an overloaded API or a dropped
# connection is "not now", not "not ever". Three attempts with 2s/4s backoff —
# long enough to ride out a blip, short enough that a genuinely dead network
# fails the case in seconds rather than minutes.
_TRANSIENT_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


class TruncatedReply(ValueError):
    """The model ran out of output budget mid-reply.

    A subclass of `ValueError` so existing handling still catches it, but named
    so the failure reads as what it is. `max_tokens` bounds thinking plus
    response text together, so a node with adaptive thinking and a long
    structured reply can exhaust it and return valid-but-incomplete JSON.
    """


class RefusedReply(ValueError):
    """The model's safety classifiers declined the request.

    Returns HTTP 200 with empty or partial content and
    `stop_reason == "refusal"`, so it must be checked rather than parsed.
    """


def is_systemic_request_error(exc: BaseException) -> bool:
    """True when retrying with a different case cannot possibly help.

    A 400 `invalid_request_error` says the *request* is malformed — a schema the
    API rejects, a parameter that does not exist. That is a defect in this
    codebase, identical for every case, so a per-case retry loop will reproduce
    it exactly N times. It did: forty-three identical 400s in one run, each
    costing a round trip and a line of output, none of them informative after
    the first.

    Rate limits, overloads and timeouts are the opposite — worth continuing
    past, because the next case may well succeed.
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is a core dependency
        return False

    if isinstance(exc, anthropic.BadRequestError):
        return True

    # The SDK raises a plain ValueError, before any request, for a
    # non-streaming call whose max_tokens could exceed ten minutes. That is a
    # configuration defect in this repository, identical for every case — and
    # it is not a BadRequestError, so it slipped through an earlier version of
    # this check and failed all forty-three cases one at a time.
    if isinstance(exc, ValueError) and "Streaming is required" in str(exc):
        return True

    cause = exc.__cause__ or exc.__context__
    if cause is None:
        return False
    if isinstance(cause, anthropic.BadRequestError):
        return True
    return isinstance(cause, ValueError) and "Streaming is required" in str(cause)


@dataclass
class LLMResponse:
    """One model reply, plus what it cost."""

    text: str
    parsed: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    model: str = ""
    cached: bool = False

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMClient(Protocol):
    """Anything the agent nodes can call."""

    def complete(
        self,
        node: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return one completion.

        Args:
            node: Which agent node is asking — selects the model, effort and
                token cap from `config/models.yaml`.
            schema: When given, the reply is constrained to this JSON schema and
                `LLMResponse.parsed` is populated. Every node that drives control
                flow uses one; free prose can never decide what happens next.
        """
        ...


# ---------------------------------------------------------------------------
# Schema sanitising
# ---------------------------------------------------------------------------
# Structured outputs accept a subset of JSON Schema. These keywords are
# rejected outright — `minItems: 2` returns
#   "For 'array' type, 'minItems' values other than 0 or 1 are not supported"
# — so they cannot reach the wire. But several of them are load-bearing here:
# "at least two possible causes" is the difference between a differential
# diagnosis and a guess, and a confidence outside [0, 1] is meaningless.
#
# So they are moved rather than dropped. Each one is stated in the field's
# description, where the model still reads it, and re-checked against the
# parsed reply by `violations()`. The constraint survives; only its enforcement
# point changes, from the API to this file.
_ARRAY_BOUNDS = ("minItems", "maxItems")
_NUMBER_BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
_STRING_BOUNDS = ("minLength", "maxLength", "pattern")
_UNSUPPORTED = (*_ARRAY_BOUNDS, *_NUMBER_BOUNDS, *_STRING_BOUNDS, "multipleOf")

# `minItems` of 0 or 1 *is* accepted, so it stays on the wire.
_ALLOWED_MIN_ITEMS = (0, 1)


def _describe(keyword: str, value: Any) -> str:
    return {
        "minItems": f"At least {value} entries.",
        "maxItems": f"At most {value} entries.",
        "minimum": f"No less than {value}.",
        "maximum": f"No greater than {value}.",
        "exclusiveMinimum": f"Greater than {value}.",
        "exclusiveMaximum": f"Less than {value}.",
        "minLength": f"At least {value} characters.",
        "maxLength": f"At most {value} characters.",
        "multipleOf": f"A multiple of {value}.",
        "pattern": f"Matching {value}.",
    }.get(keyword, f"{keyword}: {value}.")


def sanitize_schema(schema: Any) -> Any:
    """Return a copy the structured-output API will accept.

    Constraints it rejects are removed from the schema and appended to the
    field's `description`, so the requirement still reaches the model as
    instruction even though it is no longer machine-enforced upstream.
    """
    if isinstance(schema, list):
        return [sanitize_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = {}
    moved: list[str] = []
    for key, value in schema.items():
        if key == "minItems" and value in _ALLOWED_MIN_ITEMS:
            out[key] = value
            continue
        if key in _UNSUPPORTED:
            moved.append(_describe(key, value))
            continue
        out[key] = sanitize_schema(value)

    if moved:
        description = str(out.get("description", "")).strip()
        out["description"] = " ".join([description, *moved]).strip()

    # Structured outputs reject `additionalProperties: true` and require it to
    # be present and false on every object. Set here rather than trusted to
    # each schema author, because the failure is a 400 on a request that has
    # already been paid for in latency and, on an evaluation, in real money.
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out


def violations(schema: Any, value: Any, path: str = "") -> list[str]:
    """Check a parsed reply against the constraints `sanitize_schema` removed.

    Only those: everything else the API still enforces. Returns human-readable
    strings rather than raising, so a caller can decide whether one bad field
    is worth failing a whole run over.
    """
    found: list[str] = []
    if not isinstance(schema, dict):
        return found
    where = path or "(root)"

    if isinstance(value, list):
        low, high = schema.get("minItems"), schema.get("maxItems")
        if isinstance(low, int) and len(value) < low:
            found.append(f"{where}: {len(value)} entries, needs at least {low}")
        if isinstance(high, int) and len(value) > high:
            found.append(f"{where}: {len(value)} entries, allows at most {high}")
        item_schema = schema.get("items")
        for index, item in enumerate(value):
            found.extend(violations(item_schema, item, f"{where}[{index}]"))

    elif isinstance(value, bool):
        pass  # bool is an int subclass; never range-check it

    elif isinstance(value, int | float):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, int | float) and value < low:
            found.append(f"{where}: {value} is below the minimum of {low}")
        if isinstance(high, int | float) and value > high:
            found.append(f"{where}: {value} is above the maximum of {high}")

    elif isinstance(value, dict):
        for name, sub in (schema.get("properties") or {}).items():
            if name in value:
                found.extend(violations(sub, value[name], f"{where}.{name}"))

    return found


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def request_kwargs(
    cfg: ModelConfig,
    system: str,
    user: str,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the `messages.create(...)` payload for one node.

    Split out of `AnthropicClient.complete` so it can be checked against the
    pinned SDK without a network call or an API key. That check is the whole
    reason this is a separate function: the parameters below are current-API,
    but a pinned SDK that predates them raises `TypeError: got an unexpected
    keyword argument` at the *first real call* — which, on a project whose
    evaluation had never been run with a key, meant the error waited months.

    `effort` and `format` both live inside `output_config`; neither is a
    top-level parameter. `temperature`, `top_p` and `top_k` are deliberately
    absent — they were removed on the models this project uses and return 400.
    """
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    output_config: dict[str, Any] = {}
    if cfg.effort:
        output_config["effort"] = cfg.effort
    if schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": sanitize_schema(schema),
        }
    if output_config:
        kwargs["output_config"] = output_config
    if cfg.thinking == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
    elif cfg.thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}
    return kwargs


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------
@dataclass
class AnthropicClient:
    """Calls the Claude API, one model per node.

    Structured replies use `output_config.format`, so a node whose output drives
    control flow cannot return prose. Assistant prefill is not used — it returns
    a 400 on current models.
    """

    models: ModelsConfig
    api_key: str
    profile: str | None = None
    calls: int = field(default=0, init=False)
    spend_usd: float = field(default=0.0, init=False)
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def _config(self, node: str) -> ModelConfig:
        return self.models.node(node, self.profile)

    def _check_budget(self) -> None:
        limits = self.models.limits
        if self.calls >= limits.max_llm_calls_per_investigation:
            raise BudgetExceeded(
                f"investigation hit its {limits.max_llm_calls_per_investigation}-call "
                "cap; the loop is not terminating"
            )
        if self.spend_usd >= limits.max_cost_usd_per_investigation:
            raise BudgetExceeded(
                f"investigation hit its ${limits.max_cost_usd_per_investigation:.2f} "
                f"cap after {self.calls} calls"
            )

    def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Retry the transient failures; let everything else through.

        `overloaded_error`, a dropped connection and a rate limit are the API
        saying "not now", not "not ever" — and an evaluation that abandons a
        case on the first one throws away the minutes and dollars already spent
        on it. Four runs died this way.

        A malformed request is deliberately *not* retried: it will fail
        identically every time, and `is_systemic_request_error` stops the whole
        run on it rather than reproducing it once per case.
        """
        import anthropic

        transient = (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )
        last: Exception | None = None
        for attempt in range(_TRANSIENT_RETRIES):
            try:
                with self._client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()
            except transient as exc:
                last = exc
                if attempt == _TRANSIENT_RETRIES - 1:
                    break
                time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
        assert last is not None
        raise last

    def complete(
        self,
        node: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._check_budget()
        cfg = self._config(node)
        kwargs = request_kwargs(cfg, system, user, schema)

        started = time.monotonic()
        # Streamed, always. The SDK refuses a *non*-streaming request whose
        # max_tokens it estimates could run past ten minutes — an idle HTTP
        # connection would drop first — and raises before sending anything:
        #   ValueError: Streaming is required for operations that may take
        #   longer than 10 minutes.
        # Raising the token ceilings (DECISION 0049) crossed that threshold on
        # every node, so every case failed. Streaming removes the ceiling on
        # how long a reply may take without lowering how long it may be, and
        # `get_final_message()` returns the same object `create()` would.
        message = self._call_with_retry(kwargs)
        latency_ms = int((time.monotonic() - started) * 1000)

        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
        )

        # Why the reply ended matters as much as what it says. Both of these
        # arrive as a *successful* response whose text is empty or cut off
        # mid-sentence, so without this check they surface downstream as
        # "returned unparseable text" — a message that blames the model for bad
        # JSON when the truth is a budget that ran out or a refusal.
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "max_tokens":
            raise TruncatedReply(
                f"{node} hit its {cfg.max_tokens}-token cap before finishing. "
                f"max_tokens covers thinking *and* the reply, so a long answer "
                f"at effort={cfg.effort!r} can exhaust it. Raise max_tokens for "
                f"{node} in config/models.yaml, or lower its effort.\n"
                f"  reply ended: ...{text[-120:]!r}"
            )
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            raise RefusedReply(
                f"{node} was declined by the model's safety classifiers"
                + (f" (category {category})" if category else "")
                + ". This is a successful HTTP response with no usable content."
            )
        usage = message.usage
        cost = self.models.cost_usd(cfg.model, usage.input_tokens, usage.output_tokens)
        self.calls += 1
        self.spend_usd += cost

        parsed = None
        if schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{node} was asked for structured output but returned "
                    f"unparseable text: {text[:200]!r}"
                ) from exc
            # The constraints the API cannot enforce, checked here instead.
            broken = violations(schema, parsed)
            if broken:
                raise ValueError(
                    f"{node} returned a reply that breaks its schema: "
                    + "; ".join(broken)
                )

        return LLMResponse(
            text=text,
            parsed=parsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            model=cfg.model,
        )


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------
@dataclass
class CachedClient:
    """Wraps a client and memoises on `(model, node, prompt_hash)`.

    This is what makes an evaluation re-run comparable. A cache hit costs
    nothing and is reported as `cached=True`, so cost figures stay honest —
    a cached run must never be presented as evidence about price.
    """

    inner: LLMClient
    path: Path
    enabled: bool = True
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    def _key(self, node: str, system: str, user: str, schema: object) -> str:
        return stable_hash(node, system, user, schema)

    def complete(
        self,
        node: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.enabled:
            return self.inner.complete(node, system, user, schema)

        key = self._key(node, system, user, schema)
        entry = self.path / f"{key}.json"
        if entry.exists():
            payload = json.loads(entry.read_text())
            self.hits += 1
            return LLMResponse(**{**payload, "cached": True})

        response = self.inner.complete(node, system, user, schema)
        self.misses += 1
        entry.write_text(
            json.dumps(
                {
                    "text": response.text,
                    "parsed": response.parsed,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "model": response.model,
                },
                indent=2,
            )
        )
        return response


# ---------------------------------------------------------------------------
# Test / offline client
# ---------------------------------------------------------------------------
@dataclass
class ScriptedClient:
    """A deterministic stand-in, for tests and for running without an API key.

    Replies come from a per-node queue or a callable. This is what lets the
    whole orchestration be regression-tested: the loop's control flow, the
    cycle cap, the abstention path and the trace it writes are all properties
    of the *loop*, not of any model, and they should be verifiable without a
    network call.
    """

    replies: dict[str, list[dict[str, Any] | str]] = field(default_factory=dict)
    default: dict[str, Any] | str | None = None
    calls: list[tuple[str, str, str]] = field(default_factory=list, init=False)

    def complete(
        self,
        node: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append((node, system, user))
        queue = self.replies.get(node)
        if queue:
            payload = queue.pop(0)
        elif self.default is not None:
            payload = self.default
        else:
            raise AssertionError(
                f"ScriptedClient has no reply queued for node {node!r}; "
                f"queued nodes: {sorted(self.replies)}"
            )

        if isinstance(payload, str):
            return LLMResponse(text=payload, model="scripted")
        return LLMResponse(text=json.dumps(payload), parsed=payload, model="scripted")


def build_client(
    models: ModelsConfig,
    api_key: str | None,
    cache_root: Path | None = None,
) -> LLMClient:
    """Construct the client stack the agent should use.

    Raises:
        RuntimeError: If no API key is available. Failing here is deliberate —
            silently falling back to a stub would produce an evaluation run that
            looks complete and means nothing.
    """
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set, so the agent cannot run. Copy "
            ".env.example to .env and fill it in. Physics, detectors, the "
            "simulator, the rules baseline and the dashboard all run without it."
        )
    client: LLMClient = AnthropicClient(models=models, api_key=api_key)
    cache = models.response_cache
    if cache.enabled:
        root = cache_root or Path(cache.path)
        client = CachedClient(inner=client, path=root)
    return client
