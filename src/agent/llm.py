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
    "ScriptedClient",
]


class BudgetExceeded(RuntimeError):
    """Raised when an investigation would exceed its configured cost or call cap."""


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

    def complete(
        self,
        node: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._check_budget()
        cfg = self._config(node)

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
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
        if cfg.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        elif cfg.thinking == "disabled":
            kwargs["thinking"] = {"type": "disabled"}

        started = time.monotonic()
        message = self._client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - started) * 1000)

        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
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
