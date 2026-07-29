"""Typed loaders for ``config/*.yaml`` and the environment.

Config is validated at load time rather than read ad hoc, so a typo in a
threshold fails at startup instead of silently changing a detector's behaviour
three steps into an evaluation run.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.clock import ReplayClock

__all__ = [
    "REPO_ROOT",
    "ModelConfig",
    "ModelsConfig",
    "Settings",
    "SiteConfig",
    "SiteDefaults",
    "build_clock",
    "load_models_config",
    "load_site_defaults",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    """Runtime settings from the environment. Secrets live only here."""

    model_config = ConfigDict(extra="ignore")

    anthropic_api_key: str | None = None
    model_profile: str = "default"
    data_dir: Path = REPO_ROOT / "data"
    traces_dir: Path = REPO_ROOT / "traces"
    corpus_dir: Path = REPO_ROOT / "corpus"

    @classmethod
    def from_env(cls) -> Settings:
        # Load .env if present, without adding a hard dependency on it existing.
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model_profile=os.environ.get("PV_MODEL_PROFILE", "default"),
            data_dir=Path(os.environ.get("PV_DATA_DIR", REPO_ROOT / "data")),
            traces_dir=Path(os.environ.get("PV_TRACES_DIR", REPO_ROOT / "traces")),
            corpus_dir=Path(os.environ.get("PV_CORPUS_DIR", REPO_ROOT / "corpus")),
        )

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "fill it in. Only the agent nodes need it — physics, "
                "detectors, the simulator and the dashboard all run without."
            )
        return self.anthropic_api_key


# ---------------------------------------------------------------------------
# site_defaults.yaml
# ---------------------------------------------------------------------------
class SiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    altitude_m: float
    timezone: str
    albedo: float = Field(..., ge=0.0, le=1.0)
    source: str | None = None


class ClockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    speed: float = Field(default=0.0, ge=0.0)
    step: str = "1D"


class SiteDefaults(BaseModel):
    """Whole-file model for ``config/site_defaults.yaml``."""

    model_config = ConfigDict(extra="forbid")

    sites: dict[str, SiteConfig]
    expectation_model: dict[str, object]
    stc: dict[str, float]
    thresholds: dict[str, float]
    clock: ClockConfig

    def site(self, key: str) -> SiteConfig:
        try:
            return self.sites[key]
        except KeyError:
            known = ", ".join(sorted(self.sites)) or "(none)"
            raise KeyError(f"unknown site {key!r}; configured sites: {known}") from None


# ---------------------------------------------------------------------------
# models.yaml
# ---------------------------------------------------------------------------
Effort = Literal["low", "medium", "high", "xhigh", "max"]
Thinking = Literal["adaptive", "disabled"]

# Models that reject the `effort` parameter outright. Sending it is a 400.
_MODELS_WITHOUT_EFFORT = {"claude-haiku-4-5"}


class ModelConfig(BaseModel):
    """Per-node LLM settings.

    Note the absence of `temperature`. It was removed on Claude Opus 5,
    Sonnet 5, Opus 4.8 and Opus 4.7 — sending it returns HTTP 400. See
    docs/DECISION.md#0004.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    effort: Effort | None = None
    thinking: Thinking | None = None
    max_tokens: int = Field(..., gt=0)
    cache_system_prompt: bool = True

    @model_validator(mode="after")
    def _reject_unsupported_effort(self) -> ModelConfig:
        if self.effort is not None and self.model in _MODELS_WITHOUT_EFFORT:
            raise ValueError(
                f"{self.model} rejects the `effort` parameter (HTTP 400). "
                f"Set effort: null for this node, or pick a model that "
                f"supports it."
            )
        return self


class Pricing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: float = Field(..., ge=0.0, description="USD per million input tokens")
    output: float = Field(..., ge=0.0, description="USD per million output tokens")


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_llm_calls_per_investigation: int = Field(..., gt=0)
    max_cost_usd_per_investigation: float = Field(..., gt=0.0)
    max_planner_critic_cycles: int = Field(..., gt=0)
    max_tools_per_cycle: int = Field(..., gt=0)
    request_timeout_seconds: int = Field(..., gt=0)


class ResponseCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    path: str = "traces/llm_cache"


class ModelsConfig(BaseModel):
    """Whole-file model for ``config/models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    active_profile: str
    pricing: dict[str, Pricing]
    profiles: dict[str, dict[str, ModelConfig]]
    limits: Limits
    response_cache: ResponseCacheConfig

    @model_validator(mode="after")
    def _validate_profiles(self) -> ModelsConfig:
        required_nodes = {"planner", "router", "synthesizer", "critic"}
        for profile_name, nodes in self.profiles.items():
            missing = required_nodes - set(nodes)
            if missing:
                raise ValueError(
                    f"profile {profile_name!r} is missing node config for: "
                    + ", ".join(sorted(missing))
                )
            for node_name, node in nodes.items():
                if node.model not in self.pricing:
                    raise ValueError(
                        f"profile {profile_name!r} node {node_name!r} uses model "
                        f"{node.model!r}, which has no pricing entry — cost "
                        f"accounting would silently report $0"
                    )
        if self.active_profile not in self.profiles:
            raise ValueError(
                f"active_profile {self.active_profile!r} is not defined; "
                f"available: {', '.join(sorted(self.profiles))}"
            )
        return self

    def node(self, node_name: str, profile: str | None = None) -> ModelConfig:
        name = profile or self.active_profile
        try:
            return self.profiles[name][node_name]
        except KeyError:
            raise KeyError(
                f"no config for node {node_name!r} in profile {name!r}"
            ) from None

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.pricing[model]
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return loaded


@lru_cache(maxsize=1)
def load_site_defaults(path: Path | None = None) -> SiteDefaults:
    return SiteDefaults.model_validate(
        _read_yaml(path or CONFIG_DIR / "site_defaults.yaml")
    )


@lru_cache(maxsize=1)
def load_models_config(path: Path | None = None) -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(path or CONFIG_DIR / "models.yaml"))


def build_clock(defaults: SiteDefaults | None = None) -> ReplayClock:
    """Construct the replay clock from config. The one place it is created."""
    cfg = (defaults or load_site_defaults()).clock
    return ReplayClock(start=cfg.start, speed=cfg.speed)
