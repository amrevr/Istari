"""Swarm configuration: the object that interventions modify.

A SwarmConfig describes *which* components are active and *what* failures
are injected.  The recorder enforces most of it at runtime (a disabled agent
cannot be entered, a disabled tool cannot be called, a dropped message is not
delivered) so that counterfactual experiments do not depend on every swarm
implementing intervention logic itself.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


FAILURE_KINDS = {
    "agent_crash",      # agent raises on entry
    "agent_delay",      # agent entry is delayed
    "llm_error",        # llm call raises
    "llm_delay",        # llm call is delayed
    "tool_error",       # tool call raises
    "tool_malformed",   # tool returns `payload` instead of its real result (call_tool only)
    "tool_delay",       # tool call is delayed
    "message_drop",     # message is recorded but not delivered
    "message_corrupt",  # message content replaced with payload
}


@dataclass
class FailureSpec:
    """Describes an injected failure.

    target semantics by kind:
      agent_*      -> agent id ("*" = any agent)
      llm_*        -> agent id ("*" = any agent)
      tool_*       -> tool name ("*" = any tool)
      message_*    -> "sender->receiver", receiver id, or "*"
    """
    kind: str
    target: str = "*"
    probability: float = 1.0
    on_call: Optional[int] = None     # fire only on the Nth matching call (1-based)
    delay_s: float = 0.0
    payload: Any = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise ValueError(f"unknown failure kind {self.kind!r}; expected one of {sorted(FAILURE_KINDS)}")

    @property
    def key(self) -> str:
        return self.label or f"{self.kind}:{self.target}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SwarmConfig:
    label: str = "baseline"
    disabled_agents: Set[str] = field(default_factory=set)
    disabled_tools: Set[str] = field(default_factory=set)
    disabled_channels: Set[str] = field(default_factory=set)   # "sender->receiver"
    model_overrides: Dict[str, str] = field(default_factory=dict)
    token_budgets: Dict[str, int] = field(default_factory=dict)  # agent -> max total tokens
    context_limits: Dict[str, int] = field(default_factory=dict) # agent -> max input tokens per call
    failures: List[FailureSpec] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)        # swarm-specific knobs
    interventions: List[str] = field(default_factory=list)      # human-readable trail

    def copy(self) -> "SwarmConfig":
        return copy.deepcopy(self)

    # -- queries --------------------------------------------------------------
    def agent_enabled(self, agent_id: str) -> bool:
        return agent_id not in self.disabled_agents

    def tool_enabled(self, tool: str) -> bool:
        return tool not in self.disabled_tools

    def channel_open(self, sender: str, receiver: str) -> bool:
        return (f"{sender}->{receiver}" not in self.disabled_channels
                and f"*->{receiver}" not in self.disabled_channels
                and f"{sender}->*" not in self.disabled_channels)

    def model_for(self, agent_id: str, default: Optional[str] = None) -> Optional[str]:
        return self.model_overrides.get(agent_id, self.model_overrides.get("*", default))

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def is_baseline(self) -> bool:
        return not (self.disabled_agents or self.disabled_tools or self.disabled_channels
                    or self.model_overrides or self.token_budgets or self.context_limits
                    or self.failures or self.params)

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "disabled_agents": sorted(self.disabled_agents),
            "disabled_tools": sorted(self.disabled_tools),
            "disabled_channels": sorted(self.disabled_channels),
            "model_overrides": dict(self.model_overrides),
            "token_budgets": dict(self.token_budgets),
            "context_limits": dict(self.context_limits),
            "failures": [f.to_dict() for f in self.failures],
            "params": copy.deepcopy(self.params),
            "interventions": list(self.interventions),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SwarmConfig":
        return cls(
            label=d.get("label", "baseline"),
            disabled_agents=set(d.get("disabled_agents", [])),
            disabled_tools=set(d.get("disabled_tools", [])),
            disabled_channels=set(d.get("disabled_channels", [])),
            model_overrides=dict(d.get("model_overrides", {})),
            token_budgets=dict(d.get("token_budgets", {})),
            context_limits=dict(d.get("context_limits", {})),
            failures=[FailureSpec.from_dict(f) for f in d.get("failures", [])],
            params=dict(d.get("params", {})),
            interventions=list(d.get("interventions", [])),
        )

    def describe(self) -> str:
        if self.is_baseline():
            return f"{self.label} (no interventions)"
        parts = []
        if self.disabled_agents:
            parts.append("agents-off=" + ",".join(sorted(self.disabled_agents)))
        if self.disabled_tools:
            parts.append("tools-off=" + ",".join(sorted(self.disabled_tools)))
        if self.disabled_channels:
            parts.append("channels-off=" + ",".join(sorted(self.disabled_channels)))
        if self.model_overrides:
            parts.append("models=" + ",".join(f"{k}:{v}" for k, v in sorted(self.model_overrides.items())))
        if self.token_budgets:
            parts.append("budgets=" + ",".join(f"{k}:{v}" for k, v in sorted(self.token_budgets.items())))
        if self.context_limits:
            parts.append("ctx-limits=" + ",".join(f"{k}:{v}" for k, v in sorted(self.context_limits.items())))
        if self.failures:
            parts.append("failures=" + ",".join(f.key for f in self.failures))
        if self.params:
            parts.append("params=" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items())))
        return f"{self.label}: " + "; ".join(parts)
