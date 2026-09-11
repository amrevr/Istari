"""Interventions: composable edits to a SwarmConfig.

An intervention is the unit of a counterfactual experiment: "the same swarm,
the same benchmark, the same seeds -- except for this one change."
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from ..config import FailureSpec, SwarmConfig


class Intervention:
    name: str = "intervention"

    def apply(self, config: SwarmConfig) -> SwarmConfig:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, config: SwarmConfig) -> SwarmConfig:
        cfg = self.apply(config.copy())
        cfg.interventions.append(self.name)
        return cfg

    def __repr__(self) -> str:
        return self.name


class RemoveAgent(Intervention):
    def __init__(self, *agent_ids: str) -> None:
        self.agent_ids = list(agent_ids)
        self.name = "remove_agent:" + ",".join(agent_ids)

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.disabled_agents.update(self.agent_ids)
        return config


class RemoveTool(Intervention):
    def __init__(self, *tools: str) -> None:
        self.tools = list(tools)
        self.name = "remove_tool:" + ",".join(tools)

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.disabled_tools.update(self.tools)
        return config


class DisableChannel(Intervention):
    def __init__(self, sender: str, receiver: str) -> None:
        self.channel = f"{sender}->{receiver}"
        self.name = f"disable_channel:{self.channel}"

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.disabled_channels.add(self.channel)
        return config


class SetModel(Intervention):
    def __init__(self, agent_id: str, model: str) -> None:
        self.agent_id = agent_id
        self.model = model
        self.name = f"set_model:{agent_id}={model}"

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.model_overrides[self.agent_id] = self.model
        return config


class SetTokenBudget(Intervention):
    def __init__(self, agent_id: str, budget: int) -> None:
        self.agent_id = agent_id
        self.budget = budget
        self.name = f"token_budget:{agent_id}={budget}"

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.token_budgets[self.agent_id] = self.budget
        return config


class SetContextLimit(Intervention):
    def __init__(self, agent_id: str, limit: int) -> None:
        self.agent_id = agent_id
        self.limit = limit
        self.name = f"context_limit:{agent_id}={limit}"

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.context_limits[self.agent_id] = self.limit
        return config


class SetParam(Intervention):
    """Set a swarm-specific parameter (read via ``ctx.param``)."""

    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        self.value = value
        self.name = f"param:{key}={value}"

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.params[self.key] = self.value
        return config


class InjectFailure(Intervention):
    def __init__(self, kind: str, target: str = "*", probability: float = 1.0, on_call: Optional[int] = None,
                 delay_s: float = 0.0, payload: Any = None, label: Optional[str] = None) -> None:
        self.spec = FailureSpec(kind, target, probability, on_call, delay_s, payload, label)
        self.name = f"inject:{self.spec.key}"

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        config.failures.append(self.spec)
        return config


class Compose(Intervention):
    def __init__(self, *interventions: Intervention, name: Optional[str] = None) -> None:
        self.interventions = list(interventions)
        self.name = name or "+".join(i.name for i in interventions)

    def apply(self, config: SwarmConfig) -> SwarmConfig:
        for i in self.interventions:
            config = i.apply(config)
        return config


def apply_interventions(base: SwarmConfig, interventions: Iterable[Intervention],
                        label: Optional[str] = None) -> SwarmConfig:
    cfg = base.copy()
    names: List[str] = []
    for i in interventions:
        cfg = i(cfg)
        names.append(i.name)
    cfg.label = label or (" + ".join(names) if names else base.label)
    return cfg
