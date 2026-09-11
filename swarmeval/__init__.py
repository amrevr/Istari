"""SwarmEval -- evaluation, observability, diagnostics and causal analysis for
multi-agent AI systems.

Quick start::

    from swarmeval import evaluate
    result = evaluate(my_swarm, benchmark, trials=3)
    print(result.report())
"""
from . import evaluators
from .config import FailureSpec, SwarmConfig
from .evaluate import EvaluationResult, RunAnalysis, analyze_runs, analyze_trajectory, evaluate
from .recorder import (AgentDisabled, AgentSpan, BudgetExceeded, InjectedAgentCrash, InjectedFailure,
                       InjectedLLMError, InjectedToolError, InterventionError, LLMResult, Message, RunContext,
                       SimClock, ToolDisabled, WallClock)
from .runner import FunctionSwarm, RunOptions, RunSet, Swarm, SwarmRunner
from .schema import Artifact, Benchmark, Evaluation, Event, EventType, Span, Task, Trajectory
from .diagnostics import Diagnostic, SmellConfig
from .causal import (CounterfactualRunner, Compose, DisableChannel, InjectFailure, RemoveAgent, RemoveTool,
                     SetContextLimit, SetModel, SetParam, SetTokenBudget, ablate_agents, ablate_tools, compare,
                     frontier, inject_failures, intelligence_vs_coordination)
from .causal.experiments import ablate

__version__ = "0.1.0"

__all__ = [
    "evaluate", "analyze_runs", "analyze_trajectory", "EvaluationResult", "RunAnalysis",
    "Task", "Benchmark", "Trajectory", "Event", "EventType", "Span", "Artifact", "Evaluation",
    "RunContext", "AgentSpan", "Message", "LLMResult", "SimClock", "WallClock",
    "Swarm", "FunctionSwarm", "SwarmRunner", "RunOptions", "RunSet",
    "SwarmConfig", "FailureSpec",
    "InterventionError", "AgentDisabled", "ToolDisabled", "BudgetExceeded",
    "InjectedFailure", "InjectedAgentCrash", "InjectedLLMError", "InjectedToolError",
    "Diagnostic", "SmellConfig", "evaluators",
    "CounterfactualRunner", "RemoveAgent", "RemoveTool", "DisableChannel", "SetModel", "SetParam",
    "SetTokenBudget", "SetContextLimit", "InjectFailure", "Compose",
    "ablate", "ablate_agents", "ablate_tools", "compare", "frontier", "inject_failures", "intelligence_vs_coordination",
    "__version__",
]
