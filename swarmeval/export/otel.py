"""OpenTelemetry-compatible export (OTLP/JSON structure).

Agent invocations become spans; LLM and tool calls become child spans with
``gen_ai.*`` attributes; messages, artifacts and waits become span events.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from ..schema import EventType, Trajectory


def _trace_id(run_id: str) -> str:
    return hashlib.sha1(run_id.encode()).hexdigest()[:32]


def _span_id(any_id: str) -> str:
    return hashlib.sha1(any_id.encode()).hexdigest()[:16]


def _attr(key: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _ns(t: float) -> str:
    return str(int(t * 1e9))


def to_otel(traj: Trajectory) -> Dict[str, Any]:
    trace_id = _trace_id(traj.run_id)
    spans: List[Dict[str, Any]] = []
    root_id = _span_id(traj.run_id)
    started = traj.started_at or 0.0
    ended = traj.ended_at or started
    spans.append({
        "traceId": trace_id, "spanId": root_id, "name": f"swarm_run:{traj.task_id}",
        "kind": 1, "startTimeUnixNano": _ns(started), "endTimeUnixNano": _ns(ended),
        "attributes": [_attr("swarmeval.run_id", traj.run_id), _attr("swarmeval.task_id", traj.task_id),
                       _attr("swarmeval.status", traj.status),
                       _attr("swarmeval.success", bool(traj.evaluation.success) if traj.evaluation else False)],
        "status": {"code": 2 if traj.status == "error" else 1},
        "events": [],
    })
    span_events: Dict[str, List[Dict[str, Any]]] = {root_id: spans[0]["events"]}
    for s in traj.spans.values():
        sid = _span_id(s.span_id)
        parent = _span_id(s.parent_span_id) if s.parent_span_id else root_id
        ev_list: List[Dict[str, Any]] = []
        span_events[sid] = ev_list
        spans.append({
            "traceId": trace_id, "spanId": sid, "parentSpanId": parent, "name": f"agent:{s.agent_id}",
            "kind": 1, "startTimeUnixNano": _ns(s.start), "endTimeUnixNano": _ns(s.end or s.start),
            "attributes": [_attr("gen_ai.agent.id", s.agent_id), _attr("gen_ai.agent.role", s.role or ""),
                           _attr("gen_ai.request.model", s.model or ""), _attr("swarmeval.status", s.status)],
            "status": {"code": 2 if s.status in ("error", "injected") else 1,
                       "message": s.error or ""},
            "events": ev_list,
        })
    for ev in traj.events:
        parent = _span_id(ev.span_id) if ev.span_id and ev.span_id in traj.spans else root_id
        if ev.event_type == EventType.LLM_CALL.value:
            spans.append({
                "traceId": trace_id, "spanId": _span_id(ev.event_id), "parentSpanId": parent,
                "name": "llm_call", "kind": 3,
                "startTimeUnixNano": _ns(ev.timestamp), "endTimeUnixNano": _ns(ev.end_timestamp),
                "attributes": [_attr("gen_ai.operation.name", "chat"), _attr("gen_ai.request.model", ev.model or ""),
                               _attr("gen_ai.usage.input_tokens", ev.input_tokens),
                               _attr("gen_ai.usage.output_tokens", ev.output_tokens),
                               _attr("swarmeval.cost_usd", ev.cost_usd if ev.cost_usd is not None else -1.0),
                               _attr("swarmeval.status", ev.status)],
                "status": {"code": 2 if ev.status == "error" else 1, "message": ev.error or ""},
                "events": [],
            })
        elif ev.event_type == EventType.TOOL_CALL.value:
            spans.append({
                "traceId": trace_id, "spanId": _span_id(ev.event_id), "parentSpanId": parent,
                "name": f"tool:{ev.tool_name}", "kind": 3,
                "startTimeUnixNano": _ns(ev.timestamp), "endTimeUnixNano": _ns(ev.end_timestamp),
                "attributes": [_attr("gen_ai.operation.name", "execute_tool"), _attr("gen_ai.tool.name", ev.tool_name or ""),
                               _attr("gen_ai.tool.call.arguments", json.dumps(ev.tool_args, default=str)[:2000]),
                               _attr("swarmeval.status", ev.status)],
                "status": {"code": 2 if ev.status == "error" else 1, "message": ev.error or ""},
                "events": [],
            })
        elif ev.event_type in (EventType.MESSAGE.value, EventType.ARTIFACT.value, EventType.ARTIFACT_USE.value,
                               EventType.WAIT.value, EventType.ERROR.value, EventType.INTERVENTION.value):
            span_events.get(parent, spans[0]["events"]).append({
                "name": ev.event_type, "timeUnixNano": _ns(ev.timestamp),
                "attributes": [a for a in (
                    _attr("sender", ev.sender) if ev.sender else None,
                    _attr("receiver", ev.receiver) if ev.receiver else None,
                    _attr("kind", ev.kind) if ev.kind else None,
                    _attr("tokens", ev.content_tokens) if ev.content_tokens else None,
                    _attr("status", ev.status),
                    _attr("error", ev.error) if ev.error else None,
                ) if a is not None],
            })
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", "swarmeval"), _attr("swarmeval.benchmark_task", traj.task_id)]},
        "scopeSpans": [{"scope": {"name": "swarmeval", "version": "0.1.0"}, "spans": spans}],
    }]}
