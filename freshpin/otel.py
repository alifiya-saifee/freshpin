"""Export agent-benchmark runs as OpenTelemetry traces (OTLP/JSON), so they open in any tool that reads OTLP
(Jaeger, Grafana Tempo, Langfuse, Arize Phoenix, Datadog ...).

One trace per run:
  invoke_agent <agent>              the whole run (gen_ai.operation.name = invoke_agent)
    chat <model>                    each model call, from Bifrost's log (tokens, errors, fallbacks)
    execute_tool search_web         each tool call, from FreshPin's tool server (query, injected fault)
FreshPin's own verdicts ride along as freshpin.* attributes (condition, grade, success, honesty, cost ...).

Attribute names follow the OpenTelemetry GenAI semantic conventions (gen_ai.*). Those conventions are still marked
"Development" upstream, so names may change; FreshPin pins the names it writes in GENAI_ATTRIBUTES below.
Standard library only.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

SPAN_KIND_INTERNAL, SPAN_KIND_CLIENT = 1, 3
STATUS_UNSET, STATUS_OK, STATUS_ERROR = 0, 1, 2

# The gen_ai.* names this exporter writes (checked by the tests, so a rename is a deliberate change).
GENAI_ATTRIBUTES = (
    "gen_ai.operation.name", "gen_ai.agent.name", "gen_ai.provider.name", "gen_ai.request.model",
    "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.tool.name", "gen_ai.tool.type",
)


def _id(text: str, hex_len: int) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:hex_len]


_TIME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*(Z|[+-]\d{2}:?\d{2})?$")


def _ns(t: Any) -> int | None:
    """A trace step's time as Unix nanoseconds. Accepts '2026-11-15 12:09:43.049358+00:00', ISO 8601 with 'Z', any
    number of fraction digits (Bifrost's log drops trailing zeros: '.57326', which Python 3.9's fromisoformat
    rejects) and epoch seconds. Unreadable: None."""
    if t is None or t == "" or isinstance(t, bool):
        return None
    if isinstance(t, (int, float)):
        return int(float(t) * 1e9)
    m = _TIME.match(str(t).strip())
    if not m:
        return None
    y, mo, d, h, mi, se, frac, tz = m.groups()
    offset = timezone.utc
    if tz and tz != "Z":
        sign = -1 if tz[0] == "-" else 1
        hh, mm = int(tz[1:3]), int(tz[-2:])
        offset = timezone(sign * timedelta(hours=hh, minutes=mm))
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(se), tzinfo=offset)
    except ValueError:
        return None
    return int(dt.timestamp()) * 1_000_000_000 + int((frac or "0").ljust(9, "0"))


def _val(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}          # OTLP/JSON writes 64-bit ints as strings
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attrs(d: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _val(v)} for k, v in d.items() if v is not None and v != ""]


def _span(trace_id: str, span_id: str, parent: str | None, name: str, kind: int, start: int, end: int,
          attrs: dict[str, Any], status: int = STATUS_UNSET, message: str = "") -> dict[str, Any]:
    s: dict[str, Any] = {"traceId": trace_id, "spanId": span_id, "name": name, "kind": kind,
                         "startTimeUnixNano": str(start), "endTimeUnixNano": str(max(end, start)),
                         "attributes": _attrs(attrs)}
    if parent:
        s["parentSpanId"] = parent
    if status:
        s["status"] = {"code": status, **({"message": message[:500]} if message else {})}
    return s


def run_spans(r: dict[str, Any], fallback_start_ns: int = 0) -> list[dict[str, Any]]:
    """The spans of one run: the agent span first, then one child per trace step, in order."""
    run_id = str(r.get("run_id") or f"{r.get('agent')}:{r.get('condition')}:{r.get('task_id')}:{r.get('repeat')}")
    trace_id, root_id = _id(run_id, 32), _id(run_id + ":agent", 16)
    children: list[dict[str, Any]] = []
    previous_end = fallback_start_ns
    for i, step in enumerate(r.get("trace") or []):
        start = _ns(step.get("t"))
        if start is None:                 # never drop a step: an unreadable time goes right after the previous one
            start = previous_end
        end = start + int(float(step.get("ms") or 0) * 1_000_000)
        previous_end = end
        sid = _id(f"{run_id}:{i}", 16)
        if step.get("kind") == "model":
            model = str(step.get("model") or "")
            failed = step.get("status") not in (None, "success")
            children.append(_span(trace_id, sid, root_id, f"chat {model}", SPAN_KIND_CLIENT, start, end, {
                "gen_ai.operation.name": "chat", "gen_ai.provider.name": model.split("/", 1)[0] or None,
                "gen_ai.request.model": model, "gen_ai.usage.input_tokens": int(step.get("tokens_in") or 0),
                "gen_ai.usage.output_tokens": int(step.get("tokens_out") or 0),
                "error.type": "model_error" if failed else None,
                "freshpin.cost_usd": float(step.get("cost_usd") or 0.0),
                "freshpin.list_cost_usd": step.get("list_cost_usd"),
                "freshpin.fallback_index": step.get("fallback_index"),
            }, STATUS_ERROR if failed else STATUS_UNSET, str(step.get("error") or "")))
        else:
            tool = str(step.get("tool") or "tool")
            code = step.get("status")
            failed = isinstance(code, int) and code >= 400
            children.append(_span(trace_id, sid, root_id, f"execute_tool {tool}", SPAN_KIND_INTERNAL, start, end, {
                "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool, "gen_ai.tool.type": "function",
                "http.response.status_code": code if isinstance(code, int) else None,
                "error.type": str(code) if failed else None,
                "freshpin.tool.query": step.get("query"), "freshpin.tool.fault": step.get("fault"),
                "freshpin.tool.source": step.get("source"),
            }, STATUS_ERROR if failed else STATUS_UNSET, f"HTTP {code}" if failed else ""))
    starts = [int(c["startTimeUnixNano"]) for c in children]
    ends = [int(c["endTimeUnixNano"]) for c in children]
    root_start = min(starts) if starts else fallback_start_ns
    root_end = max(ends + [root_start + int(float(r.get("seconds") or 0) * 1e9)])
    agent_error = r.get("agent_error")
    no_answer = not str(r.get("answer") or "").strip()
    root = _span(trace_id, root_id, None, f"invoke_agent {r.get('agent')}", SPAN_KIND_INTERNAL, root_start, root_end, {
        "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": r.get("agent"),
        "gen_ai.usage.input_tokens": sum(int(s.get("tokens_in") or 0) for s in r.get("trace") or [] if s.get("kind") == "model"),
        "gen_ai.usage.output_tokens": sum(int(s.get("tokens_out") or 0) for s in r.get("trace") or [] if s.get("kind") == "model"),
        "freshpin.run_id": run_id, "freshpin.task_id": r.get("task_id"), "freshpin.condition": r.get("condition"),
        "freshpin.repeat": r.get("repeat"), "freshpin.grade": r.get("grade"), "freshpin.success": bool(r.get("success")),
        "freshpin.honesty": r.get("honesty"), "freshpin.followed_planted": bool(r.get("followed_planted")),
        "freshpin.excluded": bool(r.get("infra_error") or r.get("grader_error") or r.get("graders_disagree")),
        "freshpin.steps": r.get("steps"), "freshpin.tool_calls": r.get("tool_calls"),
        "freshpin.cost_usd": r.get("cost_usd"), "freshpin.list_cost_usd": r.get("list_cost_usd"),
        "freshpin.seconds": r.get("seconds"),
    }, STATUS_ERROR if (agent_error or no_answer) else STATUS_UNSET,
        str(agent_error or ("no answer" if no_answer else "")))
    return [root] + children


def to_otlp(result: dict[str, Any], source: str = "") -> dict[str, Any]:
    """A whole agentbench result as one OTLP/JSON export request (the body of POST /v1/traces)."""
    meta = result.get("meta") or {}
    fallback = _ns(f"{meta['run_on']}T00:00:00+00:00") if meta.get("run_on") else 0
    spans = [s for r in result.get("runs") or [] for s in run_spans(r, fallback or 0)]
    resource = {"service.name": "freshpin-agentbench", "freshpin.result_file": source or None,
                "freshpin.model_under_test": meta.get("model"), "freshpin.grader": meta.get("grader"),
                "freshpin.repeats": meta.get("repeats")}
    return {"resourceSpans": [{"resource": {"attributes": _attrs(resource)},
                               "scopeSpans": [{"scope": {"name": "freshpin.agentbench", "version": "1"},
                                               "spans": spans}]}]}
