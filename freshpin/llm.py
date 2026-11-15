"""LLM access through the Bifrost gateway.

Every model call goes through Bifrost's OpenAI-compatible endpoint (/v1/chat/completions).
Resilience is layered:
  1. Bifrost-level fallbacks: we send a `fallbacks` list, so Bifrost retries another
     provider/model itself if the primary fails.
  2. Client-level failover: if Bifrost returns an error for the whole request, we try the
     next model in our list.
  3. Graceful degradation: if every model is down, the pipeline still returns Tavily evidence
     and marks claims "unverified (AI unavailable)" instead of failing.
A per-run token budget stops runaway spend; an optional Bifrost virtual key adds gateway-side
budgets and rate limits.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMUnavailable(RuntimeError):
    """Raised when no model could answer."""


class BudgetExceeded(RuntimeError):
    """Raised when a run would go over its token budget."""


@dataclass
class CallTrace:
    step: str
    requested_model: str
    served_by: str | None = None
    fallback_used: bool = False
    client_failover: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    ok: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RunBudget:
    limit_tokens: int
    used_tokens: int = 0
    traces: list[CallTrace] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, tokens: int) -> None:
        with self._lock:
            self.used_tokens += tokens

    def log(self, trace: CallTrace) -> None:
        with self._lock:
            self.traces.append(trace)

    def check(self) -> None:
        if self.used_tokens >= self.limit_tokens:
            raise BudgetExceeded(f"run token budget of {self.limit_tokens} reached ({self.used_tokens} used)")


class ChatBackend(Protocol):
    def chat_json(self, step: str, models: list[str], system: str, user: str, budget: RunBudget,
                  simulate_outage: bool = False) -> dict[str, Any]: ...


def message_text(content: Any) -> str:
    """Providers return the reply as a string or as a list of parts ([{"type": "text", "text": "..."}])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def parse_json_reply(text: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in prose or code fences; pull out the first JSON object."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise
        # The first complete object wins: some models add prose, or a second object, after it.
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj


class BifrostClient:
    def __init__(self, base_url: str, virtual_key: str = "", timeout_s: int = 60,
                 outage_model: str = "openai/model-that-does-not-exist",
                 no_thinking_prefixes: tuple[str, ...] = ()):
        self.base_url = base_url.rstrip("/")
        self.virtual_key = virtual_key
        self.timeout_s = timeout_s
        self.outage_model = outage_model
        self.retry_wait_s = 3.0  # wait before the one retry after a rate limit (HTTP 429)
        # Models (by "provider/" prefix) that get "thinking off". Judges need short, clean JSON: thinking makes
        # some models up to 10x slower and can break the JSON. Only some providers accept the setting
        # (NVIDIA does; Gemini and Groq reject it with HTTP 400), so it is sent only to these.
        self.no_thinking_prefixes = tuple(p for p in no_thinking_prefixes if p)

    def _no_thinking(self, model: str) -> bool:
        return any(model.startswith(p) for p in self.no_thinking_prefixes)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One retry after a short wait when the provider says "rate limit" (HTTP 429). Free tiers reset per minute,
        so a brief pause usually works, and it beats marking the claim unverified."""
        try:
            return self._post_once(payload)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            try:
                wait = float(exc.headers.get("Retry-After") or self.retry_wait_s)
            except (TypeError, ValueError):
                wait = self.retry_wait_s
            time.sleep(min(max(wait, 0.0), 10.0))
            return self._post_once(payload)

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.virtual_key:
            headers["x-bf-vk"] = self.virtual_key
        if "chat_template_kwargs" in payload:
            headers["x-bf-passthrough-extra-params"] = "true"  # Bifrost otherwise drops unknown fields
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode())

    def chat_json(self, step: str, models: list[str], system: str, user: str, budget: RunBudget,
                  simulate_outage: bool = False) -> dict[str, Any]:
        budget.check()
        chain = list(models)
        if simulate_outage:
            # Put a model that will fail first, so Bifrost has to use its fallbacks.
            chain = [self.outage_model] + chain
        last_error = "no models configured"
        for i in range(len(chain)):
            primary = chain[i]
            # Bifrost sends the same body to its fallbacks, so only list fallbacks that accept the same settings.
            # The others are still tried, one by one, by this loop (client failover).
            same = self._no_thinking(primary)
            fallbacks = [m for m in chain[i + 1:] if self._no_thinking(m) == same]
            trace = CallTrace(step=step, requested_model=primary, client_failover=i > 0)
            payload = {
                "model": primary,
                "fallbacks": fallbacks,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if same:
                payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
            t0 = time.monotonic()
            try:
                data = self._post(payload)
                trace.latency_ms = int((time.monotonic() - t0) * 1000)
                usage = data.get("usage") or {}
                trace.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                trace.completion_tokens = int(usage.get("completion_tokens") or 0)
                extra = data.get("extra_fields") or {}
                served = data.get("model") or primary
                provider = extra.get("provider")
                # Bifrost reports the provider separately. Always prefix it, because some model names
                # contain a slash themselves (Groq serves "openai/gpt-oss-120b").
                trace.served_by = (f"{provider}/{served}"
                                   if provider and not served.startswith(f"{provider}/") else served)
                trace.fallback_used = _model_differs(primary, trace.served_by)
                budget.charge(trace.prompt_tokens + trace.completion_tokens)
                budget.log(trace)
                content = message_text(data["choices"][0]["message"]["content"])
                reply = parse_json_reply(content)
                if not isinstance(reply, dict):
                    raise ValueError("reply is not a JSON object")
                return reply
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError, TypeError,
                    AttributeError, json.JSONDecodeError, ValueError) as exc:  # odd reply shape: try the next model
                trace.ok = False
                trace.latency_ms = int((time.monotonic() - t0) * 1000)
                trace.error = _describe(exc)
                budget.log(trace)
                last_error = trace.error
                continue
        raise LLMUnavailable(last_error)


    def chat_text(self, model: str, system: str, user: str, budget: RunBudget, step: str = "ask") -> str:
        """Ask ONE model for a plain-text answer, with no fallbacks: used by the bench to test that exact model."""
        budget.check()
        trace = CallTrace(step=step, requested_model=model)
        payload = {"model": model, "temperature": 0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        t0 = time.monotonic()
        try:
            data = self._post(payload)
            usage = data.get("usage") or {}
            trace.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            trace.completion_tokens = int(usage.get("completion_tokens") or 0)
            trace.served_by = data.get("model") or model
            budget.charge(trace.prompt_tokens + trace.completion_tokens)
            return message_text(data["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError, TypeError,
                AttributeError, ValueError) as exc:
            trace.ok, trace.error = False, _describe(exc)
            raise LLMUnavailable(trace.error) from exc
        finally:
            trace.latency_ms = int((time.monotonic() - t0) * 1000)
            budget.log(trace)


def _model_differs(requested: str, served: str | None) -> bool:
    """True when the gateway answered with a different provider or model than we asked for.
    Providers often append a date to the model name (gpt-4o-mini -> gpt-4o-mini-2024-07-18), so
    the served model only has to start with the requested one."""
    if not served:
        return False
    rp, _, rm = requested.lower().rpartition("/")
    sp, _, sm = served.lower().rpartition("/")
    if rp and sp and rp != sp:
        return True
    return not sm.startswith(rm)


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode()[:300]
        except Exception:  # noqa: BLE001
            body = ""
        return f"HTTP {exc.code}: {body}"
    return f"{type(exc).__name__}: {exc}"[:300]
