"""Workflow tracing: per-stage state, timing, tool calls and events, persisted as JSON per run."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import config

STAGES = [("query", "Query agent", "🔎"), ("knowledge", "Knowledge agent", "📚"),
          ("resolution", "Resolution agent", "✍️"), ("log", "Log agent", "🗂️"),
          ("approval", "Human approval", "🧑‍💼")]
STATE_ICON = {"pending": "⚪", "running": "🔄", "done": "✅", "waiting": "⏸️", "skipped": "⏭️", "failed": "❌"}
STATE_COLOR = {"pending": "#E5E7EB", "running": "#FDE68A", "done": "#BBF7D0",
               "waiting": "#FED7AA", "skipped": "#F3F4F6", "failed": "#FCA5A5"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Trace:
    def __init__(self, customer: str = "", order_id: str = "", complaint: str = "",
                 on_update: Callable[[Trace], None] | None = None) -> None:
        self.run_id = uuid.uuid4().hex[:8]
        self.ticket_id = ""
        self.created = _now()
        self.customer, self.order_id, self.complaint = customer, order_id, complaint
        self.model = config.MODEL
        self.stages: dict[str, dict[str, Any]] = {
            k: {"state": "pending", "started": None, "ended": None, "seconds": None,
                "output": "", "detail": {}, "tools": []} for k, _, _ in STAGES}
        self.events: list[dict[str, str]] = []
        self.on_update = on_update
        self._t0: dict[str, float] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        t = cls()
        t.__dict__.update(data)
        return t

    def _emit(self, stage: str, kind: str, message: str) -> None:
        self.events.append({"time": _now(), "stage": stage, "kind": kind, "message": message})
        if self.on_update:
            self.on_update(self)

    def start(self, stage: str) -> None:
        self.stages[stage].update(state="running", started=_now())
        self._t0[stage] = time.perf_counter()
        self._emit(stage, "start", "started")

    def finish(self, stage: str, output: str = "", **detail: Any) -> None:
        s = self.stages[stage]
        s["seconds"] = round(time.perf_counter() - self._t0.get(stage, time.perf_counter()), 2)
        if s["state"] == "running":
            s["state"] = "done"
        s.update(ended=_now(), output=output)
        s["detail"].update(detail)
        self._emit(stage, "end", f"{s['state']} in {s['seconds']}s")

    def fail(self, stage: str, err: Exception) -> None:
        s = self.stages[stage]
        s.update(state="failed", ended=_now(), output=f"{type(err).__name__}: {err}")
        self._emit(stage, "error", s["output"])

    def tool(self, stage: str, name: str, args: dict[str, Any], result: Any) -> None:
        self.stages[stage]["tools"].append({"name": name, "args": args, "result": str(result)[:1500]})
        self._emit(stage, "tool", f"{name}({json.dumps(args, default=str)[:120]})")

    def note(self, stage: str, message: str, **detail: Any) -> None:
        self.stages[stage]["detail"].update(detail)
        self._emit(stage, "note", message)

    def wait(self, stage: str, message: str) -> None:
        self.stages[stage].update(state="waiting", started=_now())
        self._emit(stage, "wait", message)

    def skip(self, stage: str, message: str) -> None:
        self.stages[stage].update(state="skipped", ended=_now(), output=message)
        self._emit(stage, "skip", message)

    def human(self, decision: str, reviewer: str, text: str = "") -> None:
        self.stages["approval"].update(state="done", ended=_now(), output=text)
        self.stages["approval"]["detail"].update(decision=decision, reviewer=reviewer)
        self._emit("approval", "human", f"{decision} by {reviewer}")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ("on_update", "_t0")}

    def save(self) -> None:
        config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TRACE_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


def list_traces() -> list[dict[str, Any]]:
    if not config.TRACE_DIR.exists():
        return []
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in config.TRACE_DIR.glob("*.json")]
    return sorted(runs, key=lambda r: r["created"], reverse=True)


def record_human_decision(ticket_id: str, decision: str, reviewer: str, text: str) -> None:
    """Update the saved trace of a ticket once a human has reviewed it."""
    for run in list_traces():
        if run["ticket_id"] == ticket_id:
            trace = Trace.from_dict(run)
            trace.human(decision, reviewer, text)
            trace.save()
            return


def to_dot(run: dict[str, Any]) -> str:
    def node(key: str, label: str) -> str:
        s = run["stages"][key]
        secs = f" · {s['seconds']}s" if s.get("seconds") else ""
        return f'{key} [label="{label}\\n{s["state"]}{secs}", fillcolor="{STATE_COLOR[s["state"]]}"];'
    lines = ['digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             'cust [label="Customer\\ncomplaint", fillcolor="#DBEAFE"];',
             'reply [label="Reply to\\ncustomer", fillcolor="#DBEAFE"];']
    lines += [node(k, label) for k, label, _ in STAGES]
    lines += ["cust -> query -> knowledge -> resolution -> log;", 'log -> approval [label="escalated"];',
              "approval -> reply;", 'log -> reply [label="resolved", style=dashed];', "}"]
    return "\n".join(lines)
