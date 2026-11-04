from __future__ import annotations

import json

from autogen_core import CancellationToken

import parsing
import tools
from agents.base import StageAgent


class LogAgent(StageAgent):
    """Applies the escalation rules in code, writes the ticket, and opens the human-approval gate."""
    stage = "log"

    def __init__(self, tracer) -> None:
        super().__init__("log_agent", "Records the ticket and escalates complex cases to a human.", tracer)

    async def act(self) -> str:
        task = parsing.parse_task(self.said("user"))
        q = parsing.parse_query(self.said("query_agent"))
        res_text = self.said("resolution_agent")
        res, meta = parsing.parse_resolution(res_text), parsing.parse_resolution_meta(res_text)
        args = dict(customer_name=task["name"], order_id=task["order_id"], complaint=task["complaint"],
                    category=q["category"], urgency=q["urgency"], sentiment=q["sentiment"],
                    kb_sources=meta["sources"] or "NONE", response=res["reply"], confidence=res["confidence"],
                    citations=meta["citations"], language=parsing.detect_language(task["complaint"]))
        out = json.loads(await tools.log_ticket_tool.run_json(args, CancellationToken()))
        self.tracer.ticket_id = out["ticket_id"]
        self.tracer.tool(self.stage, "log_ticket", {k: args[k] for k in ("order_id", "category", "urgency")}, out)
        if out["status"] == "Escalated":
            reasons = "; ".join(out["reasons"])
            self.tracer.wait("approval", f"waiting for a human reviewer ({reasons})")
            return f"AWAITING_APPROVAL: {out['ticket_id']} | Escalated | {reasons}"
        self.tracer.skip("approval", "not required: ticket resolved automatically")
        return f"LOGGED: {out['ticket_id']} | Resolved"
