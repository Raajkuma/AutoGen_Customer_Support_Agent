from __future__ import annotations

import json

from autogen_core import CancellationToken

import parsing
import tools
from agents.base import StageAgent


class KnowledgeAgent(StageAgent):
    """Calls the search_kb AutoGen tool itself, so retrieval never depends on a small model's tool-calling."""
    stage = "knowledge"

    def __init__(self, tracer) -> None:
        super().__init__("knowledge_agent", "Searches company policy documents; returns cited passages.", tracer)

    async def act(self) -> str:
        q = parsing.parse_query(self.said("query_agent"))
        complaint = parsing.parse_task(self.said("user"))["complaint"]
        args = {"query": f"{q['summary']} {complaint}", "category": q["category"]}
        data = json.loads(await tools.search_kb_tool.run_json(args, CancellationToken()))
        self.tracer.tool(self.stage, "search_kb", args, data["passages"])
        self.tracer.note(self.stage, f"retrieval {data['mode']}", matched=data["matched"], mode=data["mode"],
                         chunk_ids=data["chunk_ids"], scores=data["debug"]["hits"])
        return data["passages"]
