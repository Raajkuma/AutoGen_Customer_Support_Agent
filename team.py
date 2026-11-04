"""Wires the four agents into one AutoGen team: Query -> Knowledge -> Resolution -> Log."""
from __future__ import annotations

import asyncio
from typing import Callable

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

import config
from agents.knowledge_agent import KnowledgeAgent
from agents.log_agent import LogAgent
from agents.query_agent import QueryAgent
from agents.resolution_agent import ResolutionAgent
from llm import get_model_client


def build_team(client, tracer) -> RoundRobinGroupChat:
    stop = (TextMentionTermination("LOGGED", sources=["log_agent"])
            | TextMentionTermination("AWAITING_APPROVAL", sources=["log_agent"])
            | MaxMessageTermination(config.MAX_TEAM_MESSAGES))
    return RoundRobinGroupChat([QueryAgent(client, tracer), KnowledgeAgent(tracer),
                                ResolutionAgent(client, tracer), LogAgent(tracer)],
                               termination_condition=stop)


async def _run(task: str, tracer, on_message: Callable) -> None:
    client = get_model_client()
    try:
        async for msg in build_team(client, tracer).run_stream(task=task):
            if not isinstance(msg, TaskResult):
                on_message(msg)
    finally:
        await client.close()


def run_ticket(task: str, tracer, on_message: Callable = lambda m: None) -> None:
    """Synchronous entry point for Streamlit. A fresh team per ticket (teams hold state)."""
    asyncio.run(_run(task, tracer, on_message))
