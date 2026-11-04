"""Shared base for the four support agents (AutoGen BaseChatAgent + workflow tracing)."""
from __future__ import annotations

from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import SystemMessage, UserMessage

import parsing


class StageAgent(BaseChatAgent):
    """One workflow stage: traces start/finish/failure around `act`."""
    stage = ""

    def __init__(self, name: str, description: str, tracer):
        super().__init__(name=name, description=description)
        self.tracer = tracer
        self.history: list[BaseChatMessage] = []

    @property
    def produced_message_types(self):
        return (TextMessage,)

    def said(self, source: str) -> str:
        """Latest text a given participant posted in the shared conversation."""
        for m in reversed(self.history):
            if m.source == source and isinstance(getattr(m, "content", None), str):
                return m.content
        return ""

    async def act(self) -> str:
        raise NotImplementedError

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        self.history.extend(messages)
        self.tracer.start(self.stage)
        try:
            text = await self.act()
        except Exception as exc:
            self.tracer.fail(self.stage, exc)
            raise
        self.tracer.finish(self.stage, text)
        return Response(chat_message=TextMessage(content=text, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self.history.clear()


class LLMStageAgent(StageAgent):
    """Stage that asks the local model; strips <think> blocks and records latency/tokens."""

    def __init__(self, name, description, tracer, client, system_message):
        super().__init__(name, description, tracer)
        self.client, self.system_message = client, system_message

    async def ask(self, prompt: str) -> str:
        result = await self.client.create([SystemMessage(content=self.system_message),
                                          UserMessage(content=prompt, source="user")])
        usage = getattr(result, "usage", None)
        if usage:
            self.tracer.note(self.stage, "model replied", prompt_tokens=usage.prompt_tokens,
                             completion_tokens=usage.completion_tokens)
        return parsing.strip_think(result.content if isinstance(result.content, str) else str(result.content))
