"""Memory strategy base class and factory.

Each memory strategy controls how the agent manages its conversation history
when context grows too large. This is one of the key research axes of AgentForge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import anthropic
import tiktoken

from agentforge.core.schemas import MemoryConfig, MemoryStrategy


class BaseMemory(ABC):
    """Abstract base for all memory strategies."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token count for a message list."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        if isinstance(text, str):
                            total += len(self._encoding.encode(text))
                    else:
                        # Anthropic content block objects
                        text = getattr(block, "text", "") or ""
                        if isinstance(text, str):
                            total += len(self._encoding.encode(text))
        return total

    def should_compact(
        self, messages: list[dict[str, Any]], max_tokens: int
    ) -> bool:
        """Check if context exceeds the compaction threshold."""
        current = self.count_tokens(messages)
        threshold = int(max_tokens * self.config.compact_threshold)
        return current > threshold

    @abstractmethod
    async def compact(
        self,
        messages: list[dict[str, Any]],
        client: anthropic.Anthropic,
    ) -> list[dict[str, Any]]:
        """Compact the message history to fit within context limits.

        Args:
            messages: Current conversation messages.
            client: Anthropic client for summarization calls.

        Returns:
            Compacted message list.
        """
        ...


class SlidingWindowMemory(BaseMemory):
    """Keeps only the most recent N message pairs, dropping oldest."""

    async def compact(
        self,
        messages: list[dict[str, Any]],
        client: anthropic.Anthropic,
    ) -> list[dict[str, Any]]:
        window = self.config.window_size * 2  # pairs of user/assistant
        if len(messages) <= window:
            return messages

        # Always keep the first message (task description)
        first = messages[0]
        recent = messages[-window:]

        return [first] + recent


class SummarizationMemory(BaseMemory):
    """Summarizes older context using a separate LLM call."""

    async def compact(
        self,
        messages: list[dict[str, Any]],
        client: anthropic.Anthropic,
    ) -> list[dict[str, Any]]:
        if len(messages) <= 4:
            return messages

        # Split: keep first message + last 6 messages, summarize the middle
        first = messages[0]
        middle = messages[1:-6]
        recent = messages[-6:]

        if not middle:
            return messages

        # Build a text representation of middle messages for summarization
        summary_input = _messages_to_text(middle)

        response = client.messages.create(
            model=self.config.summary_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the following agent-tool interaction history. "
                        "Focus on: what was attempted, what worked, what failed, "
                        "and the current state of the task. Be concise.\n\n"
                        f"{summary_input}"
                    ),
                }
            ],
        )

        summary_text = response.content[0].text

        # Replace middle with a summary message
        summary_message = {
            "role": "user",
            "content": f"[Previous context summary]: {summary_text}",
        }

        return [first, summary_message] + recent


class RAGMemory(BaseMemory):
    """Embeds past turns and retrieves relevant ones for the current context.

    NOTE: Requires the `rag` optional dependency (chromadb, sentence-transformers).
    Falls back to sliding window if not installed.
    """

    async def compact(
        self,
        messages: list[dict[str, Any]],
        client: anthropic.Anthropic,
    ) -> list[dict[str, Any]]:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            # Fallback to sliding window if RAG deps not installed
            fallback = SlidingWindowMemory(self.config)
            return await fallback.compact(messages, client)

        if len(messages) <= 6:
            return messages

        first = messages[0]
        recent = messages[-4:]
        older = messages[1:-4]

        if not older:
            return messages

        # Build embeddings for older messages
        SentenceTransformer("all-MiniLM-L6-v2")
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection("agent_memory")

        texts = []
        for i, msg in enumerate(older):
            text = _message_to_text(msg)
            if text.strip():
                texts.append(text)
                collection.add(
                    documents=[text],
                    ids=[f"msg_{i}"],
                )

        if not texts:
            return [first] + recent

        # Query with the most recent user message as the query
        query_text = _message_to_text(recent[0]) if recent else "current task"
        results = collection.query(query_texts=[query_text], n_results=min(5, len(texts)))

        retrieved_docs = results["documents"][0] if results["documents"] else []
        retrieved_context = "\n---\n".join(retrieved_docs)

        # Clean up
        chroma_client.delete_collection("agent_memory")

        context_message = {
            "role": "user",
            "content": f"[Retrieved relevant context from earlier steps]:\n{retrieved_context}",
        }

        return [first, context_message] + recent


class HybridMemory(BaseMemory):
    """Combines sliding window for recent + RAG for older context."""

    async def compact(
        self,
        messages: list[dict[str, Any]],
        client: anthropic.Anthropic,
    ) -> list[dict[str, Any]]:
        if len(messages) <= 8:
            return messages

        # Keep a small sliding window of recent messages
        first = messages[0]
        window_size = min(self.config.window_size, 6)
        recent = messages[-window_size:]
        older = messages[1:-window_size]

        if not older:
            return messages

        # Summarize the older messages (cheaper than full RAG)
        summary_input = _messages_to_text(older[:10])  # cap to avoid huge summaries

        response = client.messages.create(
            model=self.config.summary_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Provide a brief summary of these agent steps. "
                        "Focus on key findings and current state.\n\n"
                        f"{summary_input}"
                    ),
                }
            ],
        )

        summary = response.content[0].text

        context_message = {
            "role": "user",
            "content": f"[Context summary + recent window active]:\n{summary}",
        }

        return [first, context_message] + recent


# ─── Factory ──────────────────────────────────────────────────────────────────


class MemoryFactory:
    """Creates memory strategy instances from config."""

    _strategies: dict[MemoryStrategy, type[BaseMemory]] = {
        MemoryStrategy.SLIDING_WINDOW: SlidingWindowMemory,
        MemoryStrategy.SUMMARIZATION: SummarizationMemory,
        MemoryStrategy.RAG: RAGMemory,
        MemoryStrategy.HYBRID: HybridMemory,
    }

    @classmethod
    def create(cls, config: MemoryConfig) -> BaseMemory:
        strategy_cls = cls._strategies.get(config.strategy)
        if strategy_cls is None:
            raise ValueError(f"Unknown memory strategy: {config.strategy}")
        return strategy_cls(config)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _message_to_text(msg: dict[str, Any]) -> str:
    """Extract text from a message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or block.get("content", ""))
            else:
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Convert a list of messages to readable text."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        text = _message_to_text(msg)
        if text.strip():
            parts.append(f"[{role}]: {text[:500]}")
    return "\n\n".join(parts)
