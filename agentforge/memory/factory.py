"""Memory strategies and factory for AgentForge."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from agentforge.core.schemas import MemoryConfig, MemoryStrategy


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken, with a fallback estimator."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


class BaseMemory(ABC):
    """Abstract base class for memory strategies."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count total tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += _count_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += _count_tokens(block.get("text", str(block)))
                    else:
                        total += _count_tokens(str(block))
            else:
                total += _count_tokens(str(content))
        return total

    def should_compact(self, messages: list[dict[str, Any]]) -> bool:
        """Return True if messages exceed the compaction threshold."""
        tokens = self.count_tokens(messages)
        threshold = int(self.config.max_context_tokens * self.config.compact_threshold)
        return tokens > threshold

    @abstractmethod
    def compact(
        self, messages: list[dict[str, Any]],
        client: Any = None, model: str = "",
    ) -> list[dict[str, Any]]:
        """Compact messages to fit within the token budget."""


class SlidingWindowMemory(BaseMemory):
    """Keep the first message and the most recent N*2 messages."""

    def compact(
        self, messages: list[dict[str, Any]],
        client: Any = None, model: str = "",
    ) -> list[dict[str, Any]]:
        if len(messages) <= 3:
            return messages
        keep = max(4, len(messages) // 2)
        return [messages[0]] + messages[-keep:]


class SummarizationMemory(BaseMemory):
    """Summarize older messages using an LLM call."""

    def compact(
        self, messages: list[dict[str, Any]],
        client: Any = None, model: str = "",
    ) -> list[dict[str, Any]]:
        if len(messages) <= 3:
            return messages
        mid = len(messages) // 2
        older = messages[1:mid]
        recent = messages[mid:]
        summary_text = self._summarize(older, client, model)
        summary_msg = {
            "role": "user",
            "content": f"[Summary of prior conversation]\n{summary_text}",
        }
        return [messages[0], summary_msg] + recent

    @staticmethod
    def _summarize(
        messages: list[dict[str, Any]],
        client: Any = None, model: str = "",
    ) -> str:
        """Summarize a list of messages."""
        if client is not None and model:
            try:
                text = json.dumps([
                    {"role": m.get("role", ""),
                     "content": str(m.get("content", ""))[:500]}
                    for m in messages
                ])
                resp = client.messages.create(
                    model=model, max_tokens=300,
                    messages=[{"role": "user",
                               "content": f"Summarize this conversation concisely:\n{text}"}],
                )
                return resp.content[0].text
            except Exception:
                pass
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))[:200]
            parts.append(f"{role}: {content}")
        return "Conversation summary:\n" + "\n".join(parts)


class RAGMemory(BaseMemory):
    """Embed older turns and retrieve relevant ones with ChromaDB."""

    def compact(
        self, messages: list[dict[str, Any]],
        client: Any = None, model: str = "",
    ) -> list[dict[str, Any]]:
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return SlidingWindowMemory(self.config).compact(messages, client, model)
        if len(messages) <= 3:
            return messages
        recent = messages[-4:]
        older = messages[1:-4]
        query = str(recent[-1].get("content", ""))[:500]
        try:
            chroma = chromadb.Client()
            collection = chroma.get_or_create_collection("memory")
            docs = [str(m.get("content", ""))[:500] for m in older]
            ids = [f"msg_{i}" for i in range(len(docs))]
            collection.upsert(documents=docs, ids=ids)
            results = collection.query(query_texts=[query], n_results=min(3, len(docs)))
            retrieved = results.get("documents", [[]])[0]
            context_msg = {
                "role": "user",
                "content": "[Retrieved context]\n" + "\n---\n".join(retrieved),
            }
            return [messages[0], context_msg] + recent
        except Exception:
            return SlidingWindowMemory(self.config).compact(messages, client, model)


class HybridMemory(BaseMemory):
    """Combine sliding window for recent + summarization for older."""

    def compact(
        self, messages: list[dict[str, Any]],
        client: Any = None, model: str = "",
    ) -> list[dict[str, Any]]:
        if len(messages) <= 3:
            return messages
        keep = max(4, len(messages) // 3)
        recent = messages[-keep:]
        older = messages[1:-keep]
        summary = SummarizationMemory._summarize(older, client, model)
        summary_msg = {
            "role": "user",
            "content": f"[Summary of prior context]\n{summary}",
        }
        return [messages[0], summary_msg] + recent


class MemoryFactory:
    """Factory to create memory strategies from config."""

    _strategies: dict[MemoryStrategy, type[BaseMemory]] = {
        MemoryStrategy.SLIDING_WINDOW: SlidingWindowMemory,
        MemoryStrategy.SUMMARIZATION: SummarizationMemory,
        MemoryStrategy.RAG: RAGMemory,
        MemoryStrategy.HYBRID: HybridMemory,
    }

    @classmethod
    def create(cls, config: MemoryConfig) -> BaseMemory:
        """Create a memory strategy instance from config."""
        strategy_cls = cls._strategies.get(config.strategy)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        return strategy_cls(config)
