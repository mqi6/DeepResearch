import os
import atexit
from dataclasses import dataclass
from typing import Optional, List, Any

from dotenv import load_dotenv
from letta_client import Letta
from qwen_agent.tools.base import BaseTool, register_tool


@dataclass
class LettaConfig:
    api_key: str
    base_url: str = "https://api.letta.com"

    # Agent lifecycle
    agent_id: Optional[str] = None
    ephemeral: bool = True
    auto_delete: bool = True  # delete ephemeral agents on exit / reset

    # Cost controls (model handles are provider/model-name)
    # Letta accepts e.g. "openai/gpt-4o-mini" as the model handle.  :contentReference[oaicite:5]{index=5}
    model: str = "openai/gpt-4.1-nano"
    compaction_model: Optional[str] = None  # default to `model` if unset
    embedding: str = "openai/text-embedding-3-small"

    # If True, Letta clears message buffer after each response (saves tokens if you don't need dialogue context). :contentReference[oaicite:6]{index=6}
    message_buffer_autoclear: bool = True

    # Passage search tuning (embedding-based, no LLM)
    top_k: int = 6


def _get_bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y"}


def _get_int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@register_tool("letta")
class LettaTool(BaseTool):
    description = "Tool for interacting with Letta (MemGPT) via cloud API."
    parameters = [
        {
            "name": "message",
            "type": "string",
            "description": "Message to send to Letta agent.",
            "required": True,
        }
    ]

    def __init__(self, cfg: Optional[LettaConfig] = None):
        super().__init__()

        load_dotenv()

        if cfg is None:
            api_key = os.getenv("LETTA_API_KEY")
            if not api_key:
                raise ValueError("LETTA_API_KEY is required")

            base_url = os.getenv("LETTA_BASE_URL", "https://api.letta.com")
            agent_id = os.getenv("LETTA_AGENT_ID") or None
            ephemeral = _get_bool_env("LETTA_EPHEMERAL", True)

            # New env vars for cost control
            model = os.getenv("LETTA_MODEL", "openai/gpt-4.1-nano")
            compaction_model = os.getenv("LETTA_COMPACTION_MODEL") or None
            embedding = os.getenv("LETTA_EMBEDDING_MODEL", "openai/text-embedding-3-small")

            message_buffer_autoclear = _get_bool_env("LETTA_MESSAGE_BUFFER_AUTOCLEAR", True)
            top_k = _get_int_env("LETTA_TOP_K", 6)

            # Auto-delete ephemeral agents
            auto_delete = _get_bool_env("LETTA_AUTO_DELETE", True)

            cfg = LettaConfig(
                api_key=api_key,
                base_url=base_url,
                agent_id=agent_id,
                ephemeral=ephemeral,
                auto_delete=auto_delete,
                model=model,
                compaction_model=compaction_model,
                embedding=embedding,
                message_buffer_autoclear=message_buffer_autoclear,
                top_k=top_k,
            )

        self.cfg = cfg
        self.client = Letta(api_key=self.cfg.api_key, base_url=self.cfg.base_url)

        self.default_agent_id: Optional[str] = self.cfg.agent_id
        self.is_ephemeral: bool = self.cfg.ephemeral

        # Keep track of ephemeral agents to cleanup
        self._ephemeral_agents: List[str] = []
        atexit.register(self._cleanup_ephemeral_agents)

    def _cleanup_ephemeral_agents(self):
        if not self.cfg.auto_delete:
            return
        for agent_id in list(self._ephemeral_agents):
            try:
                # Delete agent endpoint exists in Letta API/SDK. :contentReference[oaicite:7]{index=7}
                self.client.agents.delete(agent_id)
            except Exception:
                pass  # best-effort cleanup

    def _create_agent_with_fallback(self) -> str:
        """
        Create an agent with the cheapest requested model.
        If the model isn't available on your Letta account yet, fall back to gpt-4o-mini.
        """
        preferred_models = [
            self.cfg.model,
            "openai/gpt-4o-mini",
        ]

        compaction_model = self.cfg.compaction_model or self.cfg.model

        last_err = None
        for m in preferred_models:
            try:
                agent = self.client.agents.create(
                    name="deepresearch-agent",

                    # These fields are supported on agent create:
                    # - embedding: optional string :contentReference[oaicite:8]{index=8}
                    # - message_buffer_autoclear: optional boolean :contentReference[oaicite:9]{index=9}
                    # - model: optional string (provider/model-name) :contentReference[oaicite:10]{index=10}
                    embedding=self.cfg.embedding,
                    message_buffer_autoclear=self.cfg.message_buffer_autoclear,
                    model=m,

                    # compaction_settings lets you choose the summarizer model handle :contentReference[oaicite:11]{index=11}
                    compaction_settings={"model": compaction_model},
                )
                agent_id = agent.id
                if self.is_ephemeral:
                    self._ephemeral_agents.append(agent_id)
                return agent_id
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to create Letta agent with any preferred model. Last error: {last_err}")

    def _ensure_agent(self) -> str:
        if not self.default_agent_id:
            self.default_agent_id = self._create_agent_with_fallback()
        return self.default_agent_id

    def reset_agent(self):
        """
        Force-create a new agent (useful if you want one agent per benchmark question).
        """
        if self.default_agent_id and self.cfg.auto_delete:
            try:
                self.client.agents.delete(self.default_agent_id)  # :contentReference[oaicite:12]{index=12}
            except Exception:
                pass
        self.default_agent_id = None
        self._ensure_agent()

    def call(self, params: dict, **kwargs) -> str:
        message = params["message"]
        agent_id = self._ensure_agent()

        # Send user message to the agent
        response = self.client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": message}],
        )

        # Extract the last assistant message
        assistant_messages: List[Any] = []
        for msg in getattr(response, "messages", []) or []:
            msg_type = getattr(msg, "message_type", None)
            if msg_type == "assistant_message":
                assistant_messages.append(msg)

        if not assistant_messages:
            return "No assistant response."

        last = assistant_messages[-1]
        parts = getattr(last, "content", []) or []
        text_chunks = []
        for p in parts:
            if getattr(p, "type", None) == "text":
                text_chunks.append(getattr(p, "text", ""))

        return "\n".join([t for t in text_chunks if t]).strip() or "No assistant text."

    # -------------------------
    # Cheap (no-LLM) memory ops:
    # -------------------------

    def query_memory(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Semantic search of archival memory via the API (embedding-based).
        This avoids an LLM call entirely. :contentReference[oaicite:13]{index=13}
        """
        agent_id = self._ensure_agent()
        k = top_k or self.cfg.top_k

        try:
            resp = self.client.agents.passages.search(agent_id, query=query, top_k=k)
            # Response shape can vary across SDK versions; handle dict-like and attr-like.
            results = getattr(resp, "results", None)
            if results is None and isinstance(resp, dict):
                results = resp.get("results")

            if not results:
                return "No relevant memory."

            lines = []
            for r in results:
                content = getattr(r, "content", None)
                if content is None and isinstance(r, dict):
                    content = r.get("content")
                if content:
                    lines.append(content.strip())

            return "\n\n".join(lines) if lines else "No relevant memory."
        except Exception:
            # If passages.search isn't available in your SDK, fall back to agent LLM call
            return self.call({"message": f"Search your memory for: {query}"})

    def save_memory(self, memory: str, tags: Optional[List[str]] = None) -> str:
        """
        Insert a passage into archival memory via the API (embedding-based).
        This avoids an LLM call entirely. :contentReference[oaicite:14]{index=14}
        """
        agent_id = self._ensure_agent()
        try:
            self.client.agents.passages.create(
                agent_id,
                text=memory,
                tags=tags or [],
            )
            return "Memory saved."
        except Exception:
            # Fall back to agent LLM call if needed
            return self.call({"message": f"Please save the following to memory:\n\n{memory}"})

    def list_memory(self, limit: int = 10) -> str:
        agent_id = self._ensure_agent()
        response = self.client.agents.messages.list(agent_id=agent_id, limit=limit)
        messages = getattr(response, "messages", []) or []
        return f"Retrieved {len(messages)} messages."
