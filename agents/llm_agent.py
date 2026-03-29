"""LLM agent base class for TradingAgents-style multi-agent system.

Supports:
- vLLM (local GPU, tensor-parallel across 2×3090)
- Ollama (local CPU/GPU, smaller models)
- Anthropic API (Claude 4.6)
- OpenAI API (GPT-4.x)

Designed for HFT: minimal latency, structured JSON outputs.
"""

from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from enum import Enum


class ModelProvider(Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass
class LLMConfig:
    """LLM configuration for multi-GPU HFT setup."""
    provider: ModelProvider = ModelProvider.VLLM
    model: str = "deepseek-ai/DeepSeek-Math-V2-70B-Instruct"
    # vLLM specific
    vllm_url: str = "http://localhost:8000/v1"
    tensor_parallel: int = 2  # 2×3090
    max_tokens: int = 1024
    temperature: float = 0.1  # Low for deterministic analysis
    # Ollama specific
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral-nemo:12b"  # Fits on CMP 50HX 10GB
    # API keys
    anthropic_key: Optional[str] = field(default=None)
    openai_key: Optional[str] = field(default=None)
    deepseek_key: Optional[str] = field(default=None)

    @classmethod
    def for_2x3090(cls) -> "LLMConfig":
        """Optimised config for 2×RTX 3090 (48GB total)."""
        return cls(
            provider=ModelProvider.VLLM,
            model="deepseek-ai/DeepSeek-Math-V2-70B-Instruct",
            tensor_parallel=2,
            max_tokens=512,
            temperature=0.05,
        )

    @classmethod
    def for_cmp50hx(cls) -> "LLMConfig":
        """Optimised config for CMP 50HX 10GB (all weights in VRAM)."""
        return cls(
            provider=ModelProvider.OLLAMA,
            ollama_model="mistral-nemo:latest",  # ~8GB Q4
            max_tokens=256,
            temperature=0.1,
        )

    @classmethod
    def for_claude_api(cls, api_key: str) -> "LLMConfig":
        """Cloud fallback via Anthropic API."""
        return cls(
            provider=ModelProvider.ANTHROPIC,
            model="claude-sonnet-4-6",
            anthropic_key=api_key,
            max_tokens=512,
            temperature=0.1,
        )


@dataclass
class AgentMessage:
    """Message passed between agents."""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    content: str
    structured: Optional[Dict[str, Any]] = None
    confidence: float = 0.0  # 0-1, model's self-reported confidence
    latency_ms: float = 0.0
    tokens_used: int = 0


class BaseAgent(ABC):
    """Base class for all trading agents."""

    def __init__(self, name: str, config: LLMConfig):
        self.name = name
        self.config = config
        self._client = self._init_client()

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.config.provider == ModelProvider.VLLM:
            return self._init_vllm()
        elif self.config.provider == ModelProvider.OLLAMA:
            return self._init_ollama()
        elif self.config.provider == ModelProvider.ANTHROPIC:
            return self._init_anthropic()
        elif self.config.provider == ModelProvider.OPENAI:
            return self._init_openai()
        elif self.config.provider == ModelProvider.DEEPSEEK:
            return self._init_deepseek()
        raise ValueError(f"Unknown provider: {self.config.provider}")

    def _init_vllm(self):
        """Initialize vLLM client for local GPU inference."""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.config.vllm_url,
                api_key="vllm",  # vLLM doesn't check
            )
            return client
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            return ollama.Client(host=self.config.ollama_url)
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.config.anthropic_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.config.openai_key)
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _init_deepseek(self):
        """Initialize DeepSeek API client."""
        try:
            from openai import OpenAI
            return OpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=self.config.deepseek_key or os.environ.get("DEEPSEEK_API_KEY"),
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent type."""
        pass

    @abstractmethod
    def parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse raw LLM output into structured data."""
        pass

    def generate(
        self,
        messages: List[AgentMessage],
        response_format: Optional[Dict] = None,
    ) -> AgentResponse:
        """Generate a response from the LLM."""
        import time
        start = time.perf_counter()

        system_prompt = self.get_system_prompt()
        full_messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": m.role, "content": m.content} for m in messages],
        ]

        if self.config.provider == ModelProvider.VLLM:
            response = self._vllm_generate(full_messages, response_format)
        elif self.config.provider == ModelProvider.OLLAMA:
            response = self._ollama_generate(full_messages, response_format)
        elif self.config.provider == ModelProvider.ANTHROPIC:
            response = self._anthropic_generate(full_messages, response_format)
        elif self.config.provider in (ModelProvider.OPENAI, ModelProvider.DEEPSEEK):
            response = self._openai_generate(full_messages, response_format)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        latency_ms = (time.perf_counter() - start) * 1000

        raw_content = response.get("content", "")
        structured = self.parse_response(raw_content)

        return AgentResponse(
            content=raw_content,
            structured=structured,
            confidence=response.get("confidence", 0.5),
            latency_ms=latency_ms,
            tokens_used=response.get("tokens_used", 0),
        )

    def _vllm_generate(self, messages, response_format=None):
        """Generate via vLLM (OpenAI-compatible API)."""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }

    def _ollama_generate(self, messages, response_format=None):
        """Generate via Ollama."""
        kwargs = {
            "model": self.config.ollama_model,
            "messages": messages,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = self._client.chat(**kwargs)
        return {
            "content": response["message"]["content"],
            "tokens_used": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        }

    def _anthropic_generate(self, messages, response_format=None):
        """Generate via Anthropic API."""
        system_msg = messages[0]["content"]
        user_messages = messages[1:]

        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system_msg,
            "messages": user_messages,
        }

        response = self._client.messages.create(**kwargs)
        return {
            "content": response.content[0].text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

    def _openai_generate(self, messages, response_format=None):
        """Generate via OpenAI or DeepSeek API."""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
