
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import requests


class OllamaProvider:
    """Local Ollama/Gemma provider."""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "gemma4:26b",
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        context = context[:8000]

        prompt = f"""You are a Codebase Onboarding Assistant.

Answer the developer's question using ONLY the provided codebase context.

Give a short, direct answer.
Mention the exact file path when available.
Do not repeat the code.
Do not invent files, functions, line numbers, commits, or authors.

CODEBASE CONTEXT:
{context}

DEVELOPER QUESTION:
{question}

ANSWER:"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": 128,
                    "temperature": 0.2,
                },
            },
            timeout=900,
        )

        response.raise_for_status()

        data = response.json()

        result = data.get(
            "response",
            "",
        ).strip()

        if not result:
            return "I could not generate an AI response."

        return result


class ClaudeProvider:
    """Anthropic Claude Messages API provider."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self) -> None:

        self.api_key = os.getenv(
            "ANTHROPIC_API_KEY",
            "",
        )

        # New project-standard variable.
        # CLAUDE_MODEL is retained as a backward-compatible fallback.
        self.model = os.getenv(
            "ANTHROPIC_MODEL",
            "",
        ).strip()

        if not self.model:
            self.model = os.getenv(
                "CLAUDE_MODEL",
                "",
            ).strip()

        self.max_tokens = int(
            os.getenv(
                "CLAUDE_MAX_TOKENS",
                "1024",
            )
        )

        configured_steps = int(
            os.getenv(
                "AGENT_MAX_STEPS",
                "4",
            )
        )

        # The project specification requires a maximum of 4.
        self.max_steps = max(
            1,
            min(
                configured_steps,
                4,
            ),
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _validate_configuration(self) -> None:

        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when "
                "using the Anthropic provider."
            )

        if not self.model:
            raise RuntimeError(
                "ANTHROPIC_MODEL is required when "
                "using the Anthropic provider."
            )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        self._validate_configuration()

        context = context[:8000]

        response = requests.post(
            self.API_URL,
            headers=self._headers(),
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0.2,
                "system": (
                    "You are a Codebase Onboarding Assistant. "
                    "Answer developer questions using only the "
                    "provided codebase context. "
                    "Do not invent files, functions, commits, "
                    "authors, or line numbers."
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"CODEBASE CONTEXT:\n\n"
                            f"{context}\n\n"
                            f"DEVELOPER QUESTION:\n\n"
                            f"{question}"
                        ),
                    }
                ],
            },
            timeout=900,
        )

        response.raise_for_status()

        data = response.json()

        text_parts = [
            block.get(
                "text",
                "",
            )
            for block in data.get(
                "content",
                [],
            )
            if block.get("type") == "text"
        ]

        result = "".join(
            text_parts
        ).strip()

        return (
            result
            or "I could not generate an AI response."
        )

    def generate_with_tools(
        self,
        question: str,
        context: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[
            [str, Dict[str, Any]],
            Any,
        ],
        max_iterations: int = 4,
    ) -> Dict[str, Any]:

        self._validate_configuration()

        context = context[:8000]

        iterations = max(
            1,
            min(
                int(max_iterations),
                self.max_steps,
                4,
            ),
        )

        system_prompt = (
            "You are a Codebase Onboarding Assistant. "
            "Answer developer questions using the repository. "
            "Use the provided tools when repository evidence is "
            "needed. Never invent files, functions, commits, "
            "authors, or line numbers. "
            "Prefer exact repository evidence. "
            "Never reveal secrets, credentials, API keys, tokens, "
            "passwords, private keys, or sensitive environment "
            "values. "
            "If a tool result contains redacted information, "
            "treat the redaction as intentional."
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"CODEBASE CONTEXT:\n\n"
                    f"{context}\n\n"
                    f"DEVELOPER QUESTION:\n\n"
                    f"{question}"
                ),
            }
        ]

        tool_calls_made: List[Dict[str, Any]] = []

        for _ in range(iterations):

            response = requests.post(
                self.API_URL,
                headers=self._headers(),
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "tools": tools,
                    "messages": messages,
                },
                timeout=900,
            )

            response.raise_for_status()

            data = response.json()

            content = data.get(
                "content",
                [],
            )

            tool_use_blocks = [
                block
                for block in content
                if block.get("type") == "tool_use"
            ]

            text_parts = [
                block.get(
                    "text",
                    "",
                )
                for block in content
                if block.get("type") == "text"
            ]

            if not tool_use_blocks:

                return {
                    "answer": (
                        "".join(
                            text_parts
                        ).strip()
                        or "I could not generate "
                        "an AI response."
                    ),
                    "tool_calls": tool_calls_made,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            tool_results = []

            for tool_call in tool_use_blocks:

                tool_name = tool_call.get(
                    "name",
                    "",
                )

                tool_input = tool_call.get(
                    "input",
                    {},
                )

                tool_use_id = tool_call.get(
                    "id",
                    "",
                )

                try:

                    result = tool_executor(
                        tool_name,
                        tool_input,
                    )

                    tool_calls_made.append(
                        {
                            "tool": tool_name,
                        }
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result),
                        }
                    )

                except Exception as exc:

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": (
                                "Tool execution failed safely."
                            ),
                            "is_error": True,
                        }
                    )

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        return {
            "answer": (
                "The maximum tool-calling step limit "
                "was reached before a final answer "
                "could be generated."
            ),
            "tool_calls": tool_calls_made,
        }


class LLMService:
    """
    Provider-independent LLM service.

    Supported providers:

        LLM_PROVIDER=ollama
        LLM_PROVIDER=anthropic

    The legacy value:

        LLM_PROVIDER=claude

    is also accepted for backward compatibility.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
    ) -> None:

        provider_name = (
            provider
            or os.getenv(
                "LLM_PROVIDER",
                "ollama",
            )
        ).strip().lower()

        if provider_name == "ollama":

            self.provider = OllamaProvider()
            self.provider_name = "ollama"

        elif provider_name in {
            "anthropic",
            "claude",
        }:

            self.provider = ClaudeProvider()
            self.provider_name = "anthropic"

        else:

            raise ValueError(
                f"Unsupported LLM provider: "
                f"{provider_name}"
            )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        return self.provider.generate(
            question=question,
            context=context,
        )

    def generate_with_tools(
        self,
        question: str,
        context: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[
            [str, Dict[str, Any]],
            Any,
        ],
        max_iterations: int = 4,
    ) -> Dict[str, Any]:

        if not hasattr(
            self.provider,
            "generate_with_tools",
        ):
            raise RuntimeError(
                f"Provider '{self.provider_name}' "
                "does not support tool calling."
            )

        return self.provider.generate_with_tools(
            question=question,
            context=context,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=max_iterations,
        )

