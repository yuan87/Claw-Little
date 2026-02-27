"""
Generic adapter for any LLM provider that uses the OpenAI-compatible Chat Completions API.

Works with: OpenAI, DeepSeek, OpenRouter, AnyRouter, Gemini (via OpenAI-compat endpoint),
and any other provider that implements the OpenAI Chat Completions API.

Usage:
    # Direct OpenAI
    adapter = OpenAICompatibleAdapter(api_key="sk-xxx")
    
    # DeepSeek (custom base_url)
    adapter = OpenAICompatibleAdapter(api_key="sk-xxx", base_url="https://api.deepseek.com")
    
    # OpenRouter
    adapter = OpenAICompatibleAdapter(api_key="sk-xxx", base_url="https://openrouter.ai/api/v1")
"""

import openai
from typing import List, Dict


class OpenAICompatibleAdapter:
    def __init__(self, api_key: str = None, base_url: str = None):
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Converts internal message format to OpenAI-compatible format.
        - 'tool_output' role → 'user' role with [Tool Output] prefix
        - Merges consecutive same-role messages to avoid API errors.
        """
        normalized = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "tool_output":
                role = "user"
                content = f"[Tool Output]:\n{content}"
            elif role not in ("system", "user", "assistant"):
                role = "user"  # fallback for unknown roles

            # Merge consecutive same-role messages
            if normalized and normalized[-1]["role"] == role and role != "system":
                normalized[-1]["content"] += f"\n{content}"
            else:
                normalized.append({"role": role, "content": content})

        return normalized

    def generate_response(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
        try:
            normalized = self._normalize_messages(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=normalized,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenAI-compatible API: {e}"
