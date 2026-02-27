"""
Generic adapter for any LLM provider that uses the Anthropic Messages API format.

Works with: Anthropic, AgentRouter, and any other provider that implements
the Anthropic Messages API.

Usage:
    # Direct Anthropic
    adapter = AnthropicCompatibleAdapter(api_key="sk-xxx")
    
    # AgentRouter (custom base_url + optional auth_token)
    adapter = AnthropicCompatibleAdapter(
        api_key="sk-xxx",
        base_url="https://agentrouter.org/",
        auth_token="sk-xxx",
    )
"""

import anthropic
from typing import List, Dict


class AnthropicCompatibleAdapter:
    def __init__(self, api_key: str = None, base_url: str = None, auth_token: str = None):
        """
        Args:
            api_key:    API key (sent as x-api-key header)
            base_url:   Custom API endpoint (e.g., "https://agentrouter.org/")
            auth_token: Bearer token auth (alternative to api_key, used by some providers)
        """
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if auth_token:
            kwargs["auth_token"] = auth_token
        self.client = anthropic.Anthropic(**kwargs)

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Converts internal message format to Anthropic-compatible format.
        
        Returns:
            (system_prompt, anthropic_messages) tuple
        
        - Extracts 'system' messages as the system prompt.
        - Converts 'tool_output' role → 'user' role with [Tool Output] prefix.
        - Merges consecutive same-role messages (Anthropic requires strict alternation).
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Anthropic supports a single system prompt; concatenate if multiple
                if system_prompt:
                    system_prompt += f"\n\n{content}"
                else:
                    system_prompt = content
                continue

            if role == "tool_output":
                role = "user"
                content = f"[Tool Output]:\n{content}"
            elif role not in ("user", "assistant"):
                role = "user"  # fallback for unknown roles

            # Merge consecutive same-role messages (Anthropic requires alternating turns)
            if anthropic_messages and anthropic_messages[-1]["role"] == role:
                anthropic_messages[-1]["content"] += f"\n{content}"
            else:
                anthropic_messages.append({"role": role, "content": content})

        return system_prompt, anthropic_messages

    def generate_response(self, messages: List[Dict[str, str]], model: str = "claude-sonnet-4-20250514") -> str:
        try:
            system_prompt, anthropic_messages = self._normalize_messages(messages)

            if not anthropic_messages:
                return "Error: No user messages found."

            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
            # Handle both standard Anthropic response format and custom API formats
            content = response.content[0]
            if isinstance(content, str):
                return content
            elif hasattr(content, 'text'):
                return content.text
            else:
                return str(content)
        except Exception as e:
            return f"Error communicating with Anthropic-compatible API: {e}"
