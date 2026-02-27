import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

class AnthropicAdapter:
    def __init__(self, api_key=None):
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_response(self, messages, model="claude-3-opus-20240229"):
        try:
            # Anthropic API expects messages in a specific format
            # Convert generic messages to Anthropic format
            anthropic_messages = []
            system_prompt = None
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    anthropic_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    anthropic_messages.append({"role": "assistant", "content": msg["content"]})
            
            if not anthropic_messages:
                return "Error: No user messages found."

            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=anthropic_messages
            )
            return response.content[0].text
        except Exception as e:
            return f"Error communicating with Anthropic API: {e}"
