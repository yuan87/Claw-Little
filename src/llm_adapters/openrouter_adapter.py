import os
import openai
from dotenv import load_dotenv

load_dotenv()

class OpenRouterAdapter:
    def __init__(self, api_key=None):
        if api_key:
            self.client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        else:
            self.client = openai.OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

    def generate_response(self, messages, model="openai/gpt-4o-mini"):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenRouter API: {e}"
