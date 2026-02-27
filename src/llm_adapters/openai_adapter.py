import os
import openai
from dotenv import load_dotenv

load_dotenv()

class OpenAIAdapter:
    def __init__(self, api_key=None):
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(self, messages, model="gpt-4o-mini"):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenAI API: {e}"
