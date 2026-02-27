import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiAdapter:
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_response(self, messages, model="gemini-2.5-flash"):
        try:
            # Gemini API expects messages in a specific format
            # Convert generic messages to Gemini format
            gemini_messages = []
            system_instruction = None
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg["content"]]})

            config = {}
            if system_instruction:
                config["system_instruction"] = system_instruction

            model_instance = genai.GenerativeModel(model, config=config)
            response = model_instance.generate_content(gemini_messages)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {e}"
