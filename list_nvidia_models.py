import os
import openai
from dotenv import load_dotenv

load_dotenv()

def filter_nvidia_models():
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = "https://integrate.api.nvidia.com/v1"

    if not api_key:
        print("ERROR: NVIDIA_API_KEY not found in .env")
        return

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        models = client.models.list()
        print("Relevant Models on NVIDIA NIM:")
        for model in models.data:
            m_id = model.id.lower()
            if "minimax" in m_id or "llama-3.1" in m_id or "llama-3.3" in m_id or "nemotron-70b" in m_id:
                print(f" - {model.id}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    filter_nvidia_models()
