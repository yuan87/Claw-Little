import os
import openai
from dotenv import load_dotenv

load_dotenv()

def test_nvidia_nim():
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = "https://integrate.api.nvidia.com/v1"
    model = os.getenv("DEFAULT_LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

    print(f"Testing NVIDIA NIM integration...")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Key Prefix: {api_key[:10]}..." if api_key else "No Key Found")

    if not api_key:
        print("ERROR: NVIDIA_API_KEY not found in .env")
        return

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello! Just testing the connection. Reply with 'OK' if you see this."}],
            max_tokens=10
        )
        print("\nSuccess! Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_nvidia_nim()
