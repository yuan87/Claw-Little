# OpenClaw Minimal Replication

This project is a minimal viable terminal UI replication of the OpenClaw project, focusing on core agentic loop functionality, multi-LLM backend support, and session management. It is designed to operate entirely within a terminal environment, with input exclusively through Bash or Windows CMD.

## Features

*   **Minimal Terminal UI**: Command-line interface for user interaction.
*   **Agentic Loop**: LLM can execute commands, observe output, and iterate on tasks.
*   **REPL-style Chat Interface**: Interactive conversations with the LLM.
*   **Multi-LLM Backend Support**: Integration with OpenAI, Anthropic, Google Gemini, and OpenRouter APIs.
*   **Session Management**: Persistence of conversation history and context.
*   **Technology Stack**: Python and Bash.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/openclaw_mini.git
    cd openclaw_mini
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    Create a `.env` file from the provided template and add your API keys.

## Usage

### Local Execution
Run the application using the provided shell script or directly via Python:
```bash
bash openclaw_mini.sh
# OR
python src/main.py
```

### GitHub Codespaces
1.  Open the repository on GitHub.
2.  Click the **Code** button, select the **Codespaces** tab, and click **Create codespace on main**.
3.  The environment will be automatically configured. Run `python src/main.py` in the terminal.

### Replit
1.  Import this repository into Replit.
2.  The `.replit` and `replit.nix` files will configure the environment.
3.  Click the **Run** button to start the REPL.


## Project Structure

```
openclaw_mini/
├── README.md
├── openclaw_mini_blueprint.md
└── src/
    ├── __init__.py
    ├── main.py
    ├── cli/
    │   └── __init__.py
    ├── llm_adapters/
    │   ├── __init__.py
    │   ├── openai_adapter.py
    │   ├── anthropic_adapter.py
    │   ├── gemini_adapter.py
    │   └── openrouter_adapter.py
    ├── session_manager/
    │   └── __init__.py
    │   └── session_manager.py
    ├── agentic_loop/
    │   └── __init__.py
    │   └── agentic_loop_executor.py
    └── tool_executor/
        └── __init__.py
        └── tool_executor.py
```
