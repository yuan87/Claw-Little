import os
import json
from typing import List, Dict, Any

from llm_adapters.llm_factory import (
    get_llm_adapter,
    get_default_model,
    get_api_format,
    list_providers,
)
from tool_executor.tool_executor import ToolExecutor
from agentic_loop.agentic_loop_executor import AgenticLoopExecutor
from session_manager.session_manager import SessionManager
from safety_guardrail.safety_guardrail import SafetyGuardrail

class Orchestrator:
    def __init__(self):
        self.session_manager = SessionManager()
        self.safety_guardrail = SafetyGuardrail()
        self.tool_executor = ToolExecutor(self.safety_guardrail) # Pass guardrail to tool executor
        
        # Initial LLM setup
        self.current_llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.current_llm_model = os.getenv(
            "DEFAULT_LLM_MODEL",
            get_default_model(self.current_llm_provider),
        )
        self.llm_adapter = get_llm_adapter(self.current_llm_provider)
        self.agentic_loop_executor = AgenticLoopExecutor(self.llm_adapter, self.tool_executor)

        self._initialize_session()

    def _initialize_session(self):
        if os.path.exists(self.session_manager.session_dir):
            sessions = self.session_manager.list_sessions()
            if sessions:
                print("Existing sessions found. Loading the most recent one.")
                self.session_manager.load_session(sessions[-1])
            else:
                print("No existing sessions. Creating a new one.")
                self.session_manager.create_new_session()
        else:
            print("No existing sessions directory. Creating a new session.")
            self.session_manager.create_new_session()

    def print_help(self):
        providers = ", ".join(list_providers())
        print("\nAvailable commands:")
        print(f"  /llm <provider> [model] - Change LLM provider and optionally model")
        print(f"                            Providers: {providers}")
        print("  /providers              - List all supported providers with their defaults")
        print("  /session new [id]       - Create a new session (optionally with an ID)")
        print("  /session load <id>      - Load an existing session")
        print("  /session list           - List all available sessions")
        print("  /session current        - Show current session ID")
        print("  /exit                   - Exit the application")
        print("  /help                   - Show this help message")
        print("\nType your message to the LLM or a command.")

    def _print_providers(self):
        """Print a table of all registered providers."""
        from llm_adapters.llm_factory import PROVIDERS
        print("\n  Registered LLM Providers:")
        print("  " + "-" * 72)
        print(f"  {'Provider':<15} {'API Format':<12} {'Default Model':<30} {'Base URL'}")
        print("  " + "-" * 72)
        for name in sorted(PROVIDERS.keys()):
            cfg = PROVIDERS[name]
            fmt = cfg["api_format"]
            model = cfg["default_model"]
            url = cfg.get("base_url") or "(SDK default)"
            marker = " ◀" if name == self.current_llm_provider else ""
            print(f"  {name:<15} {fmt:<12} {model:<30} {url}{marker}")
        print("  " + "-" * 72)

    def run(self):
        print("Welcome to OpenClaw Mini! Type /help for commands.")
        print(f"Current LLM: {self.current_llm_provider} ({self.current_llm_model})")
        print(f"Active session: {self.session_manager.get_current_session_id()}")

        while True:
            try:
                user_input = input(f"\n[{self.session_manager.get_current_session_id()}/{self.current_llm_provider}] You: ")
                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    parts = user_input[1:].split(maxsplit=2)
                    command = parts[0]
                    args = parts[1:]

                    if command == "exit":
                        self.session_manager.save_session()
                        self.tool_executor.shell.close()
                        print("Goodbye!")
                        break
                    elif command == "help":
                        self.print_help()
                    elif command == "providers":
                        self._print_providers()
                    elif command == "llm":
                        if len(args) >= 1:
                            new_provider = args[0].lower()
                            new_model = args[1] if len(args) >= 2 else None
                            try:
                                self.llm_adapter = get_llm_adapter(new_provider)
                                self.current_llm_provider = new_provider
                                # Use specified model, or fall back to the provider's default
                                self.current_llm_model = new_model or get_default_model(new_provider)
                                self.agentic_loop_executor.llm_adapter = self.llm_adapter
                                print(f"LLM changed to: {self.current_llm_provider} ({self.current_llm_model})")
                            except ValueError as e:
                                print(f"Error: {e}")
                        else:
                            print(f"Current LLM: {self.current_llm_provider} ({self.current_llm_model})")
                            print("Usage: /llm <provider> [model]")
                    elif command == "session":
                        if len(args) >= 1:
                            subcommand = args[0]
                            if subcommand == "new":
                                new_session_id = args[1] if len(args) >= 2 else None
                                try:
                                    self.session_manager.create_new_session(new_session_id)
                                    print(f"New session '{self.session_manager.get_current_session_id()}' created.")
                                except ValueError as e:
                                    print(f"Error: {e}")
                            elif subcommand == "load":
                                if len(args) >= 2:
                                    if self.session_manager.load_session(args[1]):
                                        print(f"Session '{self.session_manager.get_current_session_id()}' loaded.")
                                    else:
                                        print(f"Failed to load session '{args[1]}'")
                                else:
                                    print("Usage: /session load <id>")
                            elif subcommand == "list":
                                sessions = self.session_manager.list_sessions()
                                if sessions:
                                    print("Available sessions:")
                                    for s_id in sessions:
                                        print(f"  - {s_id}")
                                else:
                                    print("No sessions found.")
                            elif subcommand == "current":
                                print(f"Current session: {self.session_manager.get_current_session_id()}")
                            else:
                                print("Unknown session subcommand. Usage: /session [new|load|list|current]")
                        else:
                            print("Usage: /session [new|load|list|current]")
                    else:
                        print(f"Unknown command: {user_input}")
                else:
                    self.session_manager.add_message("user", user_input)
                    messages = self.session_manager.get_history()

                    # Adapters now handle tool_output role conversion internally,
                    # so we just pass the full history directly.
                    response = self.agentic_loop_executor.run_agentic_loop(
                        messages, model=self.current_llm_model
                    )

                    print(f"\n[{self.session_manager.get_current_session_id()}/{self.current_llm_provider}] LLM: {response}")
                    self.session_manager.add_message("assistant", response)

            except KeyboardInterrupt:
                self.session_manager.save_session()
                self.tool_executor.shell.close()
                print("\nExiting OpenClaw Mini. Goodbye!")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
