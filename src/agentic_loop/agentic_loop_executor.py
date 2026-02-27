import json
from typing import List, Dict, Any

class AgenticLoopExecutor:
    def __init__(self, llm_adapter, tool_executor):
        self.llm_adapter = llm_adapter
        self.tool_executor = tool_executor
        self.system_prompt = """
You are an AI assistant that can interact with the user and execute bash commands. 
When you need to execute a command, respond with a JSON object in the format: 
TOOL_CALL: {\"tool_name\": \"execute_bash\", \"args\": \"your command here\"}
For example: TOOL_CALL: {\"tool_name\": \"execute_bash\", \"args\": \"ls -l\"}

Use the `execute_bash` tool to:
1. Read files (e.g., `cat <filename>`, `head <filename>`)
2. Search for content (e.g., `grep <pattern> <filename>`)
3. Create or modify files (e.g., `echo \"content\" > <filename>`, `sed -i \"s/old/new/g\" <filename>`)
4. Run scripts or programs (e.g., `python3 <script.py>`, `node <script.js>`)
5. Navigate the file system (e.g., `cd <directory>`, `ls -F`)

**Safety Guardrails:**
Be aware that certain dangerous commands are blocked for your safety and the integrity of the system. If you attempt to execute a blocked command, you will receive a 'Guardrail blocked command' message. In such cases, you should re-evaluate your approach and try a safer alternative.

After executing a command, the output will be provided to you. 
If you do not need to execute a command, respond with a regular message.
"""

    def run_agentic_loop(self, messages: List[Dict[str, str]], model: str = None) -> str:
        # Add the system prompt to the beginning of the messages if it's not already there
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Pass model to the adapter (adapters handle None gracefully with their own default)
        kwargs = {"messages": messages}
        if model:
            kwargs["model"] = model
        llm_response = self.llm_adapter.generate_response(**kwargs)

        tool_call = self.tool_executor.parse_tool_call(llm_response)

        if tool_call:
            tool_name = tool_call.get("tool_name")
            args = tool_call.get("args")

            if tool_name == "execute_bash" and args:
                print(f"Executing bash command: {args}")
                tool_output = self.tool_executor.execute_tool(tool_name, args)
                messages.append({"role": "assistant", "content": llm_response}) # Store the tool call from LLM
                messages.append({"role": "tool_output", "content": json.dumps(tool_output)})
                # Recursively call the agentic loop with the tool output
                return self.run_agentic_loop(messages, model=model)
            else:
                return f"Error: Unknown tool or missing arguments: {tool_call}"
        else:
            return llm_response
