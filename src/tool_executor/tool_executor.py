import subprocess
import json
import os
import select
import time
from safety_guardrail.safety_guardrail import SafetyGuardrail

class PersistentShell:
    def __init__(self, workdir="./workspace"):
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self.process = subprocess.Popen(
            ["/bin/bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, # Line buffered
            cwd=self.workdir
        )
        self.delimiter = "---END_OF_COMMAND---"

    def execute(self, command: str, timeout: float = 30.0) -> str:
        full_command = f"{command}; echo \'{self.delimiter}\'\n"
        self.process.stdin.write(full_command)
        self.process.stdin.flush()

        output = ""
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                self.process.kill()
                return output + f"\n[Error: Command timed out after {timeout}s]"

            # Non-blocking read
            rlist, _, _ = select.select([self.process.stdout, self.process.stderr], [], [], 0.1)
            if self.process.stdout in rlist:
                line = self.process.stdout.readline()
                if self.delimiter in line:
                    output += line.replace(self.delimiter, "").strip()
                    break
                output += line
            if self.process.stderr in rlist:
                line = self.process.stderr.readline()
                if self.delimiter in line:
                    output += line.replace(self.delimiter, "").strip()
                    break
                output += line

            if self.process.poll() is not None: # Command finished
                # Read any remaining output
                stdout_remainder, stderr_remainder = self.process.communicate(timeout=0.1)
                output += stdout_remainder + stderr_remainder
                if self.delimiter in output:
                    output = output.replace(self.delimiter, "").strip()
                break

        return output.strip()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

class ToolExecutor:
    def __init__(self, safety_guardrail: SafetyGuardrail):
        self.shell = PersistentShell()
        self.safety_guardrail = safety_guardrail

    def execute_tool(self, tool_name: str, args: str) -> dict:
        if tool_name == "execute_bash":
            is_safe, message = self.safety_guardrail.is_safe(args)
            if not is_safe:
                return {"output": f"Guardrail blocked command: {message}", "returncode": 1}
            result = self.shell.execute(args)
            return {"output": result, "returncode": 0} # Assuming 0 for now, can parse later
        else:
            return {"output": f"Unknown tool: {tool_name}", "returncode": 1}

    def parse_tool_call(self, llm_response: str) -> dict | None:
        if "TOOL_CALL:" in llm_response:
            try:
                # Extract the JSON part using string manipulation
                tool_call_str = llm_response.split("TOOL_CALL:", 1)[1].strip()
                # Find the first { and the last } in the remaining string
                start_idx = tool_call_str.find('{')
                end_idx = tool_call_str.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = tool_call_str[start_idx:end_idx+1]
                    # strict=False allows unescaped control characters like literal newlines inside string values
                    tool_call = json.loads(json_str.replace("\\'", "\""), strict=False)
                    return tool_call
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                return None
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                return None
        return None

    def __del__(self):
        if hasattr(self, 'shell'):
            self.shell.close()
