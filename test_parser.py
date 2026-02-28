from src.tool_executor.tool_executor import ToolExecutor
from src.safety_guardrail.safety_guardrail import SafetyGuardrail

if __name__ == "__main__":
    executor = ToolExecutor(SafetyGuardrail())
    test_str = """
<think>The user wants me to create a bubble sort algorithm animation in a single HTML file using HTML and JavaScript. I'll create a visually appealing animation that shows how bubble sort works step by step.
</think>

I'll create a beautiful bubble sort animation for you.
TOOL_CALL: {"tool_name": "execute_bash", "args": "cat > foo.html << 'EOF'\n<html>\nhello\n</html>\nEOF\necho done!"}

More words.
"""
    result = executor.parse_tool_call(test_str)
    print("Parsed result:", result)
