'''
This module provides a SafetyGuardrail class to prevent the execution of dangerous shell commands.
'''
import shlex

class SafetyGuardrail:
    def __init__(self):
        # A list of dangerous commands that should be blocked.
        self.blocklist = {
            "rm",
            "sudo",
            "mv",
            "chmod",
            "chown",
            "dd",
            "mkfs",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "passwd",
            "userdel",
            "groupdel",
            "usermod",
            "groupmod",
            "visudo",
            "crontab",
            "iptables",
            "ufw",
            "kill",
            "pkill",
            "killall",
            "wget", # Can be used to download malicious scripts
            "curl", # Can be used to download malicious scripts or exfiltrate data
        }

    def is_safe(self, command: str) -> tuple[bool, str]:
        '''
        Checks if a command is safe to execute.

        Args:
            command: The command to check.

        Returns:
            A tuple containing a boolean indicating if the command is safe and a message.
        '''
        try:
            # Use shlex to safely parse the command into a list of tokens.
            parts = shlex.split(command)
            if not parts:
                return True, "Command is empty."

            # The first part is the command itself.
            executable = parts[0]

            # Check if the command is in the blocklist.
            if executable in self.blocklist:
                return False, f"Command \'{executable}\' is in the blocklist and is not allowed."

            # Check for dangerous combinations, like `rm -rf`
            if executable == "rm" and "-rf" in parts:
                return False, "Command 'rm -rf' is explicitly blocked for safety."

            return True, "Command is safe."
        except Exception as e:
            return False, f"Error parsing command: {e}"
