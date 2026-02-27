import json
import os
from datetime import datetime
from typing import List, Dict, Any

class SessionManager:
    def __init__(self, session_dir="./sessions"):
        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)
        self.current_session_id = None
        self.history: List[Dict[str, str]] = []

    def _get_session_file_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir, f"{session_id}.json")

    def create_new_session(self, session_id: str = None) -> str:
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_path = self._get_session_file_path(session_id)
        if os.path.exists(file_path):
            raise ValueError(f"Session with ID \'{session_id}\' already exists.")

        self.current_session_id = session_id
        self.history = []
        self.save_session()
        return session_id

    def load_session(self, session_id: str) -> bool:
        file_path = self._get_session_file_path(session_id)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                session_data = json.load(f)
            self.current_session_id = session_id
            self.history = session_data.get("history", [])
            print(f"Session \'{session_id}\' loaded successfully.")
            return True
        else:
            print(f"Session \'{session_id}\' not found.")
            return False

    def save_session(self):
        if self.current_session_id:
            file_path = self._get_session_file_path(self.current_session_id)
            session_data = {
                "session_id": self.current_session_id,
                "history": self.history,
                "last_saved": datetime.now().isoformat()
            }
            with open(file_path, "w") as f:
                json.dump(session_data, f, indent=4)
            # print(f"Session \'{self.current_session_id}\' saved.")
        else:
            print("No active session to save.")

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.save_session()

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def list_sessions(self) -> List[str]:
        sessions = []
        for filename in os.listdir(self.session_dir):
            if filename.endswith(".json"):
                sessions.append(filename.replace(".json", ""))
        return sorted(sessions)

    def get_current_session_id(self):
        return self.current_session_id
