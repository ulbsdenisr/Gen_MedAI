import sqlite3
from datetime import datetime

class ChatManager:
    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self.current_conversation_id = None
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            message TEXT,
            timestamp TEXT
        )
        """)

        conn.commit()
        conn.close()

    # 🆕 Start a new chat
    def start_new_chat(self):
        self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        return self.current_conversation_id

    # 🔹 Ensure we always have a conversation
    def ensure_conversation(self):
        if self.current_conversation_id is None:
            self.start_new_chat()
        return self.current_conversation_id

    # 💾 Save message
    def save_message(self, role, message):
        self.ensure_conversation()

        conn = self._connect()
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
        INSERT INTO chat_history (conversation_id, role, message, timestamp)
        VALUES (?, ?, ?, ?)
        """, (self.current_conversation_id, role, message, timestamp))

        conn.commit()
        conn.close()

    # 📖 Get current chat
    def get_current_chat(self):
        if self.current_conversation_id is None:
            return []

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT role, message, timestamp
        FROM chat_history
        WHERE conversation_id = ?
        ORDER BY id ASC
        """, (self.current_conversation_id,))

        rows = cursor.fetchall()
        conn.close()

        return rows

    # 📚 Get all conversations (useful later)
    def get_all_conversations(self):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT DISTINCT conversation_id
        FROM chat_history
        ORDER BY conversation_id DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [r[0] for r in rows]