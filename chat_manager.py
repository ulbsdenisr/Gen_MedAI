import sqlite3
from datetime import datetime
import json
import os
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
            user_id INTEGER,
            role TEXT,
            message TEXT,
            timestamp TEXT
        )
        """)

        conn.commit()
        conn.close()

    def set_current_conversation(self, conversation_id):
        self.current_conversation_id = conversation_id

    # 🆕 Start a new chat
    def start_new_chat(self):
        self.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        return self.current_conversation_id

    # 🔹 Ensure we always have a conversation
    def ensure_conversation(self, user_id=None):
        if self.current_conversation_id is None and user_id is not None:
            self.start_new_chat()
        return self.current_conversation_id

    # 💾 Save message
    def save_message(self, role, message,user_id=None):
        self.ensure_conversation(user_id)

        conn = self._connect()
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
           INSERT INTO chat_history (conversation_id, role, message, timestamp, user_id)
           VALUES (?, ?, ?, ?, ?)
           """, (self.current_conversation_id, role, message, timestamp, user_id))

        conn.commit()
        conn.close()

    def get_chat_by_conversation(self, conversation_id, user_id):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT role, message, timestamp
        FROM chat_history
        WHERE conversation_id = ? AND user_id = ?
        ORDER BY id ASC
        """, (conversation_id, user_id))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "role": r[0],
                "message": r[1],
                "timestamp": r[2]
            }
            for r in rows
        ]

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

    def get_user_conversations(self, user_id):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT conversation_id, MIN(timestamp)
        FROM chat_history
        WHERE user_id = ?
        GROUP BY conversation_id
        ORDER BY MIN(timestamp) DESC
        """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        conversations = []

        for convo_id, timestamp in rows:
            conversations.append({
                "id": convo_id,
                "timestamp": int(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()),
                "preview": f"Chat {convo_id}"
            })

        return conversations

    # 📚 Get all conversations (useful later)///can be replaced by the get_user_conversations
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

        return [r[0] for r in rows] #



    def export_all_conversations_to_json(self, output_dir="chats"):
        os.makedirs(output_dir, exist_ok=True)

        conn = self._connect()
        cursor = conn.cursor()

        # Get all unique conversation IDs
        cursor.execute("""
        SELECT DISTINCT conversation_id
        FROM chat_history
        ORDER BY conversation_id ASC
        """)
        conversations = [row[0] for row in cursor.fetchall()]

        for convo_id in conversations:
            # Get messages for this conversation
            cursor.execute("""
            SELECT timestamp, role, message
            FROM chat_history
            WHERE conversation_id = ?
            ORDER BY id ASC
            """, (convo_id,))

            rows = cursor.fetchall()

            messages = []
            for row in rows:
                messages.append({
                    "timestamp": row[0],
                    "role": row[1],
                    "message": row[2]
                })

            # Structure final JSON
            convo_data = {
                "conversation_id": convo_id,
                "messages": messages
            }

            # Write to file
            file_path = os.path.join(output_dir, f"{convo_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(convo_data, f, indent=2, ensure_ascii=False)

        conn.close()
    def export_current_conversation(self, output_dir="chat_exports"):
        if self.current_conversation_id is None:
            return  # nothing to export

        os.makedirs(output_dir, exist_ok=True)

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT timestamp, role, message
        FROM chat_history
        WHERE conversation_id = ?
        ORDER BY id ASC
        """, (self.current_conversation_id,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return

        messages = []
        for row in rows:
            messages.append({
                "timestamp": row[0],
                "role": row[1],
                "message": row[2]
            })

        convo_data = {
            "conversation_id": self.current_conversation_id,
            "messages": messages
        }

        file_path = os.path.join(output_dir, f"{self.current_conversation_id}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(convo_data, f, indent=2, ensure_ascii=False)



