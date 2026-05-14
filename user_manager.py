import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

class UserManager:
    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self.current_user_id = None
        self._create_user_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_user_table(self):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
        """)

        conn.commit()
        conn.close()
    def create_user(self, username, password):
        conn = self._connect()
        cursor = conn.cursor()

        password_hash = generate_password_hash(password)

        try:
            cursor.execute("""
            INSERT INTO users (username, password_hash)
            VALUES (?, ?)
            """, (username, password_hash))

            conn.commit()
            user_id = cursor.lastrowid

            self.current_user_id = user_id
            return user_id

        except sqlite3.IntegrityError:
            return None  # username already exists

        finally:
            conn.close()
    def login_user(self, username, password):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT id, password_hash FROM users WHERE username = ?
        """, (username,))

        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row[1], password):
            self.current_user_id = row[0]
            return row[0]

        return None
    def logout_user(self):
        self.current_user_id = None
    def get_current_user(self):
        return self.current_user_id