import sqlite3
from datetime import datetime

DB_NAME = "prediction_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            prediction TEXT,
            confidence REAL,
            model_used TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def insert_log(text, prediction, confidence, model_used):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_log (input_text, prediction, confidence, model_used)
        VALUES (?, ?, ?, ?)
    """, (text, prediction, confidence, model_used))

    conn.commit()
    conn.close()


def fetch_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM prediction_log ORDER BY timestamp DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows
