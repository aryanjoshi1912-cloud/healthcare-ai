from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).with_name("medimind.db")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symptoms TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                urgency TEXT NOT NULL,
                description TEXT,
                precautions TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_diagnosis(symptoms: str, disease: str, confidence: float, urgency: str, description: str = "", precautions: str = "") -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO diagnoses (symptoms, disease, confidence, urgency, description, precautions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (symptoms, disease, confidence, urgency, description, precautions, datetime.now().isoformat(timespec="seconds")),
        )


def get_diagnoses(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM diagnoses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def save_chat(role: str, message: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_history (role, message, created_at) VALUES (?, ?, ?)",
            (role, message, datetime.now().isoformat(timespec="seconds")),
        )


def get_chat_history(limit: int = 40) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM chat_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return list(reversed([dict(row) for row in rows]))
