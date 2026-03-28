# /------------------------------\
# |   START OF database.py FILE   |
# \------------------------------/

import os
import sqlite3
import time
import random
from datetime import datetime, timedelta
from collections import OrderedDict

# Allow the database path to be overridden via environment variable.
# In production (Docker), set DB_PATH=/data/synthnotes.db and mount a
# persistent volume at /data so the database survives restarts/redeploys.
# Falls back to the working directory for local development.
DB_FILE = os.environ.get("DB_PATH", "synthnotes.db")

DEFAULT_SECTORS = {
    "IT Services": """Future investments related comments (Including GenAI, AI, Data, Cloud, etc):
Capital allocation:
Talent supply chain related comments:
Org structure change:
Other comments:
Short-term comments:
- Guidance:
- Order booking:
- Impact of macro slowdown:
- Vertical wise comments:""",
    "QSR": """Customer proposition:
Menu strategy (Includes: new product launches, etc):
Operational update (Includes: SSSG, SSTG, Price hike, etc):
Unit economics:
Store opening:"""
}

def _populate_default_sectors(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sectors")
    if cursor.fetchone()[0] == 0:
        for name, topics in DEFAULT_SECTORS.items():
            cursor.execute("INSERT OR REPLACE INTO sectors (name, topics) VALUES (?, ?)", (name, topics))
        conn.commit()

def init_db():
    with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        # Note table with pdf_blob support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, meeting_type TEXT,
                file_name TEXT, content TEXT, raw_transcript TEXT,
                refined_transcript TEXT, token_usage INTEGER, processing_time REAL,
                pdf_blob BLOB
            )
        """)
        # Sectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sectors ( name TEXT PRIMARY KEY, topics TEXT NOT NULL )
        """)
        # Entities table for Knowledge Base features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id TEXT,
                entity TEXT,
                type TEXT,
                sentiment TEXT,
                context TEXT,
                FOREIGN KEY (note_id) REFERENCES notes (id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        _populate_default_sectors(conn)

def safe_db_operation(operation_func, *args, **kwargs):
    """Handles 'database is locked' errors with randomized exponential backoff."""
    max_retries = 3
    base_delay = 0.1
    for attempt in range(max_retries):
        try:
            with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
                return operation_func(conn, *args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            else:
                raise ConnectionError(f"Database operation failed after retries: {e}")
    raise ConnectionError("Database remained locked after multiple retries.")


# --- Note Management ---

def _save_note_op(conn, note_data: dict):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO notes (id, created_at, meeting_type, file_name, content, raw_transcript, refined_transcript, token_usage, processing_time, pdf_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        note_data.get('id'), note_data.get('created_at'), note_data.get('meeting_type'),
        note_data.get('file_name'), note_data.get('content'), note_data.get('raw_transcript'),
        note_data.get('refined_transcript'), note_data.get('token_usage'), note_data.get('processing_time'),
        note_data.get('pdf_blob')
    ))
    conn.commit()

def save_note(note_data: dict):
    safe_db_operation(_save_note_op, note_data)

def _get_all_notes_op(conn):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Returns lightweight list for the sidebar (excludes heavy transcripts)
    query = "SELECT id, created_at, meeting_type, file_name, content FROM notes ORDER BY created_at DESC"
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]

def get_all_notes():
    return safe_db_operation(_get_all_notes_op)

def _get_note_by_id_op(conn, note_id: str):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Returns EVERYTHING for a specific note (includes transcripts)
    cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
    row = cursor.fetchone()
    return dict(row) if row else None

def get_note_by_id(note_id: str):
    return safe_db_operation(_get_note_by_id_op, note_id)

def _delete_note_op(conn, note_id: str):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM entities WHERE note_id = ?", (note_id,))
    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()

def delete_note(note_id: str):
    safe_db_operation(_delete_note_op, note_id)


# --- Analytics ---

def _get_analytics_summary_op(conn):
    conn.row_factory = sqlite3.Row
    try:
        notes = conn.execute('SELECT processing_time, token_usage, created_at FROM notes').fetchall()
    except sqlite3.OperationalError:
        # Fallback if table doesn't exist yet
        return {'total_notes': 0, 'avg_time': 0, 'total_tokens': 0}, {}
    
    summary = {
        'total_notes': len(notes),
        'avg_time': sum(n['processing_time'] for n in notes if n['processing_time']) / len(notes) if notes else 0,
        'total_tokens': sum(n['token_usage'] for n in notes if n['token_usage'])
    }
    
    daily_counts = OrderedDict()
    today = datetime.now().date()
    for i in range(13, -1, -1):
        day = today - timedelta(days=i)
        daily_counts[day.strftime('%Y-%m-%d')] = 0

    for note in notes:
        try:
            note_date_str = datetime.fromisoformat(note['created_at']).strftime('%Y-%m-%d')
            if note_date_str in daily_counts:
                daily_counts[note_date_str] += 1
        except (ValueError, TypeError):
            continue
            
    return summary, daily_counts

def get_analytics_summary():
    return safe_db_operation(_get_analytics_summary_op)


# --- Sector Management ---

def _get_sectors_op(conn):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name, topics FROM sectors ORDER BY name")
    return {row['name']: row['topics'] for row in cursor.fetchall()}

def get_sectors():
    return safe_db_operation(_get_sectors_op)

def _save_sector_op(conn, name, topics):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO sectors (name, topics) VALUES (?, ?)", (name, topics))
    conn.commit()

def save_sector(name, topics):
    safe_db_operation(_save_sector_op, name, topics)

def _delete_sector_op(conn, name):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sectors WHERE name = ?", (name,))
    conn.commit()

def delete_sector(name):
    safe_db_operation(_delete_sector_op, name)


# --- Entity Management ---

def _save_entities_op(conn, note_id: str, entities: list):
    cursor = conn.cursor()
    for entity in entities:
        cursor.execute(
            'INSERT INTO entities (note_id, entity, type, sentiment, context) VALUES (?, ?, ?, ?, ?)',
            (note_id, entity.get('entity'), entity.get('type'), entity.get('sentiment'), entity.get('context'))
        )
    conn.commit()

def save_entities(note_id: str, entities: list):
    safe_db_operation(_save_entities_op, note_id, entities)

def _get_entities_for_note_op(conn, note_id: str):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM entities WHERE note_id = ? ORDER BY entity", (note_id,))
    return [dict(row) for row in cursor.fetchall()]

def get_entities_for_note(note_id: str):
    return safe_db_operation(_get_entities_for_note_op, note_id)

def _update_entities_for_note_op(conn, note_id: str, entities_data: list):
    cursor = conn.cursor()
    cursor.execute('DELETE FROM entities WHERE note_id = ?', (note_id,))
    for entity in entities_data:
        if entity.get('entity') and entity.get('type'):
            cursor.execute(
                'INSERT INTO entities (note_id, entity, type, sentiment, context) VALUES (?, ?, ?, ?, ?)',
                (note_id, entity.get('entity'), entity.get('type'), entity.get('sentiment'), entity.get('context'))
            )
    conn.commit()

def update_entities_for_note(note_id: str, entities_data: list):
    safe_db_operation(_update_entities_for_note_op, note_id, entities_data)

def _search_notes_by_entity_op(conn, entity_text: str, exclude_note_id: str):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT DISTINCT note_id FROM entities WHERE entity = ? AND note_id != ?',
        (entity_text, exclude_note_id)
    )
    note_ids_rows = cursor.fetchall()
    
    if not note_ids_rows:
        return []

    id_list = [row['note_id'] for row in note_ids_rows]
    placeholders = ','.join('?' for _ in id_list)
    query = f'SELECT * FROM notes WHERE id IN ({placeholders}) ORDER BY created_at DESC'
    cursor.execute(query, id_list)
    return [dict(row) for row in cursor.fetchall()]

def search_notes_by_entity(entity_text: str, exclude_note_id: str):
    return safe_db_operation(_search_notes_by_entity_op, entity_text, exclude_note_id)

# /----------------------------\
# |   END OF database.py FILE   |
# \----------------------------/
