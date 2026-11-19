"""SQLite registry: which documents exist, their versions, status, and chunk text.

Shared by the back office (writes) and the chat API (reads). Every change bumps
`kb_version`, which tells the chat API to refresh its in-memory search structures.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  doc_type TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  version INTEGER NOT NULL,
  status TEXT NOT NULL,
  chunk_count INTEGER DEFAULT 0,
  size_bytes INTEGER DEFAULT 0,
  content_hash TEXT,
  note TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks(
  doc_id INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  text TEXT NOT NULL,
  meta TEXT NOT NULL,
  PRIMARY KEY(doc_id, idx)
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Registry:
    def __init__(self, db_path="data/kb.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- documents ----------
    def add_document(self, name, doc_type, source_kind, source_ref, version,
                     chunk_count, size_bytes, content_hash, note=""):
        now = _now()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO documents(name,doc_type,source_kind,source_ref,version,status,"
                "chunk_count,size_bytes,content_hash,note,created_at,updated_at) "
                "VALUES(?,?,?,?,?,'active',?,?,?,?,?,?)",
                (name, doc_type, source_kind, str(source_ref), version, chunk_count,
                 size_bytes, content_hash, note, now, now))
            return cur.lastrowid

    def get(self, doc_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(r) if r else None

    def get_active(self, name):
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE name=? AND status='active' "
                          "ORDER BY version DESC LIMIT 1", (name,)).fetchone()
            return dict(r) if r else None

    def latest_version(self, name):
        with self._conn() as c:
            r = c.execute("SELECT MAX(version) AS v FROM documents WHERE name=?", (name,)).fetchone()
            return r["v"] or 0

    def versions_of(self, name):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM documents WHERE name=? ORDER BY version DESC", (name,))]

    def list(self, doc_types=None, status=None, search=None):
        sql, args = "SELECT * FROM documents WHERE 1=1", []
        if doc_types:
            sql += f" AND doc_type IN ({','.join('?' * len(doc_types))})"
            args += list(doc_types)
        if status:
            sql += " AND status=?"
            args.append(status)
        if search:
            sql += " AND name LIKE ?"
            args.append(f"%{search}%")
        sql += " ORDER BY (status='active') DESC, created_at DESC, id DESC"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def set_status(self, doc_id, status):
        with self._conn() as c:
            c.execute("UPDATE documents SET status=?, updated_at=? WHERE id=?",
                      (status, _now(), doc_id))

    def delete_document(self, doc_id):
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def clear_all(self):
        with self._conn() as c:
            c.execute("DELETE FROM chunks")
            c.execute("DELETE FROM documents")

    def stats(self):
        with self._conn() as c:
            r = c.execute(
                "SELECT SUM(status='active') AS active_docs, SUM(status!='active') AS archived, "
                "SUM(CASE WHEN status='active' THEN chunk_count ELSE 0 END) AS active_chunks "
                "FROM documents").fetchone()
            return {"active_docs": r["active_docs"] or 0, "archived": r["archived"] or 0,
                    "active_chunks": r["active_chunks"] or 0}

    # ---------- chunks ----------
    def save_chunks(self, doc_id, rows):
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.executemany("INSERT INTO chunks(doc_id,idx,text,meta) VALUES(?,?,?,?)",
                          [(doc_id, i, t, json.dumps(m)) for i, (t, m) in enumerate(rows)])

    def get_chunks(self, doc_id):
        with self._conn() as c:
            return [(r["text"], json.loads(r["meta"])) for r in c.execute(
                "SELECT text, meta FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))]

    def load_active_chunks(self):
        with self._conn() as c:
            return [(r["text"], json.loads(r["meta"])) for r in c.execute(
                "SELECT c.text, c.meta FROM chunks c JOIN documents d ON d.id=c.doc_id "
                "WHERE d.status='active' ORDER BY c.doc_id, c.idx")]

    # ---------- change counter ----------
    def bump(self):
        with self._conn() as c:
            c.execute("INSERT INTO meta(key,value) VALUES('kb_version','1') "
                      "ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)")

    def kb_version(self):
        with self._conn() as c:
            r = c.execute("SELECT value FROM meta WHERE key='kb_version'").fetchone()
            return int(r["value"]) if r else 0
