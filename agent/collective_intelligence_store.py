#!/usr/bin/env python3
"""Collective agent intelligence store.

Persists delegation outcomes and serves lightweight retrieval so future delegated
subtasks can benefit from prior successful patterns.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_query(text: str) -> str:
    return " ".join(token.lower() for token in _TOKEN_RE.findall(text or ""))


def _trim(text: str, max_len: int = 1600) -> str:
    value = (text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


class CollectiveIntelligenceStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            hub_dir = get_hermes_home() / "skills" / ".hub"
            hub_dir.mkdir(parents=True, exist_ok=True)
            db_path = hub_dir / "collective_intelligence.db"

        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=20.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=20000")
        self._init_db()

    def _init_db(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS collective_patterns (
            id TEXT PRIMARY KEY,
            task_goal TEXT NOT NULL,
            normalized_query TEXT NOT NULL,
            context_excerpt TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'unknown',
            duration_seconds REAL NOT NULL DEFAULT 0,
            toolsets_json TEXT NOT NULL DEFAULT '[]',
            score REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_collective_query
          ON collective_patterns(normalized_query);
        CREATE INDEX IF NOT EXISTS idx_collective_created
          ON collective_patterns(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_collective_status
          ON collective_patterns(status);
        """
        with self._lock:
            self._conn.executescript(ddl)
            self._conn.commit()

    def record_result(
        self,
        *,
        task_goal: str,
        context_excerpt: str,
        summary: str,
        status: str,
        duration_seconds: float,
        toolsets: Optional[List[str]] = None,
    ) -> str:
        score = 1.0 if (status or "").lower() == "completed" else 0.2
        if duration_seconds and duration_seconds > 0:
            score = max(0.1, score - min(duration_seconds / 600.0, 0.4))

        row_id = f"ci_{uuid.uuid4().hex[:12]}"
        payload = (
            row_id,
            _trim(task_goal, 600),
            _normalize_query(task_goal),
            _trim(context_excerpt, 800),
            _trim(summary, 3000),
            _trim(status, 32) or "unknown",
            float(duration_seconds or 0),
            json.dumps(toolsets or [], ensure_ascii=False),
            float(score),
            _utc_iso(),
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO collective_patterns (
                    id, task_goal, normalized_query, context_excerpt, summary,
                    status, duration_seconds, toolsets_json, score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            self._conn.commit()

        return row_id

    def find_relevant(self, query: str, *, limit: int = 3) -> List[Dict[str, Any]]:
        tokens = _normalize_query(query).split()
        if not tokens:
            return []

        clauses = " OR ".join("normalized_query LIKE ?" for _ in tokens)
        params: List[Any] = [f"%{tok}%" for tok in tokens]
        params.append(max(1, min(int(limit), 20)))

        sql = f"""
        SELECT id, task_goal, summary, status, duration_seconds, toolsets_json, score, created_at
        FROM collective_patterns
        WHERE {clauses}
        ORDER BY score DESC, created_at DESC
        LIMIT ?
        """

        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            try:
                toolsets = json.loads(row["toolsets_json"] or "[]")
            except Exception:
                toolsets = []
            results.append(
                {
                    "id": row["id"],
                    "task_goal": row["task_goal"],
                    "summary": row["summary"],
                    "status": row["status"],
                    "duration_seconds": row["duration_seconds"],
                    "toolsets": toolsets if isinstance(toolsets, list) else [],
                    "score": row["score"],
                    "created_at": row["created_at"],
                }
            )
        return results


_STORE: Optional[CollectiveIntelligenceStore] = None


def get_collective_intelligence_store() -> CollectiveIntelligenceStore:
    global _STORE
    if _STORE is None:
        _STORE = CollectiveIntelligenceStore()
    return _STORE
