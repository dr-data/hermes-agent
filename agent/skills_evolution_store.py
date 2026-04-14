#!/usr/bin/env python3
"""Skill evolution ledger for Hermes.

Tracks skill mutations over time with parent links, snapshots, and unified diffs.
Designed as a lightweight, profile-scoped SQLite store inspired by OpenSpace's
lineage model but adapted to Hermes skill manager flows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Dict, List, Optional, Any

from hermes_constants import get_hermes_home


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def collect_skill_snapshot(skill_dir: Path) -> Dict[str, str]:
    """Collect UTF-8 text snapshot of all files under a skill directory."""
    snapshot: Dict[str, str] = {}
    if not skill_dir.exists() or not skill_dir.is_dir():
        return snapshot

    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(skill_dir).as_posix()
        try:
            snapshot[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip binary/non-utf8 files in snapshots for now.
            continue
    return snapshot


def compute_unified_diff(before: Dict[str, str], after: Dict[str, str]) -> str:
    """Build a combined git-style unified diff across snapshot dictionaries."""
    output: List[str] = []
    files = sorted(set(before.keys()) | set(after.keys()))

    for rel in files:
        old = before.get(rel, "")
        new = after.get(rel, "")
        if old == new:
            continue

        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
        output.extend(diff)

    return "\n".join(output)


@dataclass
class SkillVersionRecord:
    version_id: str
    skill_name: str
    skill_path: str
    action: str
    parent_version_id: Optional[str]
    created_at: str
    actor: str
    summary: str
    snapshot: Dict[str, str]
    diff_text: str


class SkillsEvolutionStore:
    """SQLite-backed persistence for skill lineage metadata."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            hub_dir = get_hermes_home() / "skills" / ".hub"
            hub_dir.mkdir(parents=True, exist_ok=True)
            db_path = hub_dir / "evolution.db"

        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=20.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=20000")
        self._init_db()

    def _init_db(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS skill_versions (
            version_id TEXT PRIMARY KEY,
            skill_name TEXT NOT NULL,
            skill_path TEXT NOT NULL,
            action TEXT NOT NULL,
            parent_version_id TEXT,
            created_at TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            snapshot_json TEXT NOT NULL DEFAULT '{}',
            diff_text TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_skill_versions_name_time
        ON skill_versions(skill_name, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_skill_versions_parent
        ON skill_versions(parent_version_id);
        """
        with self._lock:
            self._conn.executescript(ddl)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def latest_version(self, skill_name: str) -> Optional[str]:
        query = """
        SELECT version_id
        FROM skill_versions
        WHERE skill_name = ?
        ORDER BY created_at DESC
        LIMIT 1
        """
        with self._lock:
            row = self._conn.execute(query, (skill_name,)).fetchone()
        return row["version_id"] if row else None

    def record_version(
        self,
        *,
        skill_name: str,
        skill_path: str,
        action: str,
        parent_version_id: Optional[str] = None,
        actor: str = "system",
        summary: str = "",
        snapshot: Optional[Dict[str, str]] = None,
        diff_text: str = "",
    ) -> str:
        version_id = f"sv_{uuid.uuid4().hex[:12]}"
        payload = (
            version_id,
            skill_name,
            _safe_text(skill_path),
            _safe_text(action),
            parent_version_id,
            _utc_now_iso(),
            _safe_text(actor),
            _safe_text(summary),
            json.dumps(snapshot or {}, ensure_ascii=False),
            _safe_text(diff_text),
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO skill_versions (
                    version_id, skill_name, skill_path, action,
                    parent_version_id, created_at, actor, summary,
                    snapshot_json, diff_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            self._conn.commit()

        return version_id

    def list_skills(self, *, limit: int = 100, query: str = "") -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        q = (query or "").strip().lower()

        sql = """
        SELECT
            skill_name,
            MAX(created_at) AS last_updated,
            COUNT(*) AS version_count,
            SUM(CASE WHEN action = 'delete' THEN 1 ELSE 0 END) AS delete_events
        FROM skill_versions
        {where_clause}
        GROUP BY skill_name
        ORDER BY last_updated DESC
        LIMIT ?
        """

        params: List[Any] = []
        where_clause = ""
        if q:
            where_clause = "WHERE LOWER(skill_name) LIKE ?"
            params.append(f"%{q}%")
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql.format(where_clause=where_clause), tuple(params)).fetchall()

        return [
            {
                "skill_name": row["skill_name"],
                "last_updated": row["last_updated"],
                "version_count": int(row["version_count"] or 0),
                "deleted": bool(row["delete_events"]),
            }
            for row in rows
        ]

    def get_skill_lineage(self, skill_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT version_id, skill_name, skill_path, action, parent_version_id,
                       created_at, actor, summary, diff_text
                FROM skill_versions
                WHERE skill_name = ?
                ORDER BY created_at ASC
                """,
                (skill_name,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_version(self, version_id: str) -> Optional[SkillVersionRecord]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT version_id, skill_name, skill_path, action, parent_version_id,
                       created_at, actor, summary, snapshot_json, diff_text
                FROM skill_versions
                WHERE version_id = ?
                LIMIT 1
                """,
                (version_id,),
            ).fetchone()

        if not row:
            return None

        snapshot_json = row["snapshot_json"] or "{}"
        try:
            snapshot = json.loads(snapshot_json)
        except Exception:
            snapshot = {}

        return SkillVersionRecord(
            version_id=row["version_id"],
            skill_name=row["skill_name"],
            skill_path=row["skill_path"],
            action=row["action"],
            parent_version_id=row["parent_version_id"],
            created_at=row["created_at"],
            actor=row["actor"],
            summary=row["summary"],
            snapshot=snapshot if isinstance(snapshot, dict) else {},
            diff_text=row["diff_text"] or "",
        )


_STORE: Optional[SkillsEvolutionStore] = None


def get_skills_evolution_store() -> SkillsEvolutionStore:
    global _STORE
    if _STORE is None:
        _STORE = SkillsEvolutionStore()
    return _STORE
