"""Central store (proposal L1 and "State and memory"): requests, constraint
records, timetable versions, the concession ledger and an audit log, in
SQLite (``:memory:`` for tests and runs, a file for the portal).

Justifications and raw request text stay here with restricted access; what
leaves the store for explanations is the constraint, never the reason.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime

from core.instance import ExtraClass
from core.schemas import (
    ConcessionEntry,
    Constraint,
    Placement,
    Request,
    RequestStatus,
    TimetableVersion,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, body TEXT NOT NULL, status TEXT NOT NULL,
                                     dedupe_key TEXT, thread_id TEXT);
CREATE TABLE IF NOT EXISTS constraints (id TEXT PRIMARY KEY, body TEXT NOT NULL, active INTEGER NOT NULL,
                                        request_id TEXT);
CREATE TABLE IF NOT EXISTS versions (version INTEGER PRIMARY KEY, body TEXT NOT NULL, parent INTEGER,
                                     approved_by TEXT, published INTEGER NOT NULL DEFAULT 0, case_id TEXT,
                                     created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS extras (id TEXT PRIMARY KEY, body TEXT NOT NULL, week INTEGER NOT NULL,
                                   active INTEGER NOT NULL, request_id TEXT);
CREATE TABLE IF NOT EXISTS ledger (n INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (n INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, at TEXT NOT NULL,
                                   kind TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS seen (person TEXT NOT NULL, item TEXT NOT NULL, n INTEGER NOT NULL,
                                 PRIMARY KEY (person, item));
CREATE TABLE IF NOT EXISTS chat (n INTEGER PRIMARY KEY AUTOINCREMENT, person TEXT NOT NULL, at TEXT NOT NULL,
                                 role TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hidden (person TEXT NOT NULL, item TEXT NOT NULL, PRIMARY KEY (person, item));
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


EXTRA_PREFIX = "X-"  # session ids of extra classes (ExtraClass), never used by the instance's own sessions


class _Rows(list):
    """A query's rows, already fetched; read like a cursor."""

    def fetchone(self):
        return self[0] if self else None

    def fetchall(self) -> list:
        return list(self)


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def load(self, data: bytes) -> None:
        """Replace the database with a saved one (``db.serialize()``), adding tables an older version lacks.

        On a new connection: ``deserialize`` keeps the old one's prepared statements, and those go on
        reading the tables where they were before, which is somewhere else in a database whose tables
        were not all created at once."""
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.deserialize(data)
        db.executescript(SCHEMA)
        with self._lock:
            old, self.db = self.db, db
        old.close()

    def _exec(self, sql: str, args: tuple = ()) -> _Rows:
        # rows are read under the lock: the connection is shared, and another thread's
        # statement or commit can reset a cursor that is still being read
        with self._lock:
            rows = _Rows(self.db.execute(sql, args).fetchall())
            self.db.commit()
            return rows

    # -- requests ----------------------------------------------------------------

    def add_request(self, r: Request, dedupe_key: str | None = None) -> bool:
        """False if an identical request (same key) is already stored; a withdrawn one does not count,
        so the sender can send it again."""
        if dedupe_key and self._exec("SELECT 1 FROM requests WHERE dedupe_key=? AND status!=?",
                                     (dedupe_key, RequestStatus.WITHDRAWN.value)).fetchone():
            return False
        self._exec("INSERT INTO requests VALUES (?,?,?,?,?)",
                   (r.id, r.model_dump_json(), r.status.value, dedupe_key, r.thread_id))
        return True

    def add_new_request(self, make: Callable[[str], Request], dedupe_key: str | None = None) -> Request | None:
        """Number and store a new request in one step: ``make`` builds it from its id. Two requests
        arriving together (two users, or one while a world is seeding) would otherwise get the same id.
        None if an identical request is already stored, as ``add_request``."""
        with self._lock:
            if dedupe_key and self.db.execute("SELECT 1 FROM requests WHERE dedupe_key=? AND status!=?",
                                              (dedupe_key, RequestStatus.WITHDRAWN.value)).fetchone():
                return None
            (n,) = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()
            r = make(f"R-{n + 1:05d}")
            self.db.execute("INSERT INTO requests VALUES (?,?,?,?,?)",
                            (r.id, r.model_dump_json(), r.status.value, dedupe_key, r.thread_id))
            self.db.commit()
            return r

    def save_request(self, r: Request) -> None:
        self._exec("UPDATE requests SET body=?, status=? WHERE id=?", (r.model_dump_json(), r.status.value, r.id))

    def request(self, rid: str) -> Request | None:
        row = self._exec("SELECT body FROM requests WHERE id=?", (rid,)).fetchone()
        return Request.model_validate_json(row[0]) if row else None

    def requests(self, status: RequestStatus | None = None) -> list[Request]:
        rows = (self._exec("SELECT body FROM requests WHERE status=?", (status.value,)) if status
                else self._exec("SELECT body FROM requests"))
        return [Request.model_validate_json(b) for (b,) in rows.fetchall()]

    # -- constraints ---------------------------------------------------------------

    def add_constraints(self, cons: list[Constraint], request_id: str | None = None, active: bool = True) -> None:
        for c in cons:
            self._exec("INSERT OR REPLACE INTO constraints VALUES (?,?,?,?)",
                       (c.id, c.model_dump_json(), int(active), request_id))

    def set_active(self, cid: str, active: bool) -> None:
        self._exec("UPDATE constraints SET active=? WHERE id=?", (int(active), cid))

    def request_constraints(self, request_id: str) -> list[str]:
        """IDs of the constraints a request added (its own, and any a negotiation reply added for it)."""
        return [cid for (cid,) in self._exec("SELECT id FROM constraints WHERE request_id=?", (request_id,)).fetchall()]

    def constraints(self, active_only: bool = True) -> list[Constraint]:
        sql = "SELECT body FROM constraints" + (" WHERE active=1" if active_only else "")
        return [Constraint.model_validate_json(b) for (b,) in self._exec(sql).fetchall()]

    # -- extra classes ---------------------------------------------------------------

    def add_extra(self, x: ExtraClass, active: bool = True) -> None:
        self._exec("INSERT OR REPLACE INTO extras VALUES (?,?,?,?,?)",
                   (x.session.id, x.model_dump_json(), x.week, int(active), x.request))

    def set_extras_active(self, request_id: str, active: bool) -> None:
        self._exec("UPDATE extras SET active=? WHERE request_id=?", (int(active), request_id))

    def extras(self, week: int | None = None, active_only: bool = True) -> list[ExtraClass]:
        """Extra classes, all weeks or one; withdrawn ones too with ``active_only=False`` (old versions
        still name them)."""
        sql, args = "SELECT body FROM extras WHERE 1=1", []
        if week is not None:
            sql += " AND week=?"
            args.append(week)
        if active_only:
            sql += " AND active=1"
        return [ExtraClass.model_validate_json(b) for (b,) in self._exec(sql + " ORDER BY id", tuple(args)).fetchall()]

    # -- timetable versions ----------------------------------------------------------

    def propose_version(self, assignment: dict[str, Placement], case_id: str,
                        week: int | None = None, cancelled: list[str] | None = None,
                        parent: TimetableVersion | None = None) -> TimetableVersion:
        """A version waiting for approval, built on ``parent`` (by default the published timetable)."""
        cur = parent or self.current_version(week) or self.current_version()
        if week is not None:  # the week's extra classes are held wherever they were placed
            assignment = {**{s: p for s, p in assignment.items() if not s.startswith(EXTRA_PREFIX)},
                          **{x.session.id: x.placement for x in self.extras(week)}}
        (n,) = self._exec("SELECT COALESCE(MAX(version), 0) FROM versions").fetchone()
        v = TimetableVersion(version=n + 1, assignment=assignment, parent=cur.version if cur else None, week=week,
                             cancelled=sorted(cancelled or []))
        self._exec("INSERT INTO versions VALUES (?,?,?,?,?,?,?)",
                   (v.version, v.model_dump_json(), v.parent, None, 0, case_id, datetime.now().isoformat()))
        return v

    def publish_version(self, version: int, approved_by: str) -> TimetableVersion:
        row = self._exec("SELECT body FROM versions WHERE version=?", (version,)).fetchone()
        if row is None:
            raise KeyError(version)
        v = TimetableVersion.model_validate_json(row[0]).model_copy(update={"approved_by": approved_by})
        self._exec("UPDATE versions SET body=?, approved_by=?, published=1 WHERE version=?",
                   (v.model_dump_json(), approved_by, version))
        return v

    def current_version(self, week: int | None = None) -> TimetableVersion | None:
        """Latest published semester timetable, or for ``week`` the latest
        published repair of that week (None if the week was never repaired)."""
        for (body,) in self._exec("SELECT body FROM versions WHERE published=1 ORDER BY version DESC").fetchall():
            v = TimetableVersion.model_validate_json(body)
            if v.week == week:
                return v
        return None

    def version(self, version: int) -> TimetableVersion | None:
        row = self._exec("SELECT body FROM versions WHERE version=?", (version,)).fetchone()
        return TimetableVersion.model_validate_json(row[0]) if row else None

    def rollback(self, to_version: int, approved_by: str) -> TimetableVersion:
        """Publish a copy of an earlier version as the newest one."""
        old = self.version(to_version)
        if old is None:
            raise KeyError(to_version)
        v = self.propose_version(old.assignment, case_id=f"rollback-{to_version}", week=old.week)
        return self.publish_version(v.version, approved_by)

    # -- ledger and audit log ----------------------------------------------------------

    def record_concession(self, e: ConcessionEntry) -> None:
        self._exec("INSERT INTO ledger (body) VALUES (?)", (e.model_dump_json(),))

    def remove_concession(self, e: ConcessionEntry) -> bool:
        """Take one recorded concession back out of the ledger (its request was withdrawn)."""
        row = self._exec("SELECT n FROM ledger WHERE body=? ORDER BY n DESC LIMIT 1", (e.model_dump_json(),)).fetchone()
        if row is None:
            return False
        self._exec("DELETE FROM ledger WHERE n=?", (row[0],))
        return True

    def ledger(self) -> list[ConcessionEntry]:
        return [ConcessionEntry.model_validate_json(b) for (b,) in
                self._exec("SELECT body FROM ledger ORDER BY n").fetchall()]

    def log(self, case_id: str | None, kind: str, **body) -> None:
        self._exec("INSERT INTO events (case_id, at, kind, body) VALUES (?,?,?,?)",
                   (case_id, datetime.now().isoformat(), kind, json.dumps(body, default=str)))

    def events(self, case_id: str | None = None, kind: str | None = None, after: int = 0,
               limit: int | None = None) -> list[dict]:
        sql, args = "SELECT n, case_id, at, kind, body FROM events WHERE n > ?", [after]
        if case_id:
            sql += " AND case_id=?"
            args.append(case_id)
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        rows = self._exec(sql + " ORDER BY n" + (f" LIMIT {int(limit)}" if limit else ""), tuple(args)).fetchall()
        return [{"n": n, "case": c, "at": a, "kind": k, **json.loads(b)} for n, c, a, k, b in rows]

    # -- what each person has read in their inbox ----------------------------------

    def seen(self, person: str) -> dict[str, int]:
        """Inbox item -> the last event of it ``person`` has read (a later event makes it unread again)."""
        return dict(self._exec("SELECT item, n FROM seen WHERE person=?", (person,)).fetchall())

    def mark_seen(self, person: str, items: dict[str, int]) -> None:
        with self._lock:
            self.db.executemany("INSERT INTO seen (person, item, n) VALUES (?,?,?) "
                                "ON CONFLICT (person, item) DO UPDATE SET n=max(n, excluded.n)",
                                [(person, i, n) for i, n in items.items()])
            self.db.commit()

    # -- the chat (web.app /api/chat) -------------------------------------------------

    def add_chat(self, person: str, role: str, kind: str, body: dict) -> dict:
        """One message of ``person``'s chat: ``role`` "you" or "assistant"; ``kind`` "text" (what they wrote),
        "lookup" (a timetable answer), "request" (a request, followed in the card), "notice" (something that
        needs them). Returns it as ``chat`` lists it."""
        at = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self.db.execute("INSERT INTO chat (person, at, role, kind, body) VALUES (?,?,?,?,?)",
                                  (person, at, role, kind, json.dumps(body)))
            self.db.commit()
            n = cur.lastrowid
        return {"n": n, "at": at, "role": role, "kind": kind, **body}

    def chat(self, person: str, after: int = 0) -> list[dict]:
        """``person``'s messages after number ``after``, oldest first."""
        rows = self._exec("SELECT n, at, role, kind, body FROM chat WHERE person=? AND n>? ORDER BY n", (person, after))
        return [{"n": n, "at": at, "role": role, "kind": kind, **json.loads(body)} for n, at, role, kind, body in rows]

    def chat_message(self, person: str, n: int) -> dict | None:
        """One of ``person``'s messages by its number (None: no such message, or someone else's)."""
        row = self._exec("SELECT n, at, role, kind, body FROM chat WHERE person=? AND n=?", (person, n)).fetchone()
        return {"n": row[0], "at": row[1], "role": row[2], "kind": row[3], **json.loads(row[4])} if row else None

    def update_chat(self, person: str, n: int, changes: dict) -> None:
        """Adds or changes fields of one of ``person``'s messages (e.g. a confirm, once answered)."""
        m = self.chat_message(person, n)
        if m is not None:
            body = {k: v for k, v in m.items() if k not in ("n", "at", "role", "kind")} | changes
            self._exec("UPDATE chat SET body=? WHERE person=? AND n=?", (json.dumps(body), person, n))

    def chat_has(self, person: str, kind: str, key: str, value: str) -> bool:
        """Whether ``person``'s chat already has a ``kind`` message with ``body[key] == value``."""
        rows = self._exec("SELECT body FROM chat WHERE person=? AND kind=?", (person, kind))
        return any(json.loads(b).get(key) == value for (b,) in rows)

    def hide(self, person: str, item: str) -> None:
        """Takes ``item`` (a request) off ``person``'s own list; it stays in the records."""
        self._exec("INSERT OR IGNORE INTO hidden (person, item) VALUES (?,?)", (person, item))

    def hidden(self, person: str) -> set[str]:
        return {i for (i,) in self._exec("SELECT item FROM hidden WHERE person=?", (person,))}

    def setting(self, key: str, default=None):
        """A value the coordinator set (``set_setting``), or ``default``."""
        row = self._exec("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value) -> None:
        self._exec("INSERT INTO settings (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, json.dumps(value)))

    def clear_chat(self, person: str) -> int:
        """Empties ``person``'s chat (their requests stay, in the Requests list). Returns how many went."""
        return len(self._exec("DELETE FROM chat WHERE person=? RETURNING n", (person,)))

    def versions(self) -> list[dict]:
        rows = self._exec("SELECT version, parent, approved_by, published, case_id, created_at, body "
                          "FROM versions ORDER BY version").fetchall()
        return [{"version": v, "parent": p, "approved_by": a, "published": bool(pub), "case": c,
                 "created_at": t, "week": json.loads(b).get("week")} for v, p, a, pub, c, t, b in rows]
