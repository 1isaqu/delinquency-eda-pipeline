"""A tiny invented database in the contract schema, written with inline rows.

Used by the extraction tests. Every value is made up. The rows are chosen so that each rule the
extraction follows has exactly one row that proves it (see the comments).

Reference date in the tests: 2026-03-31, grace period 3 days, window 2026-03-01 .. 2026-03-31.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from demo_loader import DDL

STATUS_NAMES = (
    "ACTIVE",
    "BROKEN",
    "PAID",
    "PENDING_EXCEPTION",
    "REJECTED",
    "AUTO_BROKEN",
    "PAID_MANUAL",
)

AGENTS = [
    ("a1", "Agent 01", "agent_01"),
    ("a2", "Agent 02", "agent_02"),
    ("sys", "System 01", "system_batch"),  # excluded by the default pattern on login
    ("tst", "Test 01", "test_user"),  # excluded by the default pattern on login
]

# row_id, agreement, portfolio, debtor, installment, amount, plan, paid_amount, issued, due,
# paid_at, status, agent
AGREEMENTS = [
    # 101: paid on the day after the due date, inside the grace period -> paid_on_time.
    (1, 101, 1, 1, 0, 100.0, 3, 100.0, "2026-03-01", "2026-03-05", "2026-03-06", "PAID", "a1"),
    # 102: never paid -> overdue_unpaid, mature. Has installments 1 and 2 (see below).
    (2, 102, 1, 2, 0, 200.0, 3, None, "2026-03-02", "2026-03-10", None, "ACTIVE", "a1"),
    # 103: paid, but after the grace period -> still overdue_unpaid.
    (3, 103, 1, 3, 0, 300.0, 1, 300.0, "2026-03-03", "2026-03-15", "2026-03-25", "ACTIVE", "a2"),
    # 104: due after the reference date -> to_mature (and outside the March window).
    (4, 104, 2, 4, 0, 400.0, 2, None, "2026-03-20", "2026-04-10", None, "ACTIVE", "a2"),
    # 105: owned by an excluded agent -> never extracted.
    (5, 105, 1, 1, 0, 50.0, 1, None, "2026-03-02", "2026-03-12", None, "ACTIVE", "sys"),
    # 106: pending exception is not a generated status -> never extracted.
    (6, 106, 1, 2, 0, 60.0, 1, None, "2026-03-02", "2026-03-12", None, "PENDING_EXCEPTION", "a1"),
    # 107: due before the window -> never extracted.
    (7, 107, 1, 2, 0, 70.0, 1, None, "2026-01-20", "2026-02-10", None, "ACTIVE", "a1"),
    # 108: re-written first installment. The later row (issued later) must win.
    (8, 108, 1, 2, 0, 150.0, 1, None, "2026-03-04", "2026-03-12", None, "ACTIVE", "a1"),
    (9, 108, 1, 2, 0, 175.0, 1, None, "2026-03-06", "2026-03-14", None, "BROKEN", "a1"),
    # 109: the later row has a status outside the filter. Because filters run before the ranking,
    # the earlier, valid row stays visible. Ranking first would hide the agreement.
    (10, 109, 1, 3, 0, 80.0, 1, None, "2026-03-05", "2026-03-13", None, "ACTIVE", "a2"),
    (11, 109, 1, 3, 0, 90.0, 1, None, "2026-03-07", "2026-03-13", None, "PENDING_EXCEPTION", "a2"),
    # later installments of 102; installment 2 was re-written, the later amount wins.
    (12, 102, 1, 2, 1, 200.0, 3, None, "2026-03-02", "2026-04-10", None, "ACTIVE", "a1"),
    (13, 102, 1, 2, 2, 200.0, 3, None, "2026-03-02", "2026-05-10", None, "ACTIVE", "a1"),
    (14, 102, 1, 2, 2, 250.0, 3, None, "2026-03-09", "2026-05-10", None, "ACTIVE", "a1"),
    # earlier first-installment agreements that ended broken: history for repeat-break counts.
    (20, 90, 1, 1, 0, 100.0, 1, None, "2025-12-01", "2025-12-08", None, "BROKEN", "a1"),
    (21, 91, 1, 5, 0, 100.0, 1, None, "2025-11-01", "2025-11-08", None, "AUTO_BROKEN", "a1"),
    (22, 92, 1, 6, 0, 100.0, 1, None, "2025-12-01", "2025-12-08", None, "BROKEN", "a1"),
    # a break that happened after 102 was made does not count as earlier history for 102.
    (23, 93, 1, 2, 0, 100.0, 1, None, "2026-03-20", "2026-04-02", None, "BROKEN", "a1"),
]

DEBTS = [
    (1, 1, "C-1", 500.0, "2026-01-01", 60),
    (2, 1, "C-2", 0.0, "2026-01-01", 60),  # zero balance: linked, but nothing left
    (3, 1, "C-3", 100.0, "2026-01-01", 60),
    (4, 1, "C-4", 800.0, "2026-01-01", 60),
    (5, 1, "C-5", 250.0, "2026-01-01", 60),
]
DEBT_ATTRIBUTES = [
    (1, "Portfolio A-01"),
    (2, "Portfolio A-02"),
    (3, "Portfolio B-01"),
    (4, None),
    (5, "Portfolio C-02"),
]
LINKS = [
    (101, 1, 1),
    (102, 1, 3),  # inserted before debt 2 on purpose: the lowest debt_id must still be chosen
    (102, 1, 2),
    (103, 1, 4),
    (108, 1, 5),
    # 104 and 109 have no linked debt at all.
]
DEBTORS = [
    (1, "pk_alpha"),
    (2, "pk_beta"),
    (3, "pk_alpha"),  # same person as debtor 1, another debtor row
    (4, "pk_gamma"),
    (5, "pk_alpha"),  # a third debtor row of the same person
    (6, "pk_delta"),
]
RESULTS = [
    (1, "No answer", 0, 0),
    (2, "Answered, wrong party", 1, 0),
    (3, "Right-party contact", 1, 1),
]
EVENTS = [
    (1, 2, 1, "a1", "2026-03-02 09:00:00"),  # on the agreement day of 102: not follow-up
    (2, 2, 1, "a1", "2026-03-05 10:00:00"),
    (3, 2, 3, "a1", "2026-03-07 11:00:00"),
    (4, 1, 2, "a2", "2026-03-04 09:30:00"),  # debtor 1 belongs to the same person as 3 and 5
    (5, 5, 1, "a2", "2026-03-10 14:00:00"),
    (6, 2, 3, "a1", "2026-04-20 09:00:00"),  # outside every contact window
]


def make_world(path: Path, *, status_base: int = 20, rows: list | None = None) -> Path:
    """Create the database at ``path``. ``status_base`` shifts status ids to prove that statuses
    are resolved by label, never by number."""
    ids = {name: status_base + i + 1 for i, name in enumerate(STATUS_NAMES)}
    connection = sqlite3.connect(path)
    connection.executescript(DDL)
    connection.executemany(
        "INSERT INTO agreement_statuses VALUES (?, ?)", [(v, k) for k, v in ids.items()]
    )
    connection.executemany("INSERT INTO agents VALUES (?, ?, ?)", AGENTS)
    connection.executemany("INSERT INTO debts VALUES (?, ?, ?, ?, ?, ?)", DEBTS)
    connection.executemany("INSERT INTO debt_attributes VALUES (?, ?)", DEBT_ATTRIBUTES)
    connection.executemany("INSERT INTO agreement_debts VALUES (?, ?, ?)", LINKS)
    connection.executemany("INSERT INTO debtors VALUES (?, ?)", DEBTORS)
    connection.executemany("INSERT INTO contact_results VALUES (?, ?, ?, ?)", RESULTS)
    connection.executemany("INSERT INTO contact_events VALUES (?, ?, ?, ?, ?)", EVENTS)
    for r in rows if rows is not None else AGREEMENTS:
        connection.execute(
            "INSERT INTO agreements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*r[:11], ids[r[11]], r[12]),
        )
    connection.commit()
    connection.close()
    return path


def make_second_line(path: Path) -> Path:
    """A second product line that reuses numeric ids of the first: composite keys must hold."""
    connection = sqlite3.connect(path)
    connection.executescript(DDL)
    ids = {name: 30 + i + 1 for i, name in enumerate(STATUS_NAMES)}
    connection.executemany(
        "INSERT INTO agreement_statuses VALUES (?, ?)", [(v, k) for k, v in ids.items()]
    )
    connection.executemany("INSERT INTO agents VALUES (?, ?, ?)", AGENTS[:2])
    connection.executemany("INSERT INTO debts VALUES (?, ?, ?, ?, ?, ?)", DEBTS[:1])
    connection.executemany("INSERT INTO debt_attributes VALUES (?, ?)", [(1, "Portfolio D-07")])
    connection.executemany("INSERT INTO agreement_debts VALUES (?, ?, ?)", [(101, 1, 1)])
    connection.executemany("INSERT INTO debtors VALUES (?, ?)", [(1, "pk_alpha")])
    connection.executemany("INSERT INTO contact_results VALUES (?, ?, ?, ?)", RESULTS)
    connection.executemany(
        "INSERT INTO contact_events VALUES (?, ?, ?, ?, ?)",
        [(1, 1, 3, "a1", "2026-03-04 08:00:00")],
    )
    # Same numeric agreement id 101 and debtor id 1 as the first line, but a different agreement.
    connection.execute(
        "INSERT INTO agreements VALUES "
        "(1,101,1,1,0,999.0,1,NULL,'2026-03-02','2026-03-12',NULL,?,'a1')",
        (ids["ACTIVE"],),
    )
    connection.commit()
    connection.close()
    return path


def damage(path: Path, *statements: str) -> Path:
    """Run SQL that breaks the database in one chosen way. The tests then check which rule
    notices. Each statement may hold several commands."""
    connection = sqlite3.connect(path)
    for statement in statements:
        connection.executescript(statement)
    connection.commit()
    connection.close()
    return path


def without_key(table: str, columns: str, *, extra: str = "") -> str:
    """SQL that rebuilds ``table`` with no key, so a duplicate row can be inserted. ``columns`` is
    the column list with types and ``extra`` holds the INSERT of the duplicate."""
    return (
        f"CREATE TABLE {table}_copy ({columns});"
        f"INSERT INTO {table}_copy SELECT * FROM {table};"
        f"{extra}"
        f"DROP TABLE {table}; ALTER TABLE {table}_copy RENAME TO {table};"
    )
