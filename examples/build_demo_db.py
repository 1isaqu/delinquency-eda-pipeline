"""Build two small SQLite databases in the contract schema, filled with invented data.

Usage:  python examples/build_demo_db.py data/demo [--seed 7] [--as-of 2026-03-31]

Everything here is made up by a seeded random generator: people, agents, portfolios, amounts and
dates. The generator plants the situations the pipeline is built to find, so a demo run shows each
stage doing something:

* the longer the term granted, the likelier the slip is not paid on time (the effect stage 10
  tries to isolate from the mix),
* slips due in the last days before the reference date are still inside their grace window, so the
  naive "overdue unpaid" count is inflated (stage 05),
* a pseudo-agent that passes the default exclusion patterns and never collects (stages 07-08),
* re-written installments, slips without a debt link, zero balances, repeat debtors, people who
  appear in both product lines, and ids that repeat across the two lines.

Nothing here is calibrated on real data and the effect sizes mean nothing outside the demo.
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

DDL = """
CREATE TABLE agreement_statuses (status_id INTEGER PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE agents (agent_id TEXT PRIMARY KEY, display_name TEXT, login TEXT NOT NULL);
CREATE TABLE debts (
    debt_id INTEGER PRIMARY KEY, portfolio_id INTEGER, contract_no TEXT, balance REAL,
    due_at TEXT, days_overdue INTEGER
);
CREATE TABLE debt_attributes (debt_id INTEGER PRIMARY KEY, portfolio_label TEXT);
CREATE TABLE agreement_debts (agreement_id INTEGER, portfolio_id INTEGER, debt_id INTEGER);
CREATE TABLE debtors (debtor_id INTEGER PRIMARY KEY, person_key TEXT);
CREATE TABLE agreements (
    row_id INTEGER PRIMARY KEY, agreement_id INTEGER, portfolio_id INTEGER, debtor_id INTEGER,
    installment_no INTEGER, amount REAL, plan_installments INTEGER, paid_amount REAL,
    issued_at TEXT, due_at TEXT, paid_at TEXT, status_id INTEGER, agent_id TEXT
);
CREATE TABLE contact_results (
    result_id INTEGER PRIMARY KEY, label TEXT NOT NULL, answered INTEGER, right_party INTEGER
);
CREATE TABLE contact_events (
    event_id INTEGER PRIMARY KEY, debtor_id INTEGER, result_id INTEGER, agent_id TEXT,
    occurred_at TEXT
);
"""

STATUS_NAMES = (
    "ACTIVE",
    "BROKEN",
    "PAID",
    "PENDING_EXCEPTION",
    "REJECTED",
    "AUTO_BROKEN",
    "PAID_MANUAL",
)
RESULTS = [
    (1, "No answer", 0, 0),
    (2, "Answered, wrong party", 1, 0),
    (3, "Right-party contact", 1, 1),
    (4, "Voicemail", 0, 0),
    (5, "Message sent", 0, 0),
]
RESULT_WEIGHTS = [0.45, 0.15, 0.20, 0.10, 0.10]
PORTFOLIOS = [
    "Bank A I",
    "Bank A II",
    "Bank A III",
    "Bank B I",
    "Bank B II",
    "Bank C 01",
    "Bank C 02",
    "Bank D I",
]
GRACE = 3  # the generator pays "on time" within this many days after the due date


@dataclass(frozen=True)
class LineSpec:
    name: str
    status_base: int
    slips: int
    ticket_scale: float
    zero_balance_rate: float


LINES = (
    LineSpec("auto", 20, 900, 1500.0, 0.06),
    LineSpec("consumer", 30, 600, 150.0, 0.65),
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method; fine for the small rates used here."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def stamp(day: date, rng: random.Random) -> str:
    moment = datetime(day.year, day.month, day.day, rng.randint(8, 18), rng.randint(0, 59))
    return moment.strftime("%Y-%m-%d %H:%M:%S")


class LineBuilder:
    """Accumulates the rows of one product line, then writes them to a database file."""

    def __init__(self, spec: LineSpec, people: list[str], rng: random.Random, as_of: date):
        self.spec, self.people, self.rng, self.as_of = spec, people, rng, as_of
        self.status = {n: spec.status_base + i + 1 for i, n in enumerate(STATUS_NAMES)}
        self.agents = self._agents()
        self.agreements: list[tuple] = []
        self.debts: list[tuple] = []
        self.attributes: list[tuple] = []
        self.links: list[tuple] = []
        self.debtors: list[tuple] = []
        self.events: list[tuple] = []
        self.debtor_ids: dict[str, list[int]] = {}
        self.prior_breaks: dict[str, int] = {}
        self.next_row = self.next_debt = self.next_debtor = self.next_event = 1
        self.next_agreement = 1000  # deliberately the same start in every line

    # ---- entities -------------------------------------------------------------------------
    def _agents(self) -> dict[str, dict]:
        rng = self.rng
        agents = {
            f"a{i:02d}": {
                "name": f"Agent {i:02d}",
                "login": f"agent_{i:02d}",
                "shift": rng.uniform(-2.0, 6.0),  # how generous this agent is with terms
                "effect": rng.gauss(0.0, 0.25),
                "weight": 1.0,
            }
            for i in range(1, 13)
        }
        agents["sys"] = {
            "name": "System 01",
            "login": "system_batch",
            "shift": 0.0,
            "effect": 0.0,
            "weight": 0.4,
        }
        agents["tst"] = {
            "name": "Test 01",
            "login": "test_user",
            "shift": 0.0,
            "effect": 0.0,
            "weight": 0.3,
        }
        agents["ldr"] = {
            "name": "Batch Loader 01",
            "login": "batch_loader",
            "shift": 8.0,
            "effect": 0.0,
            "weight": 0.5,
        }
        return agents

    def _debtor(self, person: str | None) -> int:
        if person is None or self.rng.random() < 0.03:
            return self._new_debtor(None)
        ids = self.debtor_ids.setdefault(person, [])
        if not ids or (len(ids) < 2 and self.rng.random() < 0.15):
            ids.append(self._new_debtor(person))
        return self.rng.choice(ids)

    def _new_debtor(self, person: str | None) -> int:
        debtor_id = self.next_debtor
        self.next_debtor += 1
        self.debtors.append((debtor_id, person))
        return debtor_id

    def _earlier_breaks(self, person: str, debtor_id: int) -> None:
        """History: first-installment agreements of this person that ended broken, long ago."""
        if person in self.prior_breaks:
            return
        count = self.rng.choices([0, 1, 2, 3], weights=[0.8, 0.12, 0.05, 0.03])[0]
        self.prior_breaks[person] = count
        for _ in range(count):
            issued = self.as_of - timedelta(days=self.rng.randint(160, 400))
            status = self.rng.choice(["BROKEN", "BROKEN", "AUTO_BROKEN"])
            self._add_agreement_row(
                self._take_agreement(),
                1,
                debtor_id,
                0,
                120.0,
                1,
                None,
                issued,
                issued + timedelta(days=7),
                None,
                status,
                "a01",
            )

    # ---- rows -----------------------------------------------------------------------------
    def _take_agreement(self) -> int:
        agreement_id = self.next_agreement
        self.next_agreement += 1
        return agreement_id

    def _add_agreement_row(
        self,
        agreement_id,
        portfolio_id,
        debtor_id,
        installment,
        amount,
        plan,
        paid_amount,
        issued,
        due,
        paid_at,
        status,
        agent_id,
    ) -> None:
        self.agreements.append(
            (
                self.next_row,
                agreement_id,
                portfolio_id,
                debtor_id,
                installment,
                amount,
                plan,
                paid_amount,
                issued.isoformat(),
                due.isoformat(),
                paid_at.isoformat() if paid_at else None,
                self.status[status],
                agent_id,
            )
        )
        self.next_row += 1

    def _debts_for(self, agreement_id: int, portfolio_id: int, total: float) -> None:
        if self.rng.random() < 0.03:
            return  # no linked debt at all
        count = self.rng.choice([1, 1, 2])
        zero = self.rng.random() < self.spec.zero_balance_rate
        label = PORTFOLIOS[portfolio_id - 1]
        for _ in range(count):
            debt_id = self.next_debt
            self.next_debt += 1
            balance = 0.0 if zero else round(total * self.rng.uniform(1.0, 4.0) / count, 2)
            self.debts.append(
                (debt_id, portfolio_id, f"C-{debt_id:06d}", balance, "2025-09-01", 90)
            )
            shown = None if self.rng.random() < 0.02 else label
            self.attributes.append((debt_id, shown))
            self.links.append((agreement_id, portfolio_id, debt_id))

    def _outcome(self, agent: dict, term: int, first: float, prior: int, due: date, issued: date):
        """Decide payment for the first installment. Returns (paid_at, status)."""
        rng = self.rng
        ticket = max(first / self.spec.ticket_scale, 1e-3)
        logit = -0.70 + 0.12 * term + 0.30 * min(prior, 3) - 0.15 * math.log10(ticket)
        unpaid = rng.random() < sigmoid(logit + agent["effect"])
        if agent["login"] == "batch_loader":
            unpaid = True
        if not unpaid:
            paid = max(issued, due + timedelta(days=rng.randint(-2, GRACE)))
            if paid <= self.as_of:
                return paid, "PAID" if rng.random() < 0.9 else "ACTIVE"
        paid_late = None
        if agent["login"] != "batch_loader" and rng.random() < 0.15:
            late = due + timedelta(days=GRACE + rng.randint(1, 12))
            paid_late = late if late <= self.as_of else None
        if due + timedelta(days=7) < self.as_of:
            return paid_late, rng.choices(
                ["BROKEN", "AUTO_BROKEN", "ACTIVE"], weights=[0.45, 0.2, 0.35]
            )[0]
        return paid_late, "ACTIVE"

    def _events(self, debtor_id: int, agent_id: str, issued: date, due: date, unpaid: bool) -> None:
        rng = self.rng
        last = min(due, self.as_of)
        span = max((last - issued).days, 0)
        for _ in range(min(poisson(rng, 1.0 + (2.0 if unpaid else 0.0)), 8)):
            day = issued + timedelta(days=rng.randint(0, span))
            result = rng.choices(RESULTS, weights=RESULT_WEIGHTS)[0][0]
            self.events.append((self.next_event, debtor_id, result, agent_id, stamp(day, rng)))
            self.next_event += 1

    # ---- the slips ------------------------------------------------------------------------
    def build(self) -> None:
        rng = self.rng
        ids = list(self.agents)
        weights = [self.agents[a]["weight"] for a in ids]
        for _ in range(self.spec.slips):
            agent_id = rng.choices(ids, weights=weights)[0]
            self._slip(agent_id, self.agents[agent_id], rng.choice(self.people))

    def _slip(self, agent_id: str, agent: dict, person: str) -> None:
        rng = self.rng
        debtor_id = self._debtor(person)
        self._earlier_breaks(person, debtor_id)
        issued = self.as_of - timedelta(days=rng.randint(0, 150))
        term = min(40, int(rng.expovariate(1.0 / max(4.0 + agent["shift"], 1.0))))
        due = issued + timedelta(days=term)
        plan = rng.choice([1, 1, 2, 3, 6])
        total = rng.lognormvariate(math.log(self.spec.ticket_scale), 0.8)
        first = round(total / plan, 2)
        portfolio_id = rng.randint(1, len(PORTFOLIOS))
        agreement_id = self._take_agreement()
        paid_at, status = self._outcome(
            agent, term, first, self.prior_breaks.get(person, 0), due, issued
        )
        if rng.random() < 0.03:
            status = "PENDING_EXCEPTION"  # not a generated status: the extraction skips it
        paid_amount = first if paid_at else None
        self._add_agreement_row(
            agreement_id,
            portfolio_id,
            debtor_id,
            0,
            first,
            plan,
            paid_amount,
            issued,
            due,
            paid_at,
            status,
            agent_id,
        )
        if rng.random() < 0.02:  # re-written by mistake: the later row must win
            self._add_agreement_row(
                agreement_id,
                portfolio_id,
                debtor_id,
                0,
                first + 5,
                plan,
                paid_amount,
                issued + timedelta(days=1),
                due,
                paid_at,
                status,
                agent_id,
            )
        for k in range(1, plan):
            self._add_agreement_row(
                agreement_id,
                portfolio_id,
                debtor_id,
                k,
                first,
                plan,
                None,
                issued,
                due + timedelta(days=30 * k),
                None,
                status,
                agent_id,
            )
        self._debts_for(agreement_id, portfolio_id, total)
        unpaid = not (paid_at and paid_at <= due + timedelta(days=GRACE))
        self._events(debtor_id, agent_id, issued, due, unpaid)

    # ---- output ---------------------------------------------------------------------------
    def write(self, path: Path) -> Path:
        if path.exists():
            path.unlink()
        connection = sqlite3.connect(path)
        connection.executescript(DDL)
        connection.executemany(
            "INSERT INTO agreement_statuses VALUES (?, ?)", [(v, k) for k, v in self.status.items()]
        )
        connection.executemany(
            "INSERT INTO agents VALUES (?, ?, ?)",
            [(k, a["name"], a["login"]) for k, a in self.agents.items()],
        )
        connection.executemany("INSERT INTO debts VALUES (?,?,?,?,?,?)", self.debts)
        connection.executemany("INSERT INTO debt_attributes VALUES (?, ?)", self.attributes)
        connection.executemany("INSERT INTO agreement_debts VALUES (?, ?, ?)", self.links)
        connection.executemany("INSERT INTO debtors VALUES (?, ?)", self.debtors)
        connection.executemany("INSERT INTO contact_results VALUES (?, ?, ?, ?)", RESULTS)
        connection.executemany("INSERT INTO contact_events VALUES (?,?,?,?,?)", self.events)
        connection.executemany(
            "INSERT INTO agreements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", self.agreements
        )
        connection.commit()
        connection.close()
        return path


def build_demo(
    directory: str | Path, *, seed: int = 7, as_of: date = date(2026, 3, 31)
) -> list[Path]:
    """Write ``auto.db`` and ``consumer.db`` into ``directory`` and return their paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    people = [f"pk_{i:05d}" for i in range(int(sum(s.slips for s in LINES) * 0.55))]
    paths = []
    for spec in LINES:
        builder = LineBuilder(spec, people, rng, as_of)
        builder.build()
        paths.append(builder.write(target / f"{spec.name}.db"))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", help="where to write auto.db and consumer.db")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date(2026, 3, 31),
        help="the 'today' of the invented world (YYYY-MM-DD)",
    )
    args = parser.parse_args(argv)
    for path in build_demo(args.directory, seed=args.seed, as_of=args.as_of):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
