"""D170-R2U: append-only, fail-closed CPU event journal (NO TRAINING).

This tool does not import or execute competitor algorithms. It is a measurement
interface for future separately authorised, external-source native drivers.
One writer per journal; completed events survive process interruption. Unknown
or incomplete costs are never converted into zero or counted as completed.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from tools.cw20_protocol import CW20
from tools.d170_r2t_cost import KINDS

SCHEMA = "D170-R2U/event-journal-v1"
METHOD_SHA = {
    "nevo": "fe0456e1ae5382213d12421270209eb3f47a9a0f",
    "ewc": "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa",
    "packnet": "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa",
}
FINALS = ("complete", "interrupted")
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")


class JournalError(ValueError):
    pass


def check(condition, reason):
    if not condition:
        raise JournalError(reason)


def valid_nonnegative(value, *, nullable=False):
    return (nullable and value is None) or (type(value) is int and value >= 0)


def _valid_identity(row):
    check(row.get("schema") == SCHEMA, "Unexpected event schema")
    check(row.get("phase") in ("start",) + FINALS, "Unknown event lifecycle phase")
    check(isinstance(row.get("execution_id"), str) and
          TOKEN.fullmatch(row["execution_id"]), "Invalid execution identity")
    check(isinstance(row.get("event_id"), str) and
          TOKEN.fullmatch(row["event_id"]) and
          row["event_id"].startswith(row["execution_id"]+":"),
          "Invalid or unbound event ID")
    check(row.get("method") in METHOD_SHA, "Unknown original public method")
    check(row.get("source_revision") == METHOD_SHA[row["method"]],
          "Wrong pinned public source SHA")
    s = row.get("stage_index")
    check(type(s) is int and 0 <= s < len(CW20), "Invalid original stage index")
    check(row.get("task_identity") == CW20[s], "Original CW20 task mismatch")
    check(row.get("operation") in KINDS, "Unknown native operation")
    check(row.get("scope") in ("whole", "actor", "critic", "temperature",
                               "population", "encoder", "simulator",
                               "replay", "task_transition"), "Invalid measured scope")
    population = row.get("population_member")
    check(population is None or (row["method"] == "nevo" and
          type(population) is int and 0 <= population < 5),
          "Non-native population identity")
    check(row.get("device") == "cpu", "Only free CPU event accounting is allowed")
    check(valid_nonnegative(row.get("configured_iterations"), nullable=True),
          "Configured iterations invalid")


def _fields(row):
    return tuple(row[k] for k in
                 ("execution_id", "event_id", "method", "source_revision",
                  "stage_index", "task_identity", "operation",
                  "scope", "population_member", "configured_iterations", "device"))


def validate_journal(records, *, allow_training=False, expected_execution_id=None):
    """Validate full append log; return completed/open/interrupted without loss.

    Duplicate start or terminal records, reordered finish, invalid timers,
    and changing stage/source/identity for a single event all fail closed.
    """
    check(isinstance(records, list), "Records must be a list")
    starts, terminals = {}, {}
    for ordinal, row in enumerate(records):
        check(isinstance(row, dict), "Record must be object")
        _valid_identity(row)
        if expected_execution_id is not None:
            check(row["execution_id"] == expected_execution_id,
                  "Mismatched experiment in journal")
        ident = row["event_id"]
        phase = row["phase"]
        if phase == "start":
            check(ident not in starts and ident not in terminals,
                  "Duplicate event start or reused ID")
            check(valid_nonnegative(row.get("wall_started_ns")), "Missing start wall timestamp")
            check(valid_nonnegative(row.get("cpu_started_ns")), "Missing start CPU timestamp")
            check(set(row) == {
                "schema", "phase", "execution_id", "event_id", "method",
                "source_revision", "stage_index", "task_identity", "operation",
                "scope", "population_member", "configured_iterations", "device",
                "wall_started_ns", "cpu_started_ns"
            }, "Unexpected start record fields")
            starts[ident] = row
            continue
        check(ident in starts, "Terminal record without durable start")
        check(ident not in terminals, "Duplicate terminal event or double counting")
        check(_fields(starts[ident]) == _fields(row),
              "Terminal operation metadata differs from start")
        if phase == "complete":
            check(set(row) == {
                "schema", "phase", "execution_id", "event_id", "method",
                "source_revision", "stage_index", "task_identity", "operation",
                "scope", "population_member", "configured_iterations", "device",
                "actual_invocations", "parameter_updates", "batch_elements",
                "elapsed_wall_ns", "elapsed_cpu_ns", "peak_rss_kib"
            }, "Missing/extra completed measurement")
            for key in ("actual_invocations", "parameter_updates", "batch_elements",
                        "elapsed_wall_ns", "elapsed_cpu_ns", "peak_rss_kib"):
                check(valid_nonnegative(row.get(key), nullable=key in
                       ("parameter_updates", "batch_elements", "peak_rss_kib")),
                      "Missing/invalid actual measurement: "+key)
            check(row["actual_invocations"] > 0, "A completed event needs an actual invocation")
            if not allow_training:
                check(row["parameter_updates"] in (None, 0),
                      "Neural updates forbidden by current authorization")
            if row["operation"] in ("actor_optimizer", "critic_optimizer",
                                     "temperature_optimizer", "packnet_retrain",
                                     "nevo_distillation", "population_update"):
                check(row["parameter_updates"] is not None,
                      "Optimizer-like work requires explicit update count")
            check(row["elapsed_wall_ns"] >= 0 and row["elapsed_cpu_ns"] >= 0,
                  "Invalid clock measurement")
        else:
            check(set(row) == {
                "schema", "phase", "execution_id", "event_id", "method",
                "source_revision", "stage_index", "task_identity", "operation",
                "scope", "population_member", "configured_iterations", "device",
                "reason"
            }, "Interrupted operation cannot claim completed work")
            check(isinstance(row.get("reason"), str) and
                  0 < len(row["reason"]) <= 240, "Missing bounded interruption reason")
        terminals[ident] = row

    return {"complete": [terminals[i] for i in starts
                         if i in terminals and terminals[i]["phase"] == "complete"],
            "interrupted": [terminals[i] for i in starts
                            if i in terminals and terminals[i]["phase"] == "interrupted"],
            "open": [starts[i] for i in starts if i not in terminals],
            "start_count": len(starts), "terminal_count": len(terminals),
            "record_count": len(records)}


def aggregate(records, *, expected_execution_id=None, allow_training=False):
    status = validate_journal(records, allow_training=allow_training,
                              expected_execution_id=expected_execution_id)
    groups = {}
    # Count every *completed* event once. Never add configured-only or incomplete.
    for row in status["complete"]:
        key = (row["method"], row["stage_index"], row["task_identity"],
               row["operation"], row["scope"], row["population_member"])
        g = groups.setdefault(key, {"completed_events": 0, "actual_invocations": 0,
                                     "elapsed_wall_ns": 0, "elapsed_cpu_ns": 0,
                                     "parameter_updates": 0, "batch_elements": 0,
                                     "unknown_updates_events": 0,
                                     "unknown_batch_events": 0,
                                     "peak_rss_kib": None})
        g["completed_events"] += 1
        g["actual_invocations"] += row["actual_invocations"]
        g["elapsed_wall_ns"] += row["elapsed_wall_ns"]
        g["elapsed_cpu_ns"] += row["elapsed_cpu_ns"]
        if row["parameter_updates"] is None:
            g["unknown_updates_events"] += 1
        else:
            g["parameter_updates"] += row["parameter_updates"]
        if row["batch_elements"] is None:
            g["unknown_batch_events"] += 1
        else:
            g["batch_elements"] += row["batch_elements"]
        if row["peak_rss_kib"] is not None:
            g["peak_rss_kib"] = max(g["peak_rss_kib"] or 0, row["peak_rss_kib"])
    output = []
    # stable explicit sort does not compare None with ints
    for key in sorted(groups, key=lambda k: (k[0],k[1],k[3],k[4],-1 if k[5] is None else k[5])):
        g = groups[key]
        output.append({
            "method": key[0], "stage_index": key[1], "task_identity": key[2],
            "operation": key[3], "scope": key[4], "population_member": key[5],
            **g,
            "parameter_updates": None if g["unknown_updates_events"] else g["parameter_updates"],
            "batch_elements": None if g["unknown_batch_events"] else g["batch_elements"]
        })
    return {"schema": "D170-R2U/stage-aggregation-v1",
            "groups": output, "completed_events": len(status["complete"]),
            "open_event_ids": sorted(x["event_id"] for x in status["open"]),
            "interrupted_event_ids": sorted(x["event_id"] for x in status["interrupted"]),
            "unresolved_operations_are_unmeasured": True,
            "counting_policy": "completed-exactly-once; missing-never-zero"}


def read_jsonl(path):
    f = Path(path)
    if not f.exists():
        return []
    out = []
    with f.open("rb") as source:
        for index, line in enumerate(source, 1):
            check(line.endswith(b"\n"), "Truncated tail in durable journal at line "+str(index))
            check(0 < len(line) < 16384, "Oversized/empty journal line")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeError, ValueError) as error:
                raise JournalError("Unparseable journal line "+str(index)) from error
            out.append(value)
    return out


class EventJournal:
    """Single-writer durable JSONL. fsync each boundary before native callback.

    Recover with new EventJournal(execution_id=...) on the same path. The reader
    never infers completed work for a started event with no terminal record.
    """

    def __init__(self, path, execution_id, method, *,
                 wall_clock=time.perf_counter_ns, cpu_clock=time.process_time_ns):
        check(isinstance(execution_id, str) and TOKEN.fullmatch(execution_id),
              "Invalid execution ID")
        check(method in METHOD_SHA, "Unknown native method")
        self.path = Path(path)
        self.execution_id = execution_id
        self.method = method
        self.wall_clock = wall_clock
        self.cpu_clock = cpu_clock
        self._lock = threading.Lock()
        self.records = read_jsonl(self.path)
        validate_journal(self.records, expected_execution_id=execution_id)
        check(all(x["method"] == method for x in self.records),
              "Mixed competitor methods in one durable journal")
        self.sequence = len([r for r in self.records if r["phase"] == "start"])

    def _write(self, row):
        # Validate BEFORE touching disk; no failed write can count as a result.
        candidate = self.records + [row]
        validate_journal(candidate, expected_execution_id=self.execution_id)
        line = (json.dumps(row, separators=(",", ":"), sort_keys=True,
                           allow_nan=False) + "\n").encode()
        check(len(line) < 16384, "Event record exceeds append limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            check(os.write(fd, line) == len(line), "Partial journal write")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.records.append(row)

    def start(self, *, stage_index, operation, scope="whole",
              population_member=None, configured_iterations=None):
        with self._lock:
            check(type(stage_index) is int and 0 <= stage_index < len(CW20),
                  "Invalid original CW20 stage")
            self.sequence += 1
            event_id = self.execution_id + ":" + format(self.sequence, "08d")
            row = {
                "schema": SCHEMA, "phase": "start", "event_id": event_id,
                "execution_id": self.execution_id, "method": self.method,
                "source_revision": METHOD_SHA[self.method],
                "stage_index": stage_index, "task_identity": CW20[stage_index],
                "operation": operation, "scope": scope,
                "population_member": population_member,
                "configured_iterations": configured_iterations,
                "device": "cpu",
                "wall_started_ns": self.wall_clock(),
                "cpu_started_ns": self.cpu_clock()
            }
            self._write(row)
            return event_id

    def _base_terminal(self, event_id, phase):
        matches = [r for r in self.records if r["event_id"] == event_id]
        check(len(matches) == 1 and matches[0]["phase"] == "start",
              "Missing/open-only event required for completion")
        row = {k: v for k,v in matches[0].items()
               if k not in ("phase","wall_started_ns","cpu_started_ns")}
        row["phase"] = phase
        return row, matches[0]

    def complete(self, event_id, *, actual_invocations=1, parameter_updates=0,
                 batch_elements=None, peak_rss_kib=None):
        with self._lock:
            row, start = self._base_terminal(event_id, "complete")
            elapsed_wall = self.wall_clock() - start["wall_started_ns"]
            elapsed_cpu = self.cpu_clock() - start["cpu_started_ns"]
            check(elapsed_wall >= 0 and elapsed_cpu >= 0, "Nonmonotone runtime clock")
            row.update(actual_invocations=actual_invocations,
                       parameter_updates=parameter_updates,
                       batch_elements=batch_elements, elapsed_wall_ns=elapsed_wall,
                       elapsed_cpu_ns=elapsed_cpu, peak_rss_kib=peak_rss_kib)
            self._write(row)
            return row

    def interrupt(self, event_id, reason="operator interrupted before completion"):
        with self._lock:
            row,_ = self._base_terminal(event_id, "interrupted")
            check(isinstance(reason,str) and bool(reason), "Missing interruption reason")
            row["reason"] = reason[:240]
            self._write(row)
            return row

    def interrupted_recovery(self):
        """Mark crash-orphaned START events as interrupted with unknown work.

        Does NOT convert unknown results into successful, zero-cost operations.
        External controller MUST decide whether to resume with a new event ID.
        """
        pending = aggregate(self.records, expected_execution_id=self.execution_id)["open_event_ids"]
        for event_id in pending:
            self.interrupt(event_id, "recovered orphan start; actual work unknown")
        return pending

    @contextmanager
    def measure(self, *, stage_index, operation, scope="whole",
                population_member=None, configured_iterations=None,
                actual_invocations=1, parameter_updates=0, batch_elements=None):
        """Only use around a callback in an independently authorised driver.

        Synthetic tests may exercise it without importing competitors.
        """
        event_id=self.start(stage_index=stage_index, operation=operation,
                            scope=scope, population_member=population_member,
                            configured_iterations=configured_iterations)
        try:
            yield event_id
        except BaseException:
            self.interrupt(event_id, "exception while executing measured callback")
            raise
        else:
            self.complete(event_id, actual_invocations=actual_invocations,
                          parameter_updates=parameter_updates,
                          batch_elements=batch_elements)
