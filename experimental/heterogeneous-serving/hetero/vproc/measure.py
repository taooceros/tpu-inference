"""C7: measurement of block switches on the driver-corrected clock.

Every host kernel stamps its RPC results with its own ``time.time()`` (``t0``/``t1``)
and answers ``describe().clock_s``. The driver estimates each kernel's clock offset
from RPC midpoints (``estimate_offsets``: the round with the smallest round trip
wins, error bound = rtt / 2) and maps kernel stamps onto its own clock. A switch is
recorded as sequential phases (``SwitchRecorder``); each phase row carries

- ``duration_s``: driver bracket around the (concurrent) kernel calls,
- ``kernel_span_s``: earliest kernel ``t0`` to latest kernel ``t1`` after correction,
- ``rpc_overhead_s``: the difference,
- ``parts``: kernel-reported sub-segments (``load``, ``graph``, ``kv_alloc``,
  ``warmup``, ...), each the max over hosts (the SPMD critical path).

``segments`` is name -> duration over the phases that ran, so
``sum(segments) + unaccounted_s == total_s`` by construction.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Comparison targets (seconds): InPlaceEngine warm switch (results/ici/InPlaceEngine),
# and the O3 mechanism.
INPLACE_ENGINE_WARM_S = 100.6
O3_S = 27.5

PHASES = ("drain", "withdraw", "stop_export", "reshard", "build", "publish")
# in-engine layout switch (``BlockController.switch_layout``): drain, worker_switch, frontend_switch, publish
LAYOUT_SWITCH_PHASES = ("drain", "worker_switch", "frontend_switch", "publish")
SCHEMA = 1


# --------------------------------------------------------------------------- clocks

@dataclasses.dataclass(frozen=True)
class ClockOffset:
    """``offset_s`` = kernel clock - driver clock, measured at the RPC midpoint."""

    rank: int
    offset_s: float
    rtt_s: float
    samples: int


def _clock_s(description: Any) -> float:
    if isinstance(description, Mapping):
        return float(description["clock_s"])
    return float(description.clock_s)


def estimate_offsets(describe: Callable[[int], Any], ranks: Iterable[int], rounds: int = 5,
                     clock: Callable[[], float] = time.time) -> "ClockMap":
    """Offset of every kernel clock against the driver clock.

    ``describe(rank)`` runs one ``describe()`` RPC. Rounds are sequential per
    kernel so the RPC midpoint is not biased by queueing behind a sibling call.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    offsets: Dict[int, ClockOffset] = {}
    for rank in ranks:
        best: Optional[Tuple[float, float]] = None
        for _ in range(rounds):
            sent = clock()
            remote = _clock_s(describe(rank))
            received = clock()
            rtt = received - sent
            if best is None or rtt < best[0]:
                best = (rtt, remote - (sent + received) / 2.0)
        offsets[rank] = ClockOffset(rank, best[1], best[0], rounds)
    return ClockMap(offsets)


class ClockMap:
    def __init__(self, offsets: Optional[Mapping[int, ClockOffset]] = None):
        self.offsets: Dict[int, ClockOffset] = dict(offsets or {})

    def to_driver(self, rank: int, kernel_time_s: float) -> float:
        offset = self.offsets.get(rank)
        return kernel_time_s - (offset.offset_s if offset else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {str(r): {"offset_s": o.offset_s, "rtt_s": o.rtt_s, "samples": o.samples}
                for r, o in sorted(self.offsets.items())}

    def max_rtt_s(self) -> float:
        return max((o.rtt_s for o in self.offsets.values()), default=0.0)


# --------------------------------------------------------------------------- kernel results

def kernel_span(results: Mapping[int, Any], clocks: ClockMap) -> Optional[Tuple[float, float]]:
    """(earliest t0, latest t1) over the results that carry both, on the driver clock."""
    starts, ends = [], []
    for rank, result in results.items():
        if isinstance(result, Mapping) and _number(result.get("t0")) and _number(result.get("t1")):
            starts.append(clocks.to_driver(rank, float(result["t0"])))
            ends.append(clocks.to_driver(rank, float(result["t1"])))
    return (min(starts), max(ends)) if starts else None


def critical_parts(results: Mapping[int, Any]) -> Dict[str, float]:
    """Kernel-reported ``segments``: per name the max over hosts."""
    parts: Dict[str, float] = {}
    for result in results.values():
        if isinstance(result, Mapping) and isinstance(result.get("segments"), Mapping):
            for name, seconds in result["segments"].items():
                if _number(seconds):
                    parts[name] = max(parts.get(name, 0.0), float(seconds))
    return parts


def compile_count(result: Any) -> Optional[int]:
    """``compiles`` of one kernel report: an int, or a dict of counters (summed)."""
    if not isinstance(result, Mapping) or "compiles" not in result:
        return None
    value = result["compiles"]
    if isinstance(value, Mapping):
        return int(sum(v for v in value.values() if _number(v)))
    return int(value) if _number(value) else None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------------------------------------------- switch recording

class _Phase:
    def __init__(self, name: str, hosts: Sequence[int]):
        self.name, self.hosts = name, tuple(hosts)
        self.results: Dict[int, Any] = {}
        self.frontends: Dict[str, Any] = {}
        self.note: Dict[str, Any] = {}


class SwitchRecorder:
    """Sequential phases of one switch (or rebuild) on the driver clock."""

    def __init__(self, clocks: ClockMap, clock: Callable[[], float] = time.time):
        self.clocks, self.clock = clocks, clock
        self.rows: List[Dict[str, Any]] = []
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None

    def begin(self) -> None:
        self.t_start = self.clock()

    @contextlib.contextmanager
    def phase(self, name: str, hosts: Sequence[int] = ()):
        if self.t_start is None:
            self.begin()
        ph = _Phase(name, hosts)
        start = self.clock()
        error: Optional[BaseException] = None
        try:
            yield ph
        except BaseException as exc:
            error = exc
            raise
        finally:
            end = self.clock()
            self.rows.append(self._row(ph, start, end, error))

    def _row(self, ph: _Phase, start: float, end: float, error: Optional[BaseException]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": ph.name, "hosts": list(ph.hosts), "ok": error is None,
            "start_s": start - self.t_start, "end_s": end - self.t_start, "duration_s": end - start,
            "kernel_span_s": None, "rpc_overhead_s": None, "kernel_outside_bracket_s": None,
            "parts": critical_parts({**ph.results, **{("frontend", e): v for e, v in ph.frontends.items()}})}
        if error is not None:
            row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        span = kernel_span(ph.results, self.clocks)
        if span is not None:
            row["kernel_span_s"] = span[1] - span[0]
            row["rpc_overhead_s"] = (end - start) - (span[1] - span[0])
            row["kernel_outside_bracket_s"] = max(0.0, start - span[0], span[1] - end)
        counts = {r: compile_count(v) for r, v in ph.results.items()}
        counts = {r: c for r, c in counts.items() if c is not None}
        by_frontend = {e: compile_count(v) for e, v in ph.frontends.items()}
        by_frontend = {e: c for e, c in by_frontend.items() if c is not None}
        if counts:
            row["compiles_by_host"] = {str(r): c for r, c in sorted(counts.items())}
        if by_frontend:
            row["compiles_by_frontend"] = dict(sorted(by_frontend.items()))
        if counts or by_frontend:
            # SPMD kernels compile the same programs concurrently (max over hosts); each frontend's
            # own compiles come after the kernel build and are added.
            row["compiles"] = (max(counts.values()) if counts else 0) + sum(by_frontend.values())
        row.update(ph.note)
        return row

    def finish(self) -> Dict[str, Any]:
        """Segment table; ``compiles`` = sum over phases of the max over hosts."""
        self.t_end = self.clock()
        total = self.t_end - self.t_start
        segments: Dict[str, float] = {}
        for row in self.rows:
            segments[row["name"]] = segments.get(row["name"], 0.0) + row["duration_s"]
        return {"total_s": total, "segments": segments,
                "unaccounted_s": total - sum(segments.values()),
                "phases": self.rows,
                "compiles": sum(r.get("compiles", 0) for r in self.rows),
                "max_clock_uncertainty_s": self.clocks.max_rtt_s() / 2.0}


# --------------------------------------------------------------------------- tokens, HBM

def token_check(label: str, got: Any, reference: Any, max_listed: int = 5) -> Dict[str, Any]:
    """Exact equality of token outputs: ``[[int]]`` or ``{key: [int]}``."""
    got_rows, ref_rows = _rows(got), _rows(reference)
    mismatches: List[Dict[str, Any]] = []
    for key in sorted(set(got_rows) | set(ref_rows), key=str):
        a, b = got_rows.get(key), ref_rows.get(key)
        if a == b:
            continue
        first = None
        if a is not None and b is not None:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        mismatches.append({"key": key, "got_len": None if a is None else len(a),
                           "ref_len": None if b is None else len(b), "first_diff": first})
    return {"label": label, "equal": not mismatches, "rows": len(ref_rows),
            "mismatched": len(mismatches), "mismatches": mismatches[:max_listed]}


def _rows(value: Any) -> Dict[Any, List[int]]:
    items = value.items() if isinstance(value, Mapping) else enumerate(value)
    return {k: [int(t) for t in row] for k, row in items}


def device_memory_stats(devices: Optional[Sequence[Any]] = None) -> Optional[Dict[int, Dict[str, int]]]:
    """Per-device HBM counters of this process; None when no device has ``memory_stats()``
    (CPU). A host kernel puts this under ``health()["hbm"]``."""
    if devices is None:
        import jax
        devices = jax.local_devices()
    out: Dict[int, Dict[str, int]] = {}
    for device in devices:
        fn = getattr(device, "memory_stats", None)
        stats = fn() if callable(fn) else None
        if stats:
            out[int(device.id)] = {k: int(stats[k]) for k in
                                   ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in stats}
    return out or None


def hbm_summary(per_rank: Mapping[int, Optional[Mapping[Any, Mapping[str, int]]]]) -> Optional[Dict[str, Any]]:
    """Max bytes per chip over all hosts; None when no host reported HBM."""
    chips = [(rank, dev, dict(stats)) for rank, devs in per_rank.items() if devs
             for dev, stats in devs.items()]
    if not chips:
        return None
    return {"chips": len(chips),
            "max_bytes_in_use_per_chip": max(s.get("bytes_in_use", 0) for _, _, s in chips),
            "max_peak_bytes_in_use_per_chip": max(s.get("peak_bytes_in_use", 0) for _, _, s in chips),
            "bytes_limit_per_chip": max(s.get("bytes_limit", 0) for _, _, s in chips),
            "per_rank": {str(r): d for r, d in sorted(per_rank.items()) if d}}


def compare(total_s: float) -> Dict[str, Any]:
    return {"inplace_engine_warm_s": INPLACE_ENGINE_WARM_S, "o3_s": O3_S,
            "vs_inplace_engine": total_s / INPLACE_ENGINE_WARM_S, "vs_o3": total_s / O3_S,
            "faster_than_inplace_engine": total_s < INPLACE_ENGINE_WARM_S,
            "faster_than_o3": total_s < O3_S}


# --------------------------------------------------------------------------- report

class Measure:
    """Accumulates switch/rebuild reports and token checks for one controller."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.clocks = ClockMap()
        self.switches: List[Dict[str, Any]] = []
        self.rebuilds: List[Dict[str, Any]] = []
        self.deploys: List[Dict[str, Any]] = []
        self.token_checks: List[Dict[str, Any]] = []
        self.block: Dict[str, Any] = {}

    def recorder(self) -> SwitchRecorder:
        return SwitchRecorder(self.clocks, self.clock)

    def check_tokens(self, label: str, got: Any, reference: Any) -> Dict[str, Any]:
        entry = token_check(label, got, reference)
        self.token_checks.append(entry)
        return entry

    def add_switch(self, report: Dict[str, Any]) -> Dict[str, Any]:
        report = {**report, "comparison": compare(report["total_s"])}
        self.switches.append(report)
        return report

    def report(self) -> Dict[str, Any]:
        body = {"schema": SCHEMA, "created": self.clock(), "block": self.block,
                "constants": {"inplace_engine_warm_s": INPLACE_ENGINE_WARM_S, "o3_s": O3_S},
                "deploys": self.deploys, "switches": self.switches, "rebuilds": self.rebuilds,
                "token_checks": self.token_checks}
        body["verdict"] = verdict(body)
        return body

    def write(self, path: Any) -> Path:
        return write_report(path, self.report())


def verdict(report: Mapping[str, Any], expect_no_compile: bool = False) -> Dict[str, Any]:
    """U2-style verdict: ``failed`` lists every violated check."""
    failures: List[str] = []
    for check in report["token_checks"]:
        if not check["equal"]:
            failures.append(f"tokens {check['label']}: {check['mismatched']} of {check['rows']} rows differ")
    for index, switch in enumerate(report["switches"]):
        if not switch.get("ok", True):
            failures.append(f"switch {index} -> {switch.get('target')}: {switch.get('error')}")
        if expect_no_compile and switch.get("compiles"):
            failures.append(f"switch {index}: {switch['compiles']} compiles on the critical path")
        if abs(switch["unaccounted_s"]) > max(0.1, 0.05 * switch["total_s"]):
            failures.append(f"switch {index}: {switch['unaccounted_s']:.3f} s outside any segment")
    for rebuild in report["rebuilds"]:
        if not rebuild.get("ok", True):
            failures.append(f"rebuild after {rebuild.get('cause')}: {rebuild.get('error')}")
    status = "failed" if failures else "passed"
    return {"status": status, "failures": failures, "switches": len(report["switches"]),
            "rebuilds": len(report["rebuilds"]),
            "token_checks": len(report["token_checks"]),
            "totals_s": [s["total_s"] for s in report["switches"]]}


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_report(path: Any, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False, default=_json_default) + "\n")
    return path
