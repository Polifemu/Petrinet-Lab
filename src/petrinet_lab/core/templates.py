"""Ready-made stochastic Petri net models with closed-form references.

Each builder returns a :class:`~petrinet_lab.core.gspn.StochasticPetriNet`
ready for ``build_gspn_ctmc`` or ``simulate_gspn_ssa``. The companion
``*_stationary`` functions return the analytic distributions used to validate
the constructions (see ``tests/test_templates.py``).
"""

from __future__ import annotations

import math

import numpy as np

from .gspn import StochasticPetriNet


def mm1k(
    capacity: int = 3,
    arrival_rate: float = 1.0,
    service_rate: float = 2.0,
    name: str | None = None,
) -> StochasticPetriNet:
    """M/M/1/K queue: one server, exponential arrivals and services, K slots."""
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    gspn = StochasticPetriNet(name or f"mm1k_K{capacity}")
    gspn.add_place("slots", initial=capacity)
    gspn.add_place("jobs")
    gspn.add_timed_transition("arrive", arrival_rate)
    gspn.add_timed_transition("serve", service_rate)
    gspn.add_arc("slots", "arrive")
    gspn.add_arc("arrive", "jobs")
    gspn.add_arc("jobs", "serve")
    gspn.add_arc("serve", "slots")
    return gspn


def mm1k_stationary(capacity: int, arrival_rate: float, service_rate: float) -> np.ndarray:
    """Stationary distribution of the number of jobs in an M/M/1/K queue."""
    rho = arrival_rate / service_rate
    if abs(rho - 1.0) < 1e-12:
        return np.full(capacity + 1, 1.0 / (capacity + 1))
    weights = np.array([rho**n for n in range(capacity + 1)])
    return weights / weights.sum()


def mmc(
    servers: int = 2,
    buffer: int = 0,
    arrival_rate: float = 1.0,
    service_rate: float = 2.0,
    name: str | None = None,
) -> StochasticPetriNet:
    """M/M/c/K queue (K = c + buffer) as a GSPN.

    A job takes a free server with an immediate ``start`` transition, so
    tangible markings always have either no queue or no idle server.
    """
    if servers < 1:
        raise ValueError("servers must be >= 1")
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    capacity = servers + buffer
    gspn = StochasticPetriNet(name or f"mmc_c{servers}_K{capacity}")
    gspn.add_place("free_servers", initial=servers)
    gspn.add_place("waiting")
    gspn.add_place("in_service")
    gspn.add_place("capacity", initial=capacity)
    gspn.add_timed_transition("arrive", arrival_rate)
    gspn.add_immediate_transition("start")
    gspn.add_timed_transition("finish", lambda marking: service_rate * marking["in_service"])
    gspn.add_arc("capacity", "arrive")
    gspn.add_arc("arrive", "waiting")
    gspn.add_arc("waiting", "start")
    gspn.add_arc("free_servers", "start")
    gspn.add_arc("start", "in_service")
    gspn.add_arc("in_service", "finish")
    gspn.add_arc("finish", "free_servers")
    gspn.add_arc("finish", "capacity")
    return gspn


def mmc_stationary(
    servers: int,
    buffer: int,
    arrival_rate: float,
    service_rate: float,
) -> np.ndarray:
    """Stationary distribution of the number of jobs in an M/M/c/K queue."""
    if servers < 1 or buffer < 0:
        raise ValueError("servers must be >= 1 and buffer >= 0")
    capacity = servers + buffer
    offered = arrival_rate / service_rate
    weights = []
    for jobs in range(capacity + 1):
        if jobs <= servers:
            weights.append(offered**jobs / math.factorial(jobs))
        else:
            weights.append(offered**jobs / (math.factorial(servers) * servers ** (jobs - servers)))
    array = np.array(weights)
    return array / array.sum()


def machine(
    break_rate: float = 1.0,
    repair_rate: float = 3.0,
    name: str = "machine",
) -> StochasticPetriNet:
    """Single unit alternating between ``up`` and ``down`` states."""
    if break_rate < 0 or repair_rate < 0:
        raise ValueError("rates must be non-negative")
    gspn = StochasticPetriNet(name)
    gspn.add_place("up", initial=1)
    gspn.add_place("down")
    gspn.add_timed_transition("break", break_rate)
    gspn.add_timed_transition("repair", repair_rate)
    gspn.add_arc("up", "break")
    gspn.add_arc("break", "down")
    gspn.add_arc("down", "repair")
    gspn.add_arc("repair", "up")
    return gspn


def producer_consumer(
    buffer: int = 3,
    production_rate: float = 1.0,
    consumption_rate: float = 2.0,
    name: str | None = None,
) -> StochasticPetriNet:
    """Bounded buffer: a producer fills ``buffer`` slots, a consumer drains them."""
    if buffer < 1:
        raise ValueError("buffer must be >= 1")
    gspn = StochasticPetriNet(name or f"producer_consumer_B{buffer}")
    gspn.add_place("empty", initial=buffer)
    gspn.add_place("full")
    gspn.add_timed_transition("produce", production_rate)
    gspn.add_timed_transition("consume", consumption_rate)
    gspn.add_arc("empty", "produce")
    gspn.add_arc("produce", "full")
    gspn.add_arc("full", "consume")
    gspn.add_arc("consume", "empty")
    return gspn


def retry(
    ok_weight: float = 3.0,
    rework_weight: float = 1.0,
    service_rate: float = 1.0,
    name: str = "retry",
) -> StochasticPetriNet:
    """A task served once per attempt, then accepted or sent back for rework.

    The first attempt always happens; each attempt succeeds with probability
    ``ok_weight / (ok_weight + rework_weight)``, so the expected number of
    attempts is ``(ok_weight + rework_weight) / ok_weight``.
    """
    if ok_weight <= 0 or rework_weight <= 0:
        raise ValueError("weights must be positive")
    gspn = StochasticPetriNet(name)
    gspn.add_place("ready", initial=1)
    gspn.add_place("busy")
    gspn.add_place("done")
    gspn.add_timed_transition("work", service_rate)
    gspn.add_immediate_transition("ok", weight=ok_weight)
    gspn.add_immediate_transition("rework", weight=rework_weight)
    gspn.add_arc("ready", "work")
    gspn.add_arc("work", "busy")
    gspn.add_arc("busy", "ok")
    gspn.add_arc("ok", "done")
    gspn.add_arc("busy", "rework")
    gspn.add_arc("rework", "ready")
    return gspn


def expected_attempts(ok_weight: float, rework_weight: float) -> float:
    """Expected number of service attempts in the :func:`retry` template."""
    if ok_weight <= 0 or rework_weight < 0:
        raise ValueError("weights must be positive")
    return (ok_weight + rework_weight) / ok_weight
