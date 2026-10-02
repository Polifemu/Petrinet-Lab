"""Token-tagged log simulation, used as the bridge towards process mining."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .model import Marking, PetriNet


@dataclass(frozen=True)
class SimulatedEvent:
    case_id: str
    activity: str
    timestamp: float


def simulate_event_log(
    net: PetriNet,
    n_cases: int = 100,
    seed: int | None = None,
    max_steps: int = 10_000,
    mean_step: float = 1.0,
    silent_label: str | None = None,
) -> list[SimulatedEvent]:
    """Simulate one case at a time, tagging tokens with a case id.

    Each case starts from the net initial marking, runs until a dead marking
    and records one event per transition firing except silent ones.
    """
    rng = random.Random(seed)
    sink_places = {pid for pid in net.place_ids if not net.output_transitions(pid)}
    events: list[SimulatedEvent] = []
    for case_index in range(n_cases):
        case_id = f"case_{case_index:06d}"
        marking: Marking = net.initial_marking()
        tokens: dict[str, list[str]] = {
            pid: [case_id] * count for pid, count in marking.items() if count
        }
        now = 0.0
        for _ in range(max_steps):
            if any(tokens.get(pid) for pid in sink_places):
                break
            enabled = net.enabled_transitions(marking)
            if not enabled:
                break
            transition_id = rng.choice(enabled)
            consumed_cases: list[str] = []
            for pid, weight in net.preset(transition_id).items():
                for _ in range(weight):
                    if tokens.get(pid):
                        consumed_cases.append(tokens[pid].pop())
                    marking[pid] = marking.get(pid, 0) - 1
            produced_case = consumed_cases[0] if consumed_cases else case_id
            for pid, weight in net.postset(transition_id).items():
                tokens.setdefault(pid, [])
                for _ in range(weight):
                    tokens[pid].append(produced_case)
                marking[pid] = marking.get(pid, 0) + weight
            now += rng.expovariate(1.0 / mean_step)
            label = net.transitions[transition_id].label
            if label and label != silent_label:
                events.append(SimulatedEvent(case_id, label, now))
    return events


def events_to_dataframe(events: list[SimulatedEvent]):
    import pandas as pd

    return pd.DataFrame(
        {
            "case:concept:name": [event.case_id for event in events],
            "concept:name": [event.activity for event in events],
            "time:timestamp": pd.to_datetime([event.timestamp for event in events], unit="s"),
        }
    )


def traces_from_events(events: list[SimulatedEvent]) -> list[list[str]]:
    grouped: dict[str, list[tuple[float, str]]] = {}
    for event in events:
        grouped.setdefault(event.case_id, []).append((event.timestamp, event.activity))
    return [[activity for _, activity in sorted(items)] for items in grouped.values()]
