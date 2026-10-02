"""Process discovery bridge: PM4Py models in, core Petri nets out."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import pnml
from ..core.model import PetriNet
from ..core.simulation import SimulatedEvent, events_to_dataframe

_ALGORITHMS = {
    "inductive": "discover_petri_net_inductive",
    "alpha": "discover_petri_net_alpha",
    "heuristics": "discover_petri_net_heuristics",
    "ilp": "discover_petri_net_ilp",
}


@dataclass
class DiscoveryResult:
    algorithm: str
    net: PetriNet
    initial_marking: dict[str, int]
    final_marking: dict[str, int]
    pm4py_net: Any = field(repr=False, default=None)
    pm4py_initial_marking: Any = field(repr=False, default=None)
    pm4py_final_marking: Any = field(repr=False, default=None)
    stats: dict[str, Any] = field(default_factory=dict)


def events_to_log(events: Iterable[SimulatedEvent]):
    import pm4py

    dataframe = events_to_dataframe(list(events))
    dataframe = pm4py.format_dataframe(
        dataframe,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    return pm4py.convert_to_event_log(dataframe), dataframe


def read_log(path: str | Path):
    import pm4py

    path = Path(path)
    if path.suffix.lower() in {".xes", ".gz"} or path.name.lower().endswith(".xes.gz"):
        return pm4py.read_xes(str(path))
    import pandas as pd

    dataframe = pd.read_csv(path)
    return pm4py.convert_to_event_log(dataframe)


def log_stats(log) -> dict[str, Any]:
    import pm4py

    dataframe = pm4py.convert_to_dataframe(log)
    case_key = "case:concept:name"
    activity_key = "concept:name"
    cases = dataframe[case_key].nunique()
    events = len(dataframe)
    activities = sorted(dataframe[activity_key].dropna().unique().tolist())
    variants = (
        dataframe.sort_values("time:timestamp")
        .groupby(case_key)[activity_key]
        .apply(lambda trace: " → ".join(map(str, trace)))
    )
    variant_counts = variants.value_counts()
    return {
        "cases": int(cases),
        "events": int(events),
        "activities": activities,
        "variants": int(variant_counts.shape[0]),
        "top_variants": [
            {"variant": variant, "count": int(count)}
            for variant, count in variant_counts.head(5).items()
        ],
    }


def discover(log, algorithm: str = "inductive", **kwargs) -> DiscoveryResult:
    import pm4py

    if algorithm not in _ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {sorted(_ALGORITHMS)}")
    function = getattr(pm4py, _ALGORITHMS[algorithm])
    pm4py_net, initial_marking, final_marking = function(log, **kwargs)
    lab_net = to_lab_net(pm4py_net, initial_marking)
    result = DiscoveryResult(
        algorithm=algorithm,
        net=lab_net,
        initial_marking=_marking_to_dict(initial_marking),
        final_marking=_marking_to_dict(final_marking),
        pm4py_net=pm4py_net,
        pm4py_initial_marking=initial_marking,
        pm4py_final_marking=final_marking,
    )
    result.stats = log_stats(log)
    return result


def to_lab_net(pm4py_net, initial_marking) -> PetriNet:
    import pm4py
    from pm4py.objects.petri_net.obj import Marking

    handle, path = tempfile.mkstemp(suffix=".pnml")
    os.close(handle)
    try:
        pm4py.write_pnml(pm4py_net, initial_marking, Marking(), path)
        return pnml.load_pnml(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _marking_to_dict(marking) -> dict[str, int]:
    return {str(place.name if hasattr(place, "name") else place): int(count) for place, count in marking.items()}
