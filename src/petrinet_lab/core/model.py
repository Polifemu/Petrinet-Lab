"""Place/Transition (P/T) Petri net model with weighted arcs and standard semantics."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

Marking = dict[str, int]
ArcWeights = dict[str, int]


@dataclass(frozen=True)
class Place:
    id: str
    name: str = ""
    initial: int = 0

    def __post_init__(self) -> None:
        if self.initial < 0:
            raise ValueError(f"place {self.id!r} has a negative initial marking")


@dataclass(frozen=True)
class Transition:
    id: str
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or self.id


@dataclass(frozen=True)
class Arc:
    source: str
    target: str
    weight: int = 1

    def __post_init__(self) -> None:
        if self.weight < 1:
            raise ValueError(f"arc {self.source!r} -> {self.target!r} must have weight >= 1")

    def __str__(self) -> str:
        suffix = f" ({self.weight})" if self.weight != 1 else ""
        return f"{self.source} -> {self.target}{suffix}"


class PetriNet:
    """A P/T net: places, transitions, weighted arcs and firing semantics."""

    def __init__(self, name: str = "PetriNet") -> None:
        self.name = name
        self._places: dict[str, Place] = {}
        self._transitions: dict[str, Transition] = {}
        self._arcs: list[Arc] = []

    def add_place(self, place_id: str, name: str = "", initial: int = 0) -> Place:
        if place_id in self._places:
            raise ValueError(f"duplicate place id {place_id!r}")
        place = Place(place_id, name, initial)
        self._places[place_id] = place
        return place

    def add_transition(self, transition_id: str, name: str = "") -> Transition:
        if transition_id in self._transitions:
            raise ValueError(f"duplicate transition id {transition_id!r}")
        transition = Transition(transition_id, name)
        self._transitions[transition_id] = transition
        return transition

    def add_arc(self, source: str, target: str, weight: int = 1) -> Arc:
        source_is_place = source in self._places
        target_is_place = target in self._places
        source_is_transition = source in self._transitions
        target_is_transition = target in self._transitions
        if not (source_is_place or source_is_transition):
            raise KeyError(f"unknown arc source {source!r}")
        if not (target_is_place or target_is_transition):
            raise KeyError(f"unknown arc target {target!r}")
        if source_is_place == target_is_place:
            raise ValueError("arcs must connect a place to a transition or vice versa")
        arc = Arc(source, target, weight)
        self._arcs.append(arc)
        return arc

    @property
    def places(self) -> dict[str, Place]:
        return dict(self._places)

    @property
    def transitions(self) -> dict[str, Transition]:
        return dict(self._transitions)

    @property
    def arcs(self) -> list[Arc]:
        return list(self._arcs)

    @property
    def place_ids(self) -> list[str]:
        return list(self._places)

    @property
    def transition_ids(self) -> list[str]:
        return list(self._transitions)

    def preset(self, transition_id: str) -> ArcWeights:
        self._require_transition(transition_id)
        return self._collect_weights(lambda arc: arc.target == transition_id, lambda arc: arc.source)

    def postset(self, transition_id: str) -> ArcWeights:
        self._require_transition(transition_id)
        return self._collect_weights(lambda arc: arc.source == transition_id, lambda arc: arc.target)

    def input_transitions(self, place_id: str) -> ArcWeights:
        self._require_place(place_id)
        return self._collect_weights(lambda arc: arc.target == place_id, lambda arc: arc.source)

    def output_transitions(self, place_id: str) -> ArcWeights:
        self._require_place(place_id)
        return self._collect_weights(lambda arc: arc.source == place_id, lambda arc: arc.target)

    def _collect_weights(self, selector, key) -> ArcWeights:
        weights: ArcWeights = {}
        for arc in self._arcs:
            if selector(arc):
                node = key(arc)
                weights[node] = weights.get(node, 0) + arc.weight
        return weights

    def initial_marking(self) -> Marking:
        return {pid: place.initial for pid, place in self._places.items()}

    def marking_key(self, marking: Mapping[str, int]) -> tuple[int, ...]:
        return tuple(marking.get(pid, 0) for pid in self._places)

    def marking_from_key(self, key: Iterable[int]) -> Marking:
        return dict(zip(self._places, key, strict=False))

    def enabled(self, marking: Mapping[str, int], transition_id: str) -> bool:
        return all(marking.get(pid, 0) >= weight for pid, weight in self.preset(transition_id).items())

    def enabled_transitions(self, marking: Mapping[str, int]) -> list[str]:
        return [tid for tid in self._transitions if self.enabled(marking, tid)]

    def fire(self, marking: Mapping[str, int], transition_id: str) -> Marking:
        if not self.enabled(marking, transition_id):
            raise ValueError(f"transition {transition_id!r} is not enabled in the given marking")
        new_marking = dict(marking)
        for pid, weight in self.preset(transition_id).items():
            new_marking[pid] = new_marking.get(pid, 0) - weight
        for pid, weight in self.postset(transition_id).items():
            new_marking[pid] = new_marking.get(pid, 0) + weight
        return new_marking

    def try_fire(self, marking: Mapping[str, int], transition_id: str) -> Marking | None:
        return self.fire(marking, transition_id) if self.enabled(marking, transition_id) else None

    def incidence_matrix(self) -> list[list[int]]:
        place_index = {pid: i for i, pid in enumerate(self._places)}
        transition_ids = self.transition_ids
        matrix = [[0] * len(transition_ids) for _ in self._places]
        for j, tid in enumerate(transition_ids):
            for pid, weight in self.preset(tid).items():
                matrix[place_index[pid]][j] -= weight
            for pid, weight in self.postset(tid).items():
                matrix[place_index[pid]][j] += weight
        return matrix

    def reachable_from(self, start: str) -> set[str]:
        return self._traverse(start, forward=True)

    def co_reachable_from(self, start: str) -> set[str]:
        return self._traverse(start, forward=False)

    def is_workflow_net(self) -> bool:
        sources = [pid for pid in self._places if not self.input_transitions(pid)]
        sinks = [pid for pid in self._places if not self.output_transitions(pid)]
        if len(sources) != 1 or len(sinks) != 1:
            return False
        source, sink = sources[0], sinks[0]
        if self._places[source].initial != 1 or self._places[sink].initial != 0:
            return False
        downstream = self.reachable_from(source)
        upstream = self.co_reachable_from(sink)
        nodes = set(self._places) | set(self._transitions)
        return nodes <= (downstream & upstream)

    def strongly_connected(self) -> bool:
        nodes = set(self._nodes())
        if not nodes:
            return True
        start = next(iter(nodes))
        return nodes <= self.reachable_from(start) and nodes <= self.co_reachable_from(start)

    def validate(self) -> list[str]:
        issues: list[str] = []
        for pid in self._places:
            if not self.input_transitions(pid) and not self.output_transitions(pid):
                issues.append(f"place {pid!r} is isolated")
        for tid in self._transitions:
            if not self.preset(tid) and not self.postset(tid):
                issues.append(f"transition {tid!r} is isolated")
        return issues

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "places": len(self._places),
            "transitions": len(self._transitions),
            "arcs": len(self._arcs),
            "workflow_net": self.is_workflow_net(),
            "strongly_connected": self.strongly_connected(),
            "issues": self.validate(),
        }

    def copy(self) -> PetriNet:
        clone = PetriNet(self.name)
        for place in self._places.values():
            clone.add_place(place.id, place.name, place.initial)
        for transition in self._transitions.values():
            clone.add_transition(transition.id, transition.name)
        for arc in self._arcs:
            clone.add_arc(arc.source, arc.target, arc.weight)
        return clone

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "places": [{"id": p.id, "name": p.name, "initial": p.initial} for p in self._places.values()],
            "transitions": [{"id": t.id, "name": t.name} for t in self._transitions.values()],
            "arcs": [{"source": a.source, "target": a.target, "weight": a.weight} for a in self._arcs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PetriNet:
        net = cls(str(data.get("name", "PetriNet")))
        for place in data.get("places", []):  # type: ignore[union-attr]
            net.add_place(str(place["id"]), str(place.get("name", "")), int(place.get("initial", 0)))
        for transition in data.get("transitions", []):  # type: ignore[union-attr]
            net.add_transition(str(transition["id"]), str(transition.get("name", "")))
        for arc in data.get("arcs", []):  # type: ignore[union-attr]
            net.add_arc(str(arc["source"]), str(arc["target"]), int(arc.get("weight", 1)))
        return net

    def _nodes(self) -> list[str]:
        return list(self._places) + list(self._transitions)

    def _neighbors(self, node: str, forward: bool) -> list[str]:
        if forward:
            return [a.target for a in self._arcs if a.source == node]
        return [a.source for a in self._arcs if a.target == node]

    def _traverse(self, start: str, forward: bool) -> set[str]:
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in self._neighbors(node, forward):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return seen

    def _require_place(self, place_id: str) -> None:
        if place_id not in self._places:
            raise KeyError(f"unknown place {place_id!r}")

    def _require_transition(self, transition_id: str) -> None:
        if transition_id not in self._transitions:
            raise KeyError(f"unknown transition {transition_id!r}")

    def __repr__(self) -> str:
        return (
            f"PetriNet(name={self.name!r}, places={len(self._places)}, "
            f"transitions={len(self._transitions)}, arcs={len(self._arcs)})"
        )
