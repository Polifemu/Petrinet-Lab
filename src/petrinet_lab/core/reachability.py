"""Reachability graph construction and classical behavioural properties."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .model import Marking, PetriNet


class StateSpaceTooLarge(RuntimeError):
    """Raised when the reachability graph exceeds the configured state limit."""


class UnboundedNetError(RuntimeError):
    """Raised when a marking exceeds the configured token cap."""


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    transition: str


@dataclass
class ReachabilityGraph:
    net: PetriNet
    states: list[tuple[int, ...]] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    truncated: bool = False

    def __post_init__(self) -> None:
        self._index: dict[tuple[int, ...], int] = {key: i for i, key in enumerate(self.states)}
        self._out: dict[int, list[Edge]] = {}
        self._in: dict[int, list[Edge]] = {}
        for edge in self.edges:
            self._out.setdefault(edge.source, []).append(edge)
            self._in.setdefault(edge.target, []).append(edge)

    @property
    def index(self) -> dict[tuple[int, ...], int]:
        return dict(self._index)

    def marking(self, state: int) -> Marking:
        return self.net.marking_from_key(self.states[state])

    def successor_edges(self, state: int) -> list[Edge]:
        return list(self._out.get(state, []))

    def predecessor_edges(self, state: int) -> list[Edge]:
        return list(self._in.get(state, []))

    def token_bounds(self) -> dict[str, int]:
        if not self.states:
            return {pid: 0 for pid in self.net.place_ids}
        return {
            pid: max(key[i] for key in self.states)
            for i, pid in enumerate(self.net.place_ids)
        }

    def is_bounded(self) -> bool:
        return not self.truncated

    def deadlock_markings(self) -> list[Marking]:
        return [
            self.marking(i)
            for i in range(len(self.states))
            if not self._out.get(i)
        ]

    def dead_transitions(self) -> list[str]:
        fired = {edge.transition for edge in self.edges}
        return [tid for tid in self.net.transition_ids if tid not in fired]

    def occurrence_count(self, transition_id: str, state: int | None = None) -> int:
        edges = self.edges if state is None else self._out.get(state, [])
        return sum(1 for edge in edges if edge.transition == transition_id)

    def on_cycle(self, transition_id: str) -> bool:
        component, cyclic_components = self._cycle_info()
        return any(
            edge.transition == transition_id
            and component.get(edge.source) is not None
            and component[edge.source] == component.get(edge.target)
            and component[edge.source] in cyclic_components
            for edge in self.edges
        )

    def liveness_level(self, transition_id: str) -> int:
        """Classical liveness levels: 0 dead, 1 can fire, 2 can fire on a path, 3 can fire infinitely often, 4 live."""
        if self.truncated:
            raise StateSpaceTooLarge("liveness requires the full reachability graph")
        if transition_id in self.dead_transitions():
            return 0
        reachable_states = len(self.states)
        can_always_eventually_fire = True
        for state in range(reachable_states):
            if not self._can_reach_firing(state, transition_id):
                can_always_eventually_fire = False
                break
        if can_always_eventually_fire:
            return 4
        if self.on_cycle(transition_id):
            return 3
        reachable_from_fired = any(
            edge.transition == transition_id for edge in self.edges
        )
        return 2 if reachable_from_fired else 1

    def to_networkx(self):
        import networkx as nx

        graph = nx.MultiDiGraph()
        for i, key in enumerate(self.states):
            graph.add_node(i, marking=self.net.marking_from_key(key))
        for edge in self.edges:
            graph.add_edge(edge.source, edge.target, transition=edge.transition)
        return graph

    def _can_reach_firing(self, start: int, transition_id: str) -> bool:
        seen = {start}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for edge in self._out.get(state, []):
                if edge.transition == transition_id:
                    return True
                if edge.target not in seen:
                    seen.add(edge.target)
                    queue.append(edge.target)
        return False

    def _cycle_info(self) -> tuple[dict[int, int], set[int]]:
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(self.states)))
        graph.add_edges_from((edge.source, edge.target) for edge in self.edges)
        component: dict[int, int] = {}
        cyclic: set[int] = set()
        for index, nodes in enumerate(nx.strongly_connected_components(graph)):
            for node in nodes:
                component[node] = index
            if len(nodes) > 1 or any(graph.has_edge(node, node) for node in nodes):
                cyclic.add(index)
        return component, cyclic


def build_reachability_graph(
    net: PetriNet,
    max_states: int = 200_000,
    token_cap: int = 10_000,
) -> ReachabilityGraph:
    initial = net.marking_key(net.initial_marking())
    states: list[tuple[int, ...]] = [initial]
    edges: list[Edge] = []
    seen = {initial: 0}
    queue = deque([initial])
    while queue:
        key = queue.popleft()
        source = seen[key]
        marking = net.marking_from_key(key)
        for transition_id in net.enabled_transitions(marking):
            successor = net.fire(marking, transition_id)
            if any(value > token_cap for value in successor.values()):
                raise UnboundedNetError(
                    f"marking exceeded token cap {token_cap} after firing {transition_id!r}; the net is unbounded"
                )
            successor_key = net.marking_key(successor)
            target = seen.get(successor_key)
            if target is None:
                if len(states) >= max_states:
                    raise StateSpaceTooLarge(
                        f"reachability graph exceeded max_states={max_states}"
                    )
                target = len(states)
                seen[successor_key] = target
                states.append(successor_key)
                queue.append(successor_key)
            edges.append(Edge(source, target, transition_id))
    return ReachabilityGraph(net, states, edges)
