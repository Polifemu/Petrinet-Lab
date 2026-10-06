import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from petrinet_lab.agent.multiagent import (
    SPECIALISTS,
    _summaries,
    _worker_turns,
    graph_topology,
    parse_decision,
    parse_route,
)
from petrinet_lab.agent.tools import TOOL_SPECS


def test_every_tool_belongs_to_exactly_one_specialist():
    assigned = [tool for spec in SPECIALISTS.values() for tool in spec["tools"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {spec["name"] for spec in TOOL_SPECS}


def test_specialists_have_descriptions():
    for name, spec in SPECIALISTS.items():
        assert name.endswith("_agent")
        assert spec["description"]
        assert spec["tools"]


def test_parse_route():
    assert parse_route("structure_agent") == "structure_agent"
    assert parse_route("  Timing_Agent\n") == "timing_agent"
    assert parse_route("I would call the stochastic_agent next.") == "stochastic_agent"
    assert parse_route("FINISH") == "FINISH"
    assert parse_route("we can finish now") == "FINISH"
    assert parse_route("banana") == "FINISH"
    assert parse_route("") == "FINISH"


def test_parse_decision_with_task_decomposition():
    route, task = parse_decision(
        "AGENT: timing_agent\nTASK: When can start fire in cyclic_cell?"
    )
    assert route == "timing_agent"
    assert "cyclic_cell" in task
    route, task = parse_decision("AGENT: FINISH\nTASK: done")
    assert route == "FINISH"
    assert task is None
    route, task = parse_decision("stochastic_agent")
    assert route == "stochastic_agent"
    assert task is None
    route, task = parse_decision("I pick behaviour_agent because...")
    assert route == "behaviour_agent"
    assert task is None


def test_graph_topology_mentions_all_nodes():
    topology = graph_topology()
    assert topology.startswith("graph TD")
    assert "supervisor" in topology
    for name in SPECIALISTS:
        assert name in topology
    assert "final_answer" in topology


def test_message_filtering_and_turn_count():
    tool_call = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    messages = [
        HumanMessage(content="question"),
        tool_call,
        ToolMessage(content="tool output", tool_call_id="1"),
        AIMessage(content="[structure_agent] answer", name="structure_agent"),
        AIMessage(content="[timing_agent] answer", name="timing_agent"),
    ]
    summaries = _summaries(messages)
    assert len(summaries) == 3
    assert _worker_turns(messages) == 2
