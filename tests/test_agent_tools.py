import json

import pytest

from petrinet_lab.agent.tools import TOOL_SPECS, dispatch, list_models


def test_tool_specs_are_well_formed():
    names = {spec["name"] for spec in TOOL_SPECS}
    assert names == {
        "list_models",
        "describe_model",
        "structural_analysis",
        "reachability_analysis",
        "temporal_analysis",
        "stochastic_analysis",
        "simulate_model",
        "timed_simulation",
    }
    for spec in TOOL_SPECS:
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"


def test_list_and_describe_models():
    models = list_models()
    assert {model["name"] for model in models} >= {"parallel_workflow", "machine"}
    description = dispatch("describe_model", {"name": "parallel_workflow"})
    json.dumps(description, default=str)
    assert description["workflow_net"] is True
    assert description["places"] == 6


def test_reachability_tool_detects_boundedness():
    report = dispatch("reachability_analysis", {"name": "parallel_workflow"})
    json.dumps(report, default=str)
    assert report["bounded"] is True
    assert len(report["deadlocks"]) == 1
    assert report["deadlocks"][0]["end"] == 1
    assert report["dead_transitions"] == []


def test_temporal_tool_reports_time_lock():
    report = dispatch("temporal_analysis", {"name": "timelock"})
    json.dumps(report, default=str)
    assert report["firing_time_bounds"]["slow"] is None
    assert report["firing_time_bounds"]["fast"] == [1, 2]


def test_stochastic_tool_stationary():
    report = dispatch("stochastic_analysis", {"name": "machine"})
    json.dumps(report, default=str)
    probabilities = {tuple(item["marking"].values()): item["probability"] for item in report["stationary"]}
    assert probabilities[(1, 0)] == pytest.approx(0.75)
    assert probabilities[(0, 1)] == pytest.approx(0.25)


def test_simulation_tools():
    report = dispatch("simulate_model", {"name": "parallel_workflow", "n_cases": 5, "seed": 1})
    json.dumps(report, default=str)
    variants = {tuple(variant) for variant in report["variants"]}
    assert variants == {("Split", "Task A", "Task B", "Join"), ("Split", "Task B", "Task A", "Join")}
    timed = dispatch("timed_simulation", {"name": "cyclic_cell", "steps": 3, "seed": 1})
    json.dumps(timed, default=str)
    assert [event["transition"] for event in timed["events"][:2]] == ["start", "finish"]


def test_unknown_tool_and_model_are_recoverable():
    unknown_tool = dispatch("nope", {})
    assert "error" in unknown_tool
    assert "list_models" in unknown_tool["available_tools"]

    unknown_model = dispatch("describe_model", {"name": "machine_model"})
    assert "error" in unknown_model
    assert {"machine", "parallel_workflow"} <= set(unknown_model["available_models"])

    missing_argument = dispatch("stochastic_analysis", {})
    assert "error" in missing_argument
