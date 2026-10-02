"""Streamlit dashboard for the PetriNet Lab toolkit."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import streamlit as st

from petrinet_lab.agent.tools import (
    MODELS,
    get_model,
    list_models,
    reachability_analysis,
    stochastic_analysis,
    structural_analysis,
    temporal_analysis,
    timed_simulation,
)
from petrinet_lab.core import pnml
from petrinet_lab.core.simulation import simulate_event_log, traces_from_events
from petrinet_lab.core.viz import draw_net


def _adhoc_reachability(net):  # pragma: no cover - dashboard helper
    from petrinet_lab.core.reachability import build_reachability_graph

    graph = build_reachability_graph(net)
    return {
        "states": len(graph.states),
        "bounded": graph.is_bounded(),
        "deadlocks": [graph.marking(i) for i in range(len(graph.states)) if not graph.successor_edges(i)],
        "dead_transitions": graph.dead_transitions(),
    }


def _adhoc_structural(net):  # pragma: no cover - dashboard helper
    from petrinet_lab.core.invariants import structural_properties

    return json.loads(json.dumps(structural_properties(net), default=str))


st.set_page_config(page_title="PetriNet Lab", page_icon="\u2b21", layout="wide")
st.title("\u2b21 PetriNet Lab")
st.caption("P/T, temporal and stochastic Petri nets with process-mining bridge and LLM agent")

with st.sidebar:
    st.header("Model")
    sample_names = [item["name"] for item in list_models()]
    choice = st.selectbox("Sample model", sample_names)
    uploaded = st.file_uploader("or upload a PNML file", type=["pnml", "xml"])

if uploaded is not None:
    net = pnml.from_pnml(uploaded.read().decode("utf-8"))
    model_name = uploaded.name
    metadata: dict = {}
else:
    net = get_model(choice)
    model_name = choice
    metadata = MODELS[choice]

overview, behaviour, timing, stochastic, mining, assistant = st.tabs(
    ["Structure", "Behaviour", "Time", "Stochastic", "Process mining", "Agent"]
)

with overview:
    left, right = st.columns([3, 2])
    with left:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        draw_net(net, ax=axis, title=model_name)
        st.pyplot(figure)
        plt.close(figure)
    with right:
        summary = net.summary()
        st.metric("Places", summary["places"])
        st.metric("Transitions", summary["transitions"])
        st.metric("Arcs", summary["arcs"])
        st.write("Workflow net:", "yes" if summary["workflow_net"] else "no")
        st.json(net.to_dict()["places"])

with behaviour:
    try:
        report = reachability_analysis(model_name) if not uploaded else _adhoc_reachability(net)
        st.json(report)
    except Exception as error:  # pragma: no cover - dashboard safeguard
        st.error(str(error))
    st.json(structural_analysis(model_name) if not uploaded else _adhoc_structural(net))

with timing:
    intervals = metadata.get("intervals")
    if intervals is None:
        st.info("This model has no timing metadata; showing the untimed structure.")
    else:
        st.write("Intervals (earliest, latest):", intervals)
        st.json(temporal_analysis(model_name))
        if st.button("Run timed simulation"):
            st.json(timed_simulation(model_name, steps=12, seed=1))

with stochastic:
    if "rates" not in metadata:
        st.info("This model has no stochastic rates configured.")
    else:
        st.write("Firing rates:", metadata["rates"])
        st.json(stochastic_analysis(model_name))

with mining:
    st.write("Simulate a log from the model and rediscover it with PM4Py.")
    cases = st.slider("Cases", 20, 500, 100, step=20)
    if st.button("Run discovery round-trip"):
        try:
            import pm4py

            from petrinet_lab.mining.conformance import log_fitness, token_replay
            from petrinet_lab.mining.discovery import discover, events_to_log

            events = simulate_event_log(net, n_cases=cases, seed=7)
            traces = traces_from_events(events)
            log, _ = events_to_log(events)
            result = discover(log, "inductive")
            fitness = log_fitness(token_replay(result.net, traces, final_marking=result.final_marking))
            st.json(result.stats)
            st.json(
                {
                    "discovered_places": len(result.net.places),
                    "discovered_transitions": len(result.net.transitions),
                    "token_replay_fitness": fitness,
                    "pm4py_fitness": pm4py.fitness_token_based_replay(
                        log,
                        result.pm4py_net,
                        result.pm4py_initial_marking,
                        result.pm4py_final_marking,
                    ),
                }
            )
        except Exception as error:  # pragma: no cover - dashboard safeguard
            st.error(f"Discovery failed: {error}")

with assistant:
    st.write("Ask the LangGraph agent about the models and their properties.")
    question = st.text_input("Question", "Is the parallel_workflow bounded and what are its deadlocks?")
    if st.button("Ask") and question:
        try:
            from petrinet_lab.agent.agent import ask

            with st.spinner("Thinking..."):
                answer = ask(question)
            st.markdown(answer)
        except Exception as error:  # pragma: no cover - dashboard safeguard
            st.error(f"Agent unavailable: {error}")
