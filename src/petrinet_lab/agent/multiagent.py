"""Multi-agent orchestration with LangGraph: supervisor + specialist workers.

A supervisor LLM routes the question to five specialist ReAct workers
(structure, behaviour, timing, stochastic, mining); a final node synthesises
the findings into one answer. Works with cloud providers or local
OpenAI-compatible servers (LM Studio/Ollama).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .tools import list_models as _list_models

KNOWN_MODELS = ", ".join(model["name"] for model in _list_models())

SPECIALISTS: dict[str, dict[str, Any]] = {
    "structure_agent": {
        "description": "model inventory, structure, workflow-net flag, P/T invariants",
        "tools": ["list_models", "describe_model", "structural_analysis"],
    },
    "behaviour_agent": {
        "description": "reachability graph, boundedness, deadlocks, dead transitions, liveness",
        "tools": ["reachability_analysis"],
    },
    "timing_agent": {
        "description": "Time Petri Nets: state classes, firing windows, time-locks, timed traces",
        "tools": ["temporal_analysis", "timed_simulation"],
    },
    "stochastic_agent": {
        "description": "Stochastic Petri Nets: stationary distribution, throughput, sojourn times",
        "tools": ["stochastic_analysis"],
    },
    "mining_agent": {
        "description": "log-level simulation of traces and variants for process mining",
        "tools": ["simulate_model"],
    },
}

SUPERVISOR_PROMPT = """You are the supervisor of a Petri net analysis team.
Known models: {models}

Available specialists:
{workers}

Routing rules:
- deadlock, boundedness, reachable states, liveness -> behaviour_agent
- structure, invariants, workflow net -> structure_agent
- timing, firing windows, time-lock -> timing_agent
- probability, stationary distribution, throughput, rates -> stochastic_agent
- traces, variants, event log -> mining_agent

Read the question and the findings reported so far, then reply with EXACTLY two lines:
AGENT: <specialist name, or FINISH when the question is covered>
TASK: <one focused sentence for that specialist, mentioning the exact model name>

Write the TASK as a self-contained question. Do not call a specialist whose topic is
already covered. If a specialist answered OUT_OF_SCOPE, do not call it again for that topic."""

WORKER_PROMPTS = {
    "structure_agent": (
        "You are the structure specialist. Use the tools to describe the model and "
        "compute its structural properties. Report concise numbers."
    ),
    "behaviour_agent": (
        "You are the behaviour specialist. Use reachability analysis to answer about "
        "states, boundedness, deadlocks, dead transitions and liveness. Remember that a "
        "marking with no enabled transitions is a deadlock marking, including the terminal "
        "marking of a workflow net."
    ),
    "timing_agent": (
        "You are the timing specialist for Time Petri Nets. Use temporal analysis and "
        "timed simulation; mention firing windows and time-locks explicitly."
    ),
    "stochastic_agent": (
        "You are the stochastic specialist. Use CTMC stationary distribution, throughput "
        "and sojourn times; report probabilities and rates."
    ),
    "mining_agent": (
        "You are the process-mining specialist. Simulate event traces and summarise the "
        "observed variants and activity counts."
    ),
}

WORKER_RULES = (
    "\nAlways call your tools before answering and never answer from memory. Use the exact "
    "model names. Answer only the part of the question that matches your scope and ignore "
    "the rest. If no part matches your scope, reply only with OUT_OF_SCOPE."
)

FINAL_PROMPT = (
    "You are the supervisor writing the final answer. Combine only the specialist findings "
    "above into a short, factual answer with concrete numbers. Do not invent metrics; if "
    "part of the question was not covered, say so explicitly. Use Petri net terminology "
    "precisely: a marking with no enabled transitions is a deadlock marking, even when it "
    "is the expected final marking of a workflow net."
)


class MultiAgentState(TypedDict):
    question: str
    task: str
    messages: Annotated[list, add_messages]
    next: str


_AGENT_LINE = re.compile(r"agent\s*[:=]\s*([a-z_]+)", re.IGNORECASE)
_TASK_LINE = re.compile(r"task\s*[:=]\s*(.+)", re.IGNORECASE)


def parse_route(text: str) -> str:
    tokens = re.findall(r"[a-z_]+", (text or "").lower())
    for token in tokens:
        if token in SPECIALISTS:
            return token
        if token == "finish":
            return "FINISH"
    return "FINISH"


def parse_decision(text: str) -> tuple[str, str | None]:
    agent_match = _AGENT_LINE.search(text or "")
    task_match = _TASK_LINE.search(text or "")
    if agent_match:
        token = agent_match.group(1).lower()
        if token in SPECIALISTS:
            route = token
        elif token.startswith("finish"):
            route = "FINISH"
        else:
            route = parse_route(text)
    else:
        route = parse_route(text)
    task = task_match.group(1).strip() if task_match and route != "FINISH" else None
    return route, task


def graph_topology() -> str:
    lines = ["graph TD", "    user((User)) --> supervisor"]
    for name in SPECIALISTS:
        label = name.replace("_agent", "")
        lines.append(f"    supervisor -->|{label}| {name}")
        lines.append(f"    {name} --> supervisor")
    lines.append("    supervisor -->|FINISH| final_answer")
    lines.append("    final_answer --> done((END))")
    return "\n".join(lines)


def _summaries(messages: list) -> list:
    clean = []
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            continue
        clean.append(message)
    return clean


def _worker_turns(messages: list) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage) and getattr(message, "name", None) in SPECIALISTS
    )


def build_multiagent(provider: str | None = None, llm: Any = None, max_rounds: int = 8):
    from .agent import _build_llm, build_react_agent, build_tools

    llm = llm or _build_llm(provider)
    registry = {tool.name: tool for tool in build_tools()}
    worker_agents = {
        name: build_react_agent(
            llm, [registry[tool_name] for tool_name in spec["tools"]], WORKER_PROMPTS[name]
        )
        for name, spec in SPECIALISTS.items()
    }

    def context_of(state: MultiAgentState) -> str:
        lines = [f"{message.name or message.type}: {message.content}" for message in _summaries(state["messages"])]
        return "\n\n".join(lines)

    def supervisor(state: MultiAgentState) -> dict:
        roster = "\n".join(f"- {name}: {spec['description']}" for name, spec in SPECIALISTS.items())
        decision = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT.format(workers=roster, models=KNOWN_MODELS)),
                HumanMessage(
                    content=f"Question: {state['question']}\n\nFindings so far:\n{context_of(state)}"
                ),
            ]
        )
        route, task = parse_decision(getattr(decision, "content", str(decision)))
        if _worker_turns(state["messages"]) >= max_rounds:
            route = "FINISH"
        return {"next": route, "task": task or state["question"]}

    def make_worker(name: str):
        agent = worker_agents[name]

        def worker(state: MultiAgentState) -> dict:
            task = state.get("task") or state["question"]
            result = agent.invoke(
                {
                    "messages": [
                        SystemMessage(
                            content=(
                                f"{WORKER_PROMPTS[name]}{WORKER_RULES}"
                                f"\n\nKnown models: {KNOWN_MODELS}"
                                f"\n\nTask: {task}"
                            )
                        ),
                        HumanMessage(content=task),
                    ]
                },
                config={"recursion_limit": 10},
            )
            summary = result["messages"][-1]
            content = summary.content if isinstance(summary.content, str) else str(summary.content)
            return {"messages": [AIMessage(content=f"[{name}] {content}", name=name)]}

        return worker

    def final_answer(state: MultiAgentState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content=FINAL_PROMPT),
                HumanMessage(
                    content=f"Question: {state['question']}\n\nSpecialist findings:\n{context_of(state)}"
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"messages": [AIMessage(content=content, name="final_answer")]}

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("final_answer", final_answer)
    for name in SPECIALISTS:
        graph.add_node(name, make_worker(name))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: "final_answer" if state["next"] == "FINISH" else state["next"],
        {**{name: name for name in SPECIALISTS}, "final_answer": "final_answer"},
    )
    for name in SPECIALISTS:
        graph.add_edge(name, "supervisor")
    graph.add_edge("final_answer", END)
    return graph.compile()


def run_multiagent(question: str, provider: str | None = None, max_rounds: int = 8) -> str:
    app = build_multiagent(provider=provider, max_rounds=max_rounds)
    result = app.invoke(
        {"question": question, "messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 60},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    if "--topology" in sys.argv:
        print(graph_topology())
        print("\nSpecialists:")
        for name, spec in SPECIALISTS.items():
            print(f"- {name}: {spec['description']} [{', '.join(spec['tools'])}]")
        raise SystemExit(0)
    if len(sys.argv) < 2:
        print('usage: python -m petrinet_lab.agent.multiagent "question" [--topology]')
        raise SystemExit(1)
    print(run_multiagent(" ".join(sys.argv[1:])))
