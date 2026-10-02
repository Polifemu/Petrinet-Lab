"""Demonstrate the agent tool layer with and without an LLM provider.

Without any API key the example runs the tools directly, which is exactly what
the LangGraph agent calls through function calling.
"""

import json

from petrinet_lab.agent.tools import dispatch

QUESTIONS = [
    ("list_models", {}),
    ("describe_model", {"name": "parallel_workflow"}),
    ("reachability_analysis", {"name": "parallel_workflow"}),
    ("temporal_analysis", {"name": "timelock"}),
    ("stochastic_analysis", {"name": "machine"}),
]

for tool, arguments in QUESTIONS:
    print(f"\n>>> {tool}({arguments})")
    print(json.dumps(dispatch(tool, arguments), indent=2, default=str)[:900])

try:
    from petrinet_lab.agent.agent import ask

    answer = ask("Is the parallel_workflow bounded? Which transitions are dead?")
    print("\nAgent answer:\n", answer)
except Exception as error:
    print("\nLLM agent not configured:", error)
    print("Set GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY to enable it.")
