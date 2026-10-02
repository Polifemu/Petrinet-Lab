"""Multi-agent orchestration demo (supervisor + 5 specialists) with LangGraph.

The topology is printed without any LLM. The orchestration runs against a cloud
provider or a local LM Studio/Ollama server when one is available.
"""

import json

from petrinet_lab.agent.multiagent import SPECIALISTS, graph_topology, run_multiagent

print("LangGraph topology (mermaid):")
print(graph_topology())
print("\nSpecialists and their tools:")
print(json.dumps(SPECIALISTS, indent=2))

QUESTION = (
    "Is the parallel_workflow bounded and deadlock-free? "
    "Also give the stationary probability of the machine being down."
)

try:
    answer = run_multiagent(QUESTION, max_rounds=6)
    print("\nMulti-agent answer:\n", answer)
except Exception as error:
    print("\nLLM not configured:", error)
    print("Local example: AGENT_PROVIDER=ollama LOCAL_MODEL=qwen3:4b-instruct-2507-q4_K_M \\")
    print("               python examples/06_multiagent_demo.py")
