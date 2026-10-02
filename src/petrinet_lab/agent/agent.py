"""LangGraph ReAct agent over the Petri net analysis tools.

Providers are selected by environment variables: GOOGLE_API_KEY, OPENAI_API_KEY,
ANTHROPIC_API_KEY or a local OpenAI-compatible server (LM Studio, Ollama).
AGENT_MODEL / LOCAL_MODEL override the default chat model name.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.parse import urlparse

SYSTEM_PROMPT = (
    "You are a Petri net analysis assistant. Use the available tools to inspect "
    "models, compute structural, reachability, temporal and stochastic "
    "properties, and to simulate executions. Always answer with concrete "
    "numbers returned by the tools; never invent metrics."
)

LOCAL_ENDPOINTS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
}
LOCAL_FALLBACK_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def _probe(url: str, timeout: float = 0.3) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def available_local_models(base_url: str, timeout: float = 2.0) -> list[str]:
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return []
    return [str(item["id"]) for item in payload.get("data", []) if item.get("id")]


def pick_local_model(models: list[str]) -> str | None:
    usable = [model for model in models if "embed" not in model.lower()]
    if not usable:
        return None
    for keyword in ("30b-a3b", "instruct", "27b", "35b", "14b", "8b", "4b"):
        for model in usable:
            if keyword in model.lower():
                return model
    return usable[0]


def detect_local_endpoint() -> tuple[str, str] | None:
    for url in LOCAL_ENDPOINTS.values():
        if _probe(url):
            local_model = os.environ.get("LOCAL_MODEL")
            if not local_model:
                local_model = pick_local_model(available_local_models(url)) or LOCAL_FALLBACK_MODEL
            return url, local_model
    return None


def build_tools() -> list[Any]:
    from langchain_core.tools import tool

    @tool
    def list_models() -> str:
        """List available Petri net models with their descriptions."""
        from .tools import dispatch

        return json.dumps(dispatch("list_models", {}), default=str)

    @tool
    def describe_model(name: str) -> str:
        """Show places, transitions, arcs and structural flags of a model."""
        from .tools import dispatch

        return json.dumps(dispatch("describe_model", {"name": name}), default=str)

    @tool
    def structural_analysis(name: str) -> str:
        """Compute P/T invariants and conservative/repetitive properties."""
        from .tools import dispatch

        return json.dumps(dispatch("structural_analysis", {"name": name}), default=str)

    @tool
    def reachability_analysis(name: str) -> str:
        """Reachability graph: states, boundedness, deadlocks, dead transitions, liveness."""
        from .tools import dispatch

        return json.dumps(dispatch("reachability_analysis", {"name": name}), default=str)

    @tool
    def temporal_analysis(name: str) -> str:
        """Time Petri Net state classes, firing-time bounds and time-locks."""
        from .tools import dispatch

        return json.dumps(dispatch("temporal_analysis", {"name": name}), default=str)

    @tool
    def stochastic_analysis(name: str) -> str:
        """Stochastic Petri Net stationary distribution, throughput and sojourn times."""
        from .tools import dispatch

        return json.dumps(dispatch("stochastic_analysis", {"name": name}), default=str)

    @tool
    def simulate_model(name: str, n_cases: int = 20, seed: int = 0) -> str:
        """Simulate token-tagged event traces from a model."""
        from .tools import dispatch

        return json.dumps(
            dispatch("simulate_model", {"name": name, "n_cases": n_cases, "seed": seed}),
            default=str,
        )

    @tool
    def timed_simulation(name: str, steps: int = 10, seed: int = 0) -> str:
        """Simulate a timed execution trace of a Time Petri Net."""
        from .tools import dispatch

        return json.dumps(
            dispatch("timed_simulation", {"name": name, "steps": steps, "seed": seed}),
            default=str,
        )

    return [
        list_models,
        describe_model,
        structural_analysis,
        reachability_analysis,
        temporal_analysis,
        stochastic_analysis,
        simulate_model,
        timed_simulation,
    ]


def _build_llm(provider: str | None = None):
    provider = provider or os.environ.get("AGENT_PROVIDER", "").lower() or None
    if provider is None:
        if os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("LOCAL_BASE_URL") or detect_local_endpoint():
            provider = "local"
    if provider in {"local", "lmstudio", "ollama"}:
        from langchain_openai import ChatOpenAI

        base_url = os.environ.get("LOCAL_BASE_URL")
        model = os.environ.get("LOCAL_MODEL")
        if provider in LOCAL_ENDPOINTS and not base_url:
            base_url = LOCAL_ENDPOINTS[provider]
        if not base_url:
            detected = detect_local_endpoint()
            if detected is None:
                raise RuntimeError(
                    "No local LLM server found on ports 1234/11434. Start LM Studio "
                    "or Ollama, or set LOCAL_BASE_URL/LOCAL_MODEL."
                )
            base_url, detected_model = detected
            model = model or detected_model
        if not model:
            model = pick_local_model(available_local_models(base_url)) or LOCAL_FALLBACK_MODEL
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=os.environ.get("LOCAL_API_KEY", "not-needed"),
            temperature=0,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"))
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.environ.get("AGENT_MODEL", "claude-3-5-sonnet-latest"))
    raise RuntimeError(
        "No LLM provider configured. Set GOOGLE_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, or LOCAL_BASE_URL/LOCAL_MODEL for a local server "
        "(LM Studio/Ollama). Optionally set AGENT_PROVIDER/AGENT_MODEL."
    )


def build_react_agent(llm: Any, tools: list[Any], prompt: str):
    import warnings

    from langgraph.prebuilt import create_react_agent
    from langgraph.warnings import LangGraphDeprecationWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangGraphDeprecationWarning)
        return create_react_agent(llm, tools, prompt=prompt)


def build_agent(provider: str | None = None):
    llm = _build_llm(provider)
    return build_react_agent(llm, build_tools(), SYSTEM_PROMPT)


def ask(question: str, provider: str | None = None) -> str:
    agent = build_agent(provider)
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m petrinet_lab.agent.agent \"your question\"")
        raise SystemExit(1)
    print(ask(" ".join(sys.argv[1:])))
