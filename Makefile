PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,mining,agent,interop]"

test:
	$(PY) -m pytest

demo:
	$(PY) examples/01_structure_and_reachability.py
	$(PY) examples/02_temporal_analysis.py
	$(PY) examples/03_stochastic_analysis.py

report:
	$(PY) examples/04_process_mining_roundtrip.py

discover:
	$(PY) examples/08_stochastic_discovery.py

validate:
	$(PY) examples/07_library_cross_validation.py

app:
	$(PY) -m streamlit run src/petrinet_lab/agent/app.py

agent:
	$(PY) -m petrinet_lab.agent.agent "$(Q)"

multiagent:
	$(PY) -m petrinet_lab.agent.multiagent "$(Q)"

topology:
	$(PY) -m petrinet_lab.agent.multiagent --topology

clean:
	rm -rf .pytest_cache output/*.png output/*.json output/*.md
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
