# PetriNet Lab

[![CI](https://github.com/Polifemu/petrinet-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Polifemu/petrinet-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

Toolkit Python per **reti di Petri P/T, temporali e stocastiche**, con ponte
verso il **process mining** (PM4Py) e un **agente LLM** per l'analisi
interattiva dei modelli.

Autore: **Filippo Polidori** — AI & Process Automation | BPM & Process Mining
([github.com/Polifemu](https://github.com/Polifemu))

## Highlights

- **Reti P/T**: PNML import/export, grafo di raggiungibilità, boundedness,
  deadlock, liveness 0–4 via SCC, **P/T-invarianti esatti** su `Fraction`.
- **Reti temporali (Merlin–Farber)**: **state class graph** con Difference
  Bound Matrix esatta su frazioni, firabilità, finestre di firing, **time-lock**
  e simulazione di tracce temporali. Zero discretizzazione del tempo.
- **Reti stocastiche**: generazione esplicita del **CTMC** (tassi anche
  dipendenti dalla marcatura), distribuzione stazionaria, **uniformization**,
  throughput, sojourn time e **simulazione esatta di Gillespie**.
- **Round-trip process mining**: modello → event log token-tagged → discovery
  PM4Py (inductive miner/alpha/heuristics/ILP) → conformance con token replay
  interno e alignments PM4Py.
- **Validazione incrociata con librerie**: lo state space del motore interno
  coincide con **SNAKES** (semantica di enabling/firing indipendente) e con il
  transition system di **PM4Py** su tutti i modelli di test (stati ed edge).
- **Agenti LangGraph**: ReAct singolo con 8 tool JSON, **orchestrazione
  multi-agente** supervisor + 5 specialisti, provider cloud o **LLM locali**
  (LM Studio/Ollama con auto-detect), e **dashboard Streamlit**.
- **66 test** con casi analitici (M/M/1/K, macchina failure/repair, esempi di
  concorrenza e time-lock).

## Risultati di esempio

| Esperimento | Risultato |
|---|---|
| Round-trip scoperta (400 casi, processo con loop di rework) | 9 posti, 12 transizioni, workflow net ✓ |
| Token replay interno vs PM4Py sul modello riscoperto | **log fitness 1.00 vs 1.00** |
| Alignments PM4Py (200 tracce) | log fitness 1.00 |
| Cellula ciclica temporale `start[2,4]`, `finish[3,5]` | firing time bounds esatti `(2,4)`, `(3,5)`; traccia earliest `2,5,7,10` |
| Modello time-lock `fast[1,2]`, `slow[5,10]` | `slow` **non firabile** (strong semantics), nessuna falsa traccia |
| Macchina failure(λ=1)/repair(λ=3) | stazionaria **0.75/0.25**, throughput 0.75, SSA coerente |
| M/M/1/K (K=3, λ=1, μ=2) | stazionaria = geometrica troncata analitica |
| Cross-validation `cyclic_cell`, `parallel_workflow`, weighted, deadlock | stati/edge **identici** tra motore interno, SNAKES e PM4Py (`make validate`) |

## Quickstart

```bash
git clone https://github.com/Polifemu/petrinet-lab.git
cd petrinet-lab
make install                 # venv + dipendenze core, mining, agent, interop
make test                    # 66 test
make demo                    # esempio 01 -> 03
make report                  # round-trip process mining -> output/
make validate                # cross-validation con SNAKES e PM4Py
```

Uso minimale:

```python
from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.invariants import p_invariants
from petrinet_lab.core.reachability import build_reachability_graph

net = PetriNet("cell")
net.add_place("idle", initial=1); net.add_place("busy")
net.add_transition("start"); net.add_transition("finish")
net.add_arc("idle", "start"); net.add_arc("start", "busy")
net.add_arc("busy", "finish"); net.add_arc("finish", "idle")

print(p_invariants(net))                       # -> [{'idle': 1, 'busy': 1}]
graph = build_reachability_graph(net)
print(graph.token_bounds(), graph.liveness_level("start"))  # -> {'idle': 1, 'busy': 1} 4
```

Reti temporali:

```python
from petrinet_lab.core.temporal import TimePetriNet, build_state_class_graph

tpn = TimePetriNet(net, {"start": (2, 4), "finish": (3, 5)})
scg = build_state_class_graph(tpn)
print(scg.firing_time_bounds("start"))         # -> (2, 4)
```

Reti stocastiche:

```python
from petrinet_lab.core.stochastic import build_ctmc, stationary_distribution

ctmc = build_ctmc(net, {"start": 2.0, "finish": 1.0})
print(stationary_distribution(ctmc))           # -> [1/3, 2/3]
```

## Agenti LLM (cloud o locali) e dashboard

```bash
# locale: Ollama o LM Studio (auto-detect)
export AGENT_PROVIDER=local
export LOCAL_MODEL=qwen3:4b-instruct-2507-q4_K_M
make agent Q="Is parallel_workflow bounded? Any deadlock?"
make multiagent Q="Bounded? Deadlock? Probabilità stazionaria che la machine sia down?"
make topology                    # diagramma mermaid dell'orchestrazione

# cloud
export GOOGLE_API_KEY=...        # oppure OPENAI_API_KEY / ANTHROPIC_API_KEY
make app                         # Streamlit su http://localhost:8501
```

- `agent.agent`: ReAct singolo con 8 tool (struttura, invarianti, reachability,
  temporale, stocastica, simulazione).
- `agent.multiagent`: **supervisor + 5 specialisti** (`structure`,
  `behaviour`, `timing`, `stochastic`, `mining`) con **decomposizione in task
  focalizzati** e nodo di sintesi finale; validato end-to-end con Ollama
  (`qwen3:30b-a3b`). Topologia e dettagli in
  [`docs/multiagent.md`](docs/multiagent.md).
- Provider: Google, OpenAI, Anthropic oppure locali OpenAI-compatibili
  (`LOCAL_BASE_URL`/`LOCAL_MODEL`, default LM Studio `:1234`, Ollama `:11434`).
- Senza LLM i tool restano utilizzabili direttamente
  (`examples/05_agent_demo.py`, `examples/06_multiagent_demo.py`).

## Struttura

```text
src/petrinet_lab/
  core/       model, pnml, reachability, invariants, temporal, stochastic, simulation, viz
  mining/     discovery (PM4Py), conformance (token replay + alignments), compare
  interop/    bridge SNAKES (oracolo semantico) e PM4Py (transition system)
  agent/      tools (JSON), agent + multiagent (LangGraph), app (Streamlit)
tests/        66 test
examples/     01 struttura, 02 temporali, 03 stocastiche, 04 round-trip mining,
              05 agent, 06 multi-agente, 07 cross-validation librerie
docs/         teoria, architettura, multi-agente, materiale CV/colloquio
```

### Hand-rolled + librerie: dove e perché

| Ambito | Scelta | Motivo |
|---|---|---|
| Discovery, conformance, alignments | **PM4Py** | standard di fatto, molto più maturo di qualsiasi reimplementazione |
| PNML | parser interno + PM4Py | il parser interno serve al core; round-trip testato con PM4Py |
| Reachability P/T | interno, **cross-validato** con SNAKES e PM4Py | il motore interno è il cuore didattico; le librerie lo verificano |
| Reti temporali (state class, DBM) | **interno** | né SNAKES né PM4Py coprono le TPN |
| Reti stocastiche (CTMC, SSA) | **interno** | nessuna delle due copre le SPN; NumPy per l'algebra lineare |
| Simulazione event log token-tagged | **interno** | `pm4py.play_out` non produce log per-caso affidabili sui WF-net |
| Orchestrazione agenti | **LangGraph** | framework standard per grafi di agenti |

La teoria completa (formule, semantica, algoritmi) è in
[`docs/theory.md`](docs/theory.md); l'architettura in
[`docs/architecture.md`](docs/architecture.md].

## Progetti collegati

- [**process-mining-bpi2012**](https://github.com/Polifemu/process-mining-bpi2012) —
  analisi end-to-end di un event log reale (262.200 eventi, 13.087 casi) con
  PM4Py: discovery, conformance e performance. Il caso applicativo che
  complementa questo toolkit: qui il fondamento formale, là i dati reali.

## Note

- PM4Py (`[mining]`) e SNAKES (`[interop]`) sono dipendenze **opzionali**: i
  test correlati si auto-saltano se non installate. PM4Py è distribuito con
  licenza AGPL v3; verificare i termini per usi commerciali.
- Licenza progetto: Apache-2.0.
