# Architettura

```text
petrinet-lab/
├── src/petrinet_lab/
│   ├── core/
│   │   ├── model.py          # PetriNet, Place, Transition, Arc, semantica di firing
│   │   ├── pnml.py           # import/export PNML 2009 (compatibile PM4Py, con <page>)
│   │   ├── reachability.py   # grafo di raggiungibilità, deadlock, boundedness, liveness (SCC)
│   │   ├── invariants.py     # matrice di incidenza, spazi nulli su ℚ, P/T-invarianti
│   │   ├── temporal.py       # DBM, state class graph, simulazione temporale
│   │   ├── stochastic.py     # CTMC, stazionaria, uniformization, SSA
│   │   ├── simulation.py     # simulazione token-tagged → event log
│   │   └── viz.py            # rendering Matplotlib/NetworkX (senza Graphviz)
│   ├── mining/
│   │   ├── discovery.py      # PM4Py → rete interna, statistiche log
│   │   ├── conformance.py    # token replay (interno + PM4Py), alignments
│   │   └── compare.py        # diff strutturale per id/etichetta, fitness
│   ├── interop/
│   │   ├── snakes_bridge.py  # conversione + state space con semantica SNAKES
│   │   └── pm4py_bridge.py   # PM4Py ↔ core, transition system indipendente
│   └── agent/
│       ├── tools.py          # 8 tool JSON-serializzabili + dispatch
│       ├── agent.py          # ReAct LangGraph (Gemini/OpenAI/Anthropic/locali)
│       ├── multiagent.py     # supervisor + 5 specialisti (StateGraph)
│       └── app.py            # dashboard Streamlit
├── tests/                    # 66 test: modello, PNML, invarianti, reachability,
│                             # temporale, stocastico, mining, agent tools, viz
├── examples/                 # script eseguibili end-to-end
└── docs/                     # teoria, architettura, materiale portfolio/CV
```

## Flussi principali

1. **Analisi formale**: `model` → `reachability` / `invariants` → risultato JSON.
2. **Timing**: `model` + intervalli → `temporal.build_state_class_graph` → classi,
   bound di firing, time-lock; `simulate_timed` per tracce temporali.
3. **Performance**: `model` + tassi → `stochastic.build_ctmc` → stazionaria,
   throughput, sojourn; `simulate_ssa` per tracce stocastiche.
4. **Round-trip process mining**: `simulate_event_log` → `mining.discovery`
   (PM4Py) → rete interna via PNML → `mining.conformance` → fitness.
5. **Agente/dashboard**: `agent.tools` espone 1–4 come tool; LangGraph decide
   quali invocare (`agent.py`, singolo ReAct; `multiagent.py`, supervisor +
   5 specialisti); Streamlit li visualizza.

## Scelte implementative

- **Zero dipendenze per l'analisi formale del core**: Gauss-Jordan esatto su
  `Fraction`; DBM esatto su `Fraction`; NumPy solo per la parte stocastica;
  NetworkX/Matplotlib per grafi e rendering.
- **PM4Py opzionale**: `mining` importa PM4Py lazy; i test si auto-saltano se
  non installato (extra `[mining]`).
- **LLM opzionale**: `agent.agent` carica LangGraph/provider lazy; i tool sono
  utilizzabili e testabili senza alcuna API key.
- **PNML come ponte**: la conversione dei modelli PM4Py passa da PNML, così il
  parser interno è esercitato dal caso reale e resta indipendente dalla versione.
