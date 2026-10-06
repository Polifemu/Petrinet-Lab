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
│   │   ├── gspn.py           # builder GSPN (immediate/timed, pesi, priorità, guardie),
│   │   │                     #   eliminazione stati vanishing, throughput immediate
│   │   ├── templates.py      # M/M/1/K, M/M/c/K, machine, producer-consumer, retry
│   │   ├── simulation.py     # simulazione token-tagged → event log (e SSA per casi)
│   │   └── viz.py            # rendering Matplotlib/NetworkX (senza Graphviz)
│   ├── mining/
│   │   ├── discovery.py      # PM4Py → rete interna, statistiche log
│   │   ├── conformance.py    # token replay (interno + PM4Py), alignments
│   │   ├── stochastic_discovery.py  # stima MLE dei tassi da event log
│   │   └── compare.py        # diff strutturale per id/etichetta, fitness
│   ├── interop/
│   │   ├── snakes_bridge.py  # conversione + state space con semantica SNAKES
│   │   └── pm4py_bridge.py   # PM4Py ↔ core, transition system indipendente
│   └── agent/
│       ├── tools.py          # 8 tool JSON-serializzabili + dispatch
│       ├── agent.py          # ReAct LangGraph (Gemini/OpenAI/Anthropic/locali)
│       ├── multiagent.py     # supervisor + 5 specialisti (StateGraph)
│       └── app.py            # dashboard Streamlit
├── tests/                    # 93 test: modello, PNML, invarianti, reachability,
│                             # temporale, stocastico, GSPN, template, discovery
│                             # stocastica, mining, agent tools, viz
├── examples/                 # script eseguibili end-to-end (01–08)
└── docs/                     # teoria, architettura, materiale portfolio/CV
```

## Flussi principali

1. **Analisi formale**: `model` → `reachability` / `invariants` → risultato JSON.
2. **Timing**: `model` + intervalli → `temporal.build_state_class_graph` → classi,
   bound di firing, time-lock; `simulate_timed` per tracce temporali.
3. **Performance**: `model` + tassi → `stochastic.build_ctmc` → stazionaria,
   throughput, sojourn; `simulate_ssa` per tracce stocastiche.
4. **GSPN**: `gspn.StochasticPetriNet` (o un template) → eliminazione degli
   stati vanishing (`(I−Q)^{-1}R`) → `build_gspn_ctmc` → analisi come in 3;
   `immediate_throughputs` per i flussi delle immediate.
5. **Round-trip process mining**: `simulate_event_log` → `mining.discovery`
   (PM4Py) → rete interna via PNML → `mining.conformance` → fitness.
6. **Discovery stocastica**: `simulate_stochastic_event_log` (o log reale) →
   `mining.stochastic_discovery.estimate_rates` (replay + MLE) →
   `compare_rates`; `discover_stochastic` combina discovery PM4Py e stima.
7. **Agente/dashboard**: `agent.tools` espone 1–5 come tool; LangGraph decide
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
- **Eliminazione vanishing esatta**: l'assorbimento delle immediate è risolto
  con algebra lineare (`solve` su `I−Q`), senza discretizzare né troncare; le
  classi chiuse senza uscita tangibile sono segnalate esplicitamente.
- **Visibilità per label**: nella discovery stocastica una transizione è
  visibile se la sua label compare nel log; funziona sia per reti costruite a
  mano (label = id) sia per reti PM4Py (transizioni silenti senza nome).
