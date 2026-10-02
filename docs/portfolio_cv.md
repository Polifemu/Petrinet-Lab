# PetriNet Lab — come usarlo nel CV e nei colloqui

Materiale pronto per posizionare il progetto nel curriculum di **Filippo
Polidori** (AI & Process Automation | BPM & Process Mining) e per raccontarlo
in colloquio.

## One-liner

> **PetriNet Lab** — Toolkit Python per l'analisi di reti di Petri P/T,
> temporali (Merlin–Farber) e stocastiche: state class graph con DBM esatto,
> CTMC con distribuzione stazionaria, simulazione Gillespie, round-trip
> process mining con PM4Py e agente LLM per l'analisi interattiva dei modelli.

## Bullet per il CV

**Italiano**

- Progettato e sviluppato **PetriNet Lab**, libreria Python per modellazione e
  analisi di reti di Petri: grafo di raggiungibilità, liveness via SCC,
  P/T-invarianti calcolati esattamente su ℚ, export/import PNML.
- Implementato da zero lo **state class graph** per reti di Petri temporali
  (semantica strong) con Difference Bound Matrix esatta su frazioni: finestre
  di firing, rilevazione di time-lock e simulazione di tracce temporali.
- Costruito il **CTMC** di reti stocastiche con tassi dipendenti dalla
  marcatura, distribuzione stazionaria, uniformization e simulazione esatta di
  Gillespie; validazione analitica su modelli M/M/1/K e macchine
  failure/repair.
- Realizzato il **round-trip process mining**: modello di processo → event log
  token-tagged → discovery con PM4Py (inductive miner) → conformance checking;
  il token replay implementato internamente raggiunge la stessa fitness di
  PM4Py (1.00) sul modello riscoperto.
- Integrato un **agente LangGraph** con 8 tool di analisi e una dashboard
  **Streamlit** per l'esplorazione interattiva di proprietà strutturali,
  temporali e stocastiche.
- Orchestrato un sistema **multi-agente LangGraph** (supervisor + 5 specialisti:
  struttura, comportamento, timing, stocastica, process mining) con nodo di
  sintesi finale, funzionante anche con **LLM locali** (LM Studio/Ollama, tool
  calling e auto-detect dell'endpoint).
- **Cross-validato** il motore di reachability interno contro **SNAKES**
  (semantica P/T indipendente) e il transition system di **PM4Py**: stati ed
  edge coincidono su tutti i modelli di test (4 categorie, incluso un
  deadlock), a dimostrazione della correttezza dell'implementazione.

**English**

- Built **PetriNet Lab**, a Python toolkit for P/T, time and stochastic Petri
  nets: reachability graph, SCC-based liveness, exact rational P/T-invariants,
  PNML import/export.
- Implemented a from-scratch **state class graph** for Time Petri Nets (strong
  semantics) using exact fractional Difference Bound Matrices: firing windows,
  time-lock detection, timed trace simulation.
- Generated **CTMCs** for stochastic nets with marking-dependent rates,
  stationary distribution, uniformization and exact Gillespie simulation,
  validated against M/M/1/K and failure/repair models.
- Delivered a **process-mining round trip**: process model → token-tagged event
  log → PM4Py discovery (inductive miner) → conformance checking; the in-house
  token replay matches PM4Py fitness (1.00) on the rediscovered model.
- Wired an **LLM agent (LangGraph)** with 8 analysis tools and a **Streamlit**
  dashboard for interactive exploration of structural, temporal and stochastic
  properties.
- Orchestrated a **LangGraph multi-agent system** (supervisor + 5 specialists:
  structure, behaviour, timing, stochastic, process mining) with a final
  synthesis node, running on cloud providers and **local LLMs** (LM Studio/
  Ollama, tool calling and endpoint auto-detection).
- **Cross-validated** the in-house reachability engine against **SNAKES**
  (independent P/T semantics) and **PM4Py**'s transition system: states and
  edges match on every test model (4 categories, deadlock included).

## Mappatura competenza → codice (per chi clona il repo)

| Competenza | Dove guardare |
|---|---|
| BPMN/workflow net, process modeling | `core/model.py:is_workflow_net`, `examples/04` |
| Process mining, discovery, conformance | `mining/discovery.py`, `mining/conformance.py` |
| PM4Py, integrazione tool | `mining/discovery.py:to_lab_net`, PNML round-trip |
| Reti temporali, teoria della concorrenza | `core/temporal.py` (DBM, state class) |
| CTMC, performance evaluation | `core/stochastic.py` |
| Data engineering / simulazione | `core/simulation.py`, `examples/` |
| AI engineering, LLM + tool calling | `agent/tools.py`, `agent/agent.py` |
| Testing e cross-validation | `tests/` (66 test), `interop/`, `make validate` |

## Domande probabili in colloquio

1. **Perché le state class invece della semantica temporale esplicita?**
   Lo state class graph raggruppa infinite valutazioni degli orologi in un
   numero finito di classi con dominio convesso; il DBM consente firabilità e
   successori esatti senza discretizzare il tempo (denso).
2. **Cosa garantisce un P-invariante?**
   Che `xᵀm` è costante per ogni marcatura raggiungibile: utile per dimostrare
   boundedness (invariante positivo ⇒ conservativa) e per il dimensionamento.
3. **Differenza tra token replay e alignments?**
   Il token replay è veloce ma pessimistico e non ottimale con transizioni
   silenziose; gli alignments calcolano la sequenza legale più vicina alla
   traccia (A*) e danno fitness/diagnostica più robuste.
4. **Come hai validato il codice stocastico?**
   Due modelli con soluzione analitica nota (M/M/1/K geometrica troncata e
   catena failure/repair) più confronto tra frazione di tempo empirica via SSA
   e distribuzione stazionaria.
5. **Perché PM4Py opzionale e non obbligatorio?**
   Il core formale resta testabile e leggero; PM4Py entra solo nel ponte di
   process mining, con test che si auto-saltano se manca (extra `[mining]`).

## Pubblicazione GitHub

```bash
cd ~/Code/petrinet-lab
git init
git add .
git commit -m "feat: PetriNet Lab — P/T, temporal and stochastic Petri nets with process-mining bridge"
git branch -M main
git remote add origin git@github.com:Polifemu/petrinet-lab.git
git push -u origin main
```

Suggerimenti repo:

- Description: `Petri net toolkit: P/T + temporal (DBM state classes) + stochastic
  (CTMC/SSA) + PM4Py round-trip + LangGraph agent`
- Topics: `petri-nets`, `time-petri-nets`, `stochastic-petri-nets`,
  `process-mining`, `pm4py`, `bpm`, `python`, `langgraph`, `streamlit`
- Spuntare la Release `v0.1.0` e allegare `output/mining_report.md`.

## Post LinkedIn (bozza)

> Ho da poco completato **PetriNet Lab**, un progetto in cui ho implementato da
> zero l'analisi di reti di Petri classiche, temporali e stocastiche in Python:
> state class graph con DBM esatta, CTMC con uniformization e simulazione di
> Gillespie, più un round-trip di process mining (modello → log → discovery
> con PM4Py → conformance) e un agente LLM per l'analisi interattiva.
>
> Numeri: token replay interno allineato a PM4Py (fitness 1.00) sul modello
> riscoperto, finestre di firing esatte e time-lock corretti su esempi con
> concorrenza, stazionaria validata su M/M/1/K.
>
> Repo: github.com/Polifemu/petrinet-lab
>
> #ProcessMining #PetriNets #BPM #Python #LLM

## Collegamento con il progetto "Process Mining BPI 2012" già nel CV

I due progetti raccontano un percorso coerente e non duplicato:

- **BPI 2012**: dati reali, scala (262k eventi), discovery/conformance con
  PM4Py, analisi performance e rework.
- **PetriNet Lab**: fondamenti formali sotto a quei tool (invarianti,
  reachability, reti temporali e stocastiche), più simulazione e agente.

In colloquio: *"Conosco PM4Py perché l'ho usato su un log reale da 262k
eventi; conosco la matematica dietro PM4Py perché ho implementato da zero
state class graph, invarianti e CTMC."*
