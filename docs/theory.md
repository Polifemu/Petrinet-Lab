# Teoria — PetriNet Lab

Riferimenti sintetici per i tre formalismi implementati e per il ponte con il
process mining.

## 1. Reti P/T (Place/Transition)

Una rete di Petri P/T è una tupla `N = (P, T, F, W, m0)`:

- `P` posti, `T` transizioni, `F ⊆ (P×T) ∪ (T×P)` archi;
- `W : F → ℕ⁺` pesi degli archi;
- `m0 : P → ℕ` marcatura iniziale.

Una transizione `t` è **abilitata** in `m` se `m(p) ≥ W(p,t)` per ogni `p ∈ •t`.
Il firing produce `m'(p) = m(p) - W(p,t) + W(t,p)`.

### Proprietà comportamentali (finite se il grafo è finito)

| Proprietà | Definizione | Implementazione |
|---|---|---|
| Boundedness | esiste `k` con `m(p) ≤ k` per ogni `m` raggiungibile | `ReachabilityGraph.token_bounds` |
| Deadlock | marcatura raggiungibile senza transizioni abilitate | `deadlock_markings` |
| Dead transition | non compare in alcun firing del grafo | `dead_transitions` |
| Liveness | livelli 0–4 (dead → live) | `liveness_level` via SCC |

### Proprietà strutturali (invarianti)

Sia `C` la matrice di incidenza `C[p,t] = W(t,p) − W(p,t)`.

- **P-invariante**: `x` con `xᵀC = 0`, `x ≥ 0` ⇒ `xᵀm` costante per ogni marcatura raggiungibile.
- **T-invariante**: `y` con `Cy = 0`, `y ≥ 0` ⇒ esiste una sequenza di firing che riporta alla marcatura iniziale.
- Se esiste un P-invariante con tutte le componenti positive la rete è **conservativa** e **strutturalmente limitata**.

Lo spazio nullo è calcolato esattamente su `ℚ` con Gauss-Jordan
(`core/invariants.py`), poi si cercano combinazioni non negative minimali.

## 2. Reti di Petri temporali (Merlin–Farber)

Ogni transizione ha un intervallo statico `[α, β]` di firing. Semantica
**strong**: il tempo può avanzare solo se nessun orologio supera il proprio
`β`; una transizione che non spara entro `β` diventa non più firabile
(*time-lock*).

### State class graph

Una *state class* è una coppia `(m, D)` con `D` dominio convesso sugli
orologi. `D` è rappresentato come **Difference Bound Matrix** sulle date di
abilitazione `d_i` (data di abilitazione relativa all'istante corrente, `d_i ≤ 0`):

```
d_i - d_j ≤ B[i][j],          nodo 0 ≡ costante 0
età del orologio: x_i = -d_i
```

- **Firabilità e delay**: si aggiunge la variabile `δ` (attesa) con i vincoli
  `δ ≥ α_t + d_t`, `δ ≤ β_u + d_u` per ogni transizione abilitata `u`, `δ ≥ 0`;
  chiusura DBM e verifica di consistenza. I bound di `δ` danno l'intervallo
  esatto di attesa per il firing (usato anche per min/max firing time per edge).
- **Successore**: proiezione esatta del sistema con `δ` sulle variabili
  persistenti (quelle abilitate prima e dopo, secondo la definizione classica
  `Enabled(m') \ Enabled(m \ •t)`), con nuovi orologi a zero.
- Ogni classe è deduplicata per inclusione dei domini a parità di marcatura;
  questo garantisce terminazione sulla maggior parte dei modelli.

## 3. Reti di Petri stocastiche (SPN)

A ogni transizione è associato un tasso esponenziale `λ_t` (costante o
dipendente dalla marcatura, `λ_t : M → ℝ⁺`). Il processo delle marcature è una
**catena di Markov a tempo continuo** con generatore:

```
Q[i,j] = Σ_{t : m_i → m_j} λ_t          (i ≠ j)
Q[i,i] = - Σ_{t : m_i → m_j, j ≠ i} λ_t
```

- **Distribuzione stazionaria**: soluzione di `πQ = 0`, `Σπ = 1` (least
  squares con controllo di rango; catene riducibili segnalate).
- **Transitorio**: uniformization `π(t) = Σ_k e^{-Λt} (Λt)^k/k! · π₀ P^k`,
  con `P = I + Q/Λ`, `Λ ≥ max_i |Q[i,i]|`.
- **Throughput**: `Σ_i π_i λ_t(m_i)` (con self-loop contabilizzati a parte).
- **SSA di Gillespie**: tempo di attesa `Exp(Σλ)`, transizione scelta con
  probabilità proporzionale al tasso; simulazione esatta sample-path.

## 4. Reti di Petri stocastiche generalizzate (GSPN)

`core/gspn.py` estende le SPN con transizioni **immediate** (tempo nullo, peso
e priorità) e **guardie** sulla marcatura. Le GSPN alternano marcature
*tangibili* (solo transizioni timed abilitate) e *vanishing* (almeno una
immediata abilitata).

- **Semantica**: in una marcatura, se esistono immediate abilitate si sceglie
  prima la classe a priorità minima (numero più basso), poi si campiona la
  transizione con probabilità proporzionale al peso. Le timed competono in
  corsa esponenziale solo nelle marcature tangibili.
- **Eliminazione degli stati vanishing**: per ogni marcatura di partenza si
  costruisce la sotto-catena immediata con matrice `Q` (transizioni tra
  vanishing) e `R` (assorbimento nei tangibili), si risolve la *fundamental
  matrix* `N = (I − Q)^{-1}` e si ottiene la distribuzione di assorbimento
  `N·R`, oltre ai firing attesi per ogni immediata
  (`expected_firings[j] = Σ_u N[start,u]·p_j(u)`). Una classe chiusa senza
  uscita tangibile produce `ImmediateLoopError`.
- **Generatore tangibile**: `Q[i,j] += λ_t(m_i)·p(m_i → m_j)`; i self-loop
  netti (ritorni alla stessa marcatura tangibile) sono contabilizzati a parte
  e non alterano il generatore. I throughput delle immediate si ricavano dai
  flussi accumulati: `θ_j = Σ_i π_i · flow[i,j]`.

## 5. Discovery stocastica da event log

`mining/stochastic_discovery.py` stima i tassi esponenziali con massima
verosimiglianza. Per un processo a corse esponenziali, il tempo speso in una
marcatura è `Exp(Σ λ)` e la transizione che scatta è scelta con probabilità
`λ_t / Σ λ`; la verosimiglianza si massimizza con:

```
rate(t) = n_firing(t) / tempo_in_cui_t_abilitata(t)
```

- Il **replay token-based** individua la transizione che produce ogni evento e
  le eventuali mosse silenti; l'intervallo di tempo tra eventi consecutivi è
  attribuito alla marcatura in cui la transizione è abilitata (dopo le mosse
  silenti). La prima attesa di ogni caso non è osservabile: i processi ciclici
  (failure/repair, code) sono stimabili per intero.
- Le scelte empiriche per marcatura (`choice_probabilities`) espongono le
  frequenze relative delle transizioni in conflitto, equivalenti ai rapporti
  tra i tassi nelle corse esponenziali.
- `compare_rates` quantifica l'errore relativo rispetto ai tassi di
  riferimento; nei test sintetici l'errore medio resta sotto il 3–5% con 500
  casi.

## 6. Ponte process mining

- **Simulazione token-tagged** (`core/simulation.py`): ogni token porta il
  `case_id`; ogni caso parte dalla marcatura iniziale e termina quando un token
  raggiunge un posto pozzo. Produce un event log compatibile XES/PM4Py.
  `simulate_stochastic_event_log` genera inoltre log temporizzati con SSA a
  partire da tassi noti (usato per validare la discovery).
- **Discovery** (`mining/discovery.py`): PM4Py (inductive miner, alpha,
  heuristics, ILP) e conversione del modello PM4Py in rete interna via PNML.
- **Token-based replay** (`mining/conformance.py`): formula standard
  `fitness = 0.5·(1 − missing/consumed) + 0.5·(1 − remaining/produced)`, con
  ricerca greedy delle mosse silenziose quando l'attività non è abilitata.
  Sul modello scoperto la fitness coincide con PM4Py (1.0 nell'esempio end-to-end).
- **Template** (`core/templates.py`): M/M/1/K, M/M/c/K, macchina
  failure/repair, produttore-consumatore e retry, con forme chiuse usate come
  oracoli nei test (geometrica troncata, Erlang-B, throughput di ciclo).

## Riferimenti

- T. Murata, *Petri Nets: Properties, Analysis and Applications*, Proc. IEEE, 1989.
- B. Berthomieu, M. Diaz, *Modeling and Verification of Time Dependent Systems Using Time Petri Nets*, IEEE TSE, 1991.
- M. K. Molloy, *Performance Analysis Using Stochastic Petri Nets*, IEEE Trans. Computers, 1982.
- M. Ajmone Marsan, G. Conte, G. Balbo, *A Class of Generalized Stochastic Petri Nets*, ACM TOCS, 1984.
- M. Ajmone Marsan et al., *Modelling with Generalized Stochastic Petri Nets*, Wiley, 1995.
- G. Balbo, *Introduction to Generalized Stochastic Petri Nets*, SFM 2007.
- P. J. Haas, *Stochastic Petri Nets: Modelling, Stability, Simulation*, Springer, 2002.
- A. Rogge-Solti, W. van der Aalst, *Discovering Stochastic Petri Nets with Arbitrary Delay Distributions from Event Logs*, BPM Workshops, 2014.
- W. van der Aalst, S. J. J. Leemans, *Learning Generalized Stochastic Petri Nets From Event Data*, 2024.
- S. J. J. Leemans, W. van der Aalst, *Stochastic Process Mining: Earth Movers' Stochastic Conformance*, Information Systems, 2021.
- W. van der Aalst, *Process Mining: Data Science in Action*, Springer, 2016.
- PM4Py: https://processintelligence.solutions/pm4py

Bibliografia completa e annotata: [`references_spn.md`](references_spn.md).
