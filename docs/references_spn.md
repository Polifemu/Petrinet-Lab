# Reti di Petri stocastiche: manuali, articoli e strumenti

Bibliografia annotata per costruire e valutare reti di Petri stocastiche (SPN),
generalizzate (GSPN), deterministiche/stocastiche (DSPN) e non-Markoviane, più
il filone di *discovery* stocastica da event log implementato in
`mining/stochastic_discovery.py`.

## Manuali e libri di riferimento

- M. Ajmone Marsan, G. Balbo, G. Conte, S. Donatelli, G. Franceschinis,
  *Modelling with Generalized Stochastic Petri Nets*, Wiley, 1995.
  [10.1145/288197.581193](https://doi.org/10.1145/288197.581193)
  — il manuale di riferimento GSPN: transizioni timed/immediate, pesi,
  priorità, eliminazione degli stati vanishing, misure di performance.
- G. Balbo, *Introduction to Generalized Stochastic Petri Nets*, SFM 2007,
  LNCS 4486. [10.1007/978-3-540-72522-0_3](https://doi.org/10.1007/978-3-540-72522-0_3)
  — dispense compactte con semantica formale e dimostrazioni di eliminazione.
- F. Bause, P. S. Kritzinger, *Stochastic Petri Nets: An Introduction to the
  Theory*, 2nd ed., Vieweg, 2002 — costruzione delle catene di Markov
  soggiacenti, classificazione degli stati, analisi transiente.
- P. J. Haas, *Stochastic Petri Nets: Modelling, Stability, Simulation*,
  Springer, 2002. [10.1007/b97265](https://doi.org/10.1007/b97265)
  — stabilità, rigenerazione, simulazione (base teorica dell'SSA).
- R. German, *Performance Analysis of Communication Systems: Modeling with
  Non-Markovian Stochastic Petri Nets*, Wiley, 2000 — metodi a variabili
  supplementari per distribuzioni non esponenziali.
- C. Lindemann, *Performance Modelling with Deterministic and Stochastic Petri
  Nets*, Wiley, 1998. [10.1145/288197.581195](https://doi.org/10.1145/288197.581195)
  — DSPN, analisi transiente e stazionaria.
- T. Murata, *Petri Nets: Properties, Analysis and Applications*, Proc. IEEE,
  1989. [10.1109/5.24143](https://doi.org/10.1109/5.24143) — proprietà
  strutturali delle reti P/T (base del core).

## Articoli fondativi

- M. K. Molloy, *Performance Analysis Using Stochastic Petri Nets*, IEEE
  Trans. Computers, 1982.
  [10.1109/tc.1982.1676110](https://doi.org/10.1109/tc.1982.1676110)
  — introduzione della semantica a tempi esponenziali e della catena di Markov
  soggiacente.
- M. Ajmone Marsan, G. Conte, G. Balbo, *A Class of Generalized Stochastic
  Petri Nets for the Performance Evaluation of Multiprocessor Systems*, ACM
  TOCS, 1984. [10.1145/190.191](https://doi.org/10.1145/190.191) — definizione
  delle GSPN con transizioni immediate, pesi e priorità.
- G. Ciardo, J. Muppala, K. Trivedi, *SPNP: Stochastic Petri Net Package*,
  PNPM 1989. [10.1109/pnpm.1989.68548](https://doi.org/10.1109/pnpm.1989.68548)
  — l'implementazione storica; utili le convenzioni su priorità.
- G. Ciardo, C. Lindemann, *Analysis of Deterministic and Stochastic Petri
  Nets*, PNPM 1993.
  [10.1109/pnpm.1993.393454](https://doi.org/10.1109/pnpm.1993.393454) —
  estensione DSPN con transizioni deterministiche.
- E. G. Amparore, G. Balbo, M. Beccuti, S. Donatelli, G. Franceschinis,
  *30 Years of GreatSPN*, Springer, 2016.
  [10.1007/978-3-319-30599-8_9](https://doi.org/10.1007/978-3-319-30599-8_9)
  — panoramica dello strumento di riferimento con riferimenti per ogni
  estensione (GSPN, SWN, model checking stocastico).

## Costruzione automatica da dati (stochastic process discovery)

- A. Rogge-Solti, W. M. P. van der Aalst, *Discovering Stochastic Petri Nets
  with Arbitrary Delay Distributions from Event Logs*, BPM Workshops, 2014.
  [10.1007/978-3-319-06257-0_2](https://doi.org/10.1007/978-3-319-06257-0_2)
  — primo metodo completo: replay conforme per tempi di abilitazione e stima
  delle distribuzioni (anche non esponenziali); base dell'estimatore qui
  implementato (versione esponenziale).
- W. M. P. van der Aalst, S. J. J. Leemans, *Learning Generalized Stochastic
  Petri Nets From Event Data*, 2024.
  [10.1007/978-3-031-75778-5_1](https://doi.org/10.1007/978-3-031-75778-5_1)
  — scoperta GSPN con transizioni immediate/silenti e stima dei parametri.
- A. T. Burke, S. J. J. Leemans, M. T. Wynn, *Stochastic Process Discovery by
  Weight Estimation*, BPM Workshops, 2021.
  [10.1007/978-3-030-72693-5_20](https://doi.org/10.1007/978-3-030-72693-5_20)
  — stima dei pesi delle scelte sotto una distribuzione dei tempi globale.
- S. J. J. Leemans, T. Li, *Stochastic Process Discovery: Can It Be Done
  Optimally?*, 2024.
  [10.1007/978-3-031-61057-8_3](https://doi.org/10.1007/978-3-031-61057-8_3)
- H. Alkhammash, A. Polyvyanyy et al., *Stochastic Directly-Follows Process
  Discovery Using Grammatical Inference*, 2024.
  [10.1007/978-3-031-61057-8_6](https://doi.org/10.1007/978-3-031-61057-8_6)
- P. Cry, A. Horváth et al., *A Framework for Optimisation Based Stochastic
  Process Discovery*, 2024.
  [10.1007/978-3-031-68416-6_3](https://doi.org/10.1007/978-3-031-68416-6_3)
- F. Mannhardt, S. J. J. Leemans et al., *Modelling Data-Aware Stochastic
  Processes — Discovery and Conformance Checking*, 2023.
  [10.1007/978-3-031-33620-1_5](https://doi.org/10.1007/978-3-031-33620-1_5)
- S. J. J. Leemans, W. M. P. van der Aalst et al., *Stochastic Process Mining:
  Earth Movers' Stochastic Conformance*, Information Systems, 2021.
  [10.1016/j.is.2021.101724](https://doi.org/10.1016/j.is.2021.101724)
  — conformance stocastica (confronto di distribuzioni di varianti), utile
  per validare le reti scoperte.

## Strumenti e manuali software

- **GreatSPN** — [github.com/greatspn](https://github.com/greatspn): editor e
  analizzatore GSPN/SWN con model checking stocastico; riferimento per
  semantica e convenzioni.
- **TimeNET** — [timenet.org](https://timenet.org): DSPN e reti
  non-Markoviane, analisi transiente e stazionaria.
- **SPNP** — Duke University: pacchetto storico (Ciardo & Trivedi), analisi
  di GSPN su grandi spazi di stato.
- **PIPE2** — [github.com/sarahtattersall/PIPE](https://github.com/sarahtattersall/PIPE):
  editor PNML/GSPN con motore di analisi.
- **SNAKES** — [github.com/fpom/snakes](https://github.com/fpom/snakes):
  libreria Python usata in questo progetto come oracolo di semantica
  (`interop/snakes_bridge.py`); non copre il timing.
- **PM4Py** — [processintelligence.solutions/pm4py](https://processintelligence.solutions/pm4py):
  discovery/conformance usati in `mining/`; non copre le SPN.
- **Storm** — [stormchecker.org](https://www.stormchecker.org): model checker
  probabilistico (CTMC/MDP/MA), utile per verifiche su modelli esportati.

## Note implementative nel toolkit

- **Semantica GSPN** (`core/gspn.py`): le transizioni immediate hanno priorità
  (numero più basso vince) e peso (scelta proporzionale a parità di priorità);
  i marking con immediate abilitate sono *vanishing* e vengono eliminati
  risolvendo il sistema di assorbimento della sotto-catena immediata,
  `(I - Q)^{-1} R` (fundamental matrix), da cui si ottengono anche i firing
  attesi di ogni transizione immediata. La stessa algebra segue Balbo (2007)
  e Marsan et al. (1995).
- **Stima dei tassi** (`mining/stochastic_discovery.py`): massima
  verosimiglianza per corse esponenziali,
  `rate(t) = n_firing(t) / tempo_in_cui_t_è_abilitata(t)`, con replay
  token-based e attribuzione dell'intervallo tra eventi consecutivi al marking
  in cui la transizione che produce l'evento è abilitata (dopo eventuali mosse
  silenti). È la versione esponenziale del metodo di Rogge-Solti & van der
  Aalst (2014); `van der Aalst & Leemans (2024)` estende a GSPN con immediate
  e silenti.
- **Limiti noti**: la prima attesa di ogni caso non è osservabile (il caso
  parte dal primo evento), quindi le transizioni che firano sempre per prime
  non sono stimabili; le distribuzioni dei tempi sono assunte esponenziali;
  i pesi delle scelte sono equivalenti ai tassi nelle corse esponenziali e
  vengono esposti come conteggi empirici per marking
  (`StochasticDiscoveryResult.choice_probabilities`).
- **Template** (`core/templates.py`): M/M/1/K, M/M/c/K (con `start` immediata),
  macchina failure/repair, produttore-consumatore e retry con rework, tutti
  validati contro le forme chiuse in `tests/test_templates.py` (geometrica
  troncata, Erlang-B, throughput di ciclo).
