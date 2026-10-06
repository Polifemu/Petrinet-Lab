"""GSPN construction and stochastic discovery: rates learned from an event log."""

from petrinet_lab.core.gspn import build_gspn_ctmc, immediate_throughputs, resolve_marking
from petrinet_lab.core.simulation import simulate_stochastic_event_log
from petrinet_lab.core.stochastic import stationary_distribution, throughputs
from petrinet_lab.core.templates import machine, mmc, mmc_stationary, retry
from petrinet_lab.mining.stochastic_discovery import compare_rates, estimate_rates

print("== GSPN con transizioni immediate: M/M/2/3 (2 server, 1 in coda) ==")
gspn = mmc(servers=2, buffer=1, arrival_rate=1.5, service_rate=1.0)
ctmc = build_gspn_ctmc(gspn)
distribution = stationary_distribution(ctmc)
print("stati tangibili:", len(ctmc.states))
print("stazionaria:", distribution.round(4))
print("analitica:  ", mmc_stationary(2, 1, 1.5, 1.0).round(4))
print("throughput timed:", {k: round(v, 4) for k, v in throughputs(ctmc, distribution).items()})
print("throughput immediate:", {k: round(v, 4) for k, v in immediate_throughputs(ctmc, distribution).items()})

print("\n== Eliminazione degli stati vanishing: retry con rework ==")
gspn = retry(ok_weight=3.0, rework_weight=1.0)
resolution = resolve_marking(gspn, {"ready": 0, "busy": 1, "done": 0})
print("assorbimento:", resolution.targets)
print("firing attesi:", resolution.expected_firings)

print("\n== Discovery stocastica: stima dei tassi da un log simulato ==")
net = machine().net
true_rates = {"break": 1.0, "repair": 3.0}
events = simulate_stochastic_event_log(net, true_rates, n_cases=500, max_time=60.0, seed=42)
print("eventi simulati:", len(events))
result = estimate_rates(net, events)
print("tassi stimati:", {k: round(v, 4) for k, v in result.rates.items()})
comparison = compare_rates(result.rates, true_rates)
print("errore relativo per transizione:", {k: round(v, 4) for k, v in comparison["per_transition"].items()})
print("errore medio:", round(comparison["mean_relative_error"], 4))
print("replay fitness:", result.fitness["replay_fitness"])
