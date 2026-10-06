import math

import numpy as np
import pytest

from petrinet_lab.core.gspn import build_gspn_ctmc, resolve_marking, simulate_gspn_ssa
from petrinet_lab.core.stochastic import stationary_distribution, throughputs
from petrinet_lab.core.templates import (
    expected_attempts,
    machine,
    mm1k,
    mm1k_stationary,
    mmc,
    mmc_stationary,
    producer_consumer,
    retry,
)


def test_mm1k_matches_analytic_stationary():
    capacity, arrival, service = 4, 1.0, 1.6
    ctmc = build_gspn_ctmc(mm1k(capacity=capacity, arrival_rate=arrival, service_rate=service))
    jobs = [ctmc.marking(i)["jobs"] for i in range(len(ctmc.states))]
    assert jobs == sorted(jobs)
    assert np.allclose(stationary_distribution(ctmc), mm1k_stationary(capacity, arrival, service))


def test_mm1k_unity_load_is_uniform():
    capacity = 3
    ctmc = build_gspn_ctmc(mm1k(capacity=capacity, arrival_rate=2.0, service_rate=2.0))
    assert np.allclose(stationary_distribution(ctmc), np.full(capacity + 1, 1.0 / (capacity + 1)))


def test_mmc_loss_system_matches_erlang_b():
    servers, arrival, service = 3, 2.0, 1.0
    ctmc = build_gspn_ctmc(mmc(servers=servers, buffer=0, arrival_rate=arrival, service_rate=service))
    distribution = stationary_distribution(ctmc)
    reference = mmc_stationary(servers, 0, arrival, service)
    assert len(ctmc.states) == servers + 1
    assert np.allclose(distribution, reference)
    blocking = reference[-1]
    offered = arrival / service
    erlang_b = (offered**servers / math.factorial(servers)) / (
        sum(offered**k / math.factorial(k) for k in range(servers + 1))
    )
    assert np.isclose(blocking, erlang_b)


def test_mmc_with_buffer_matches_analytic():
    ctmc = build_gspn_ctmc(mmc(servers=2, buffer=3, arrival_rate=1.5, service_rate=1.0))
    assert len(ctmc.states) == 2 + 3 + 1
    assert np.allclose(
        stationary_distribution(ctmc), mmc_stationary(2, 3, 1.5, 1.0)
    )


def test_machine_stationary():
    ctmc = build_gspn_ctmc(machine(break_rate=1.0, repair_rate=3.0))
    assert np.allclose(stationary_distribution(ctmc), [0.75, 0.25])


def test_producer_consumer_throughput():
    production, consumption = 1.0, 2.0
    ctmc = build_gspn_ctmc(producer_consumer(buffer=1, production_rate=production, consumption_rate=consumption))
    distribution = stationary_distribution(ctmc)
    expected = production * consumption / (production + consumption)
    assert np.allclose(distribution, [consumption / (production + consumption), production / (production + consumption)])
    assert np.isclose(throughputs(ctmc, distribution)["produce"], expected)


def test_retry_expected_attempts_resolution():
    gspn = retry(ok_weight=3.0, rework_weight=1.0)
    resolution = resolve_marking(gspn, {"ready": 0, "busy": 1, "done": 0})
    rework_probability = resolution.expected_firings["rework"]
    attempts = 1.0 / (1.0 - rework_probability)
    assert np.isclose(attempts, expected_attempts(3.0, 1.0))
    assert np.isclose(resolution.expected_firings["ok"], 0.75)


def test_retry_simulation_mean_attempts():
    gspn = retry(ok_weight=3.0, rework_weight=1.0)
    trials = 300
    total = sum(
        sum(1 for event in simulate_gspn_ssa(gspn, seed=seed) if event.transition == "work")
        for seed in range(trials)
    )
    mean = total / trials
    assert abs(mean - 4.0 / 3.0) < 0.12


def test_templates_reject_invalid_parameters():
    with pytest.raises(ValueError):
        mm1k(capacity=0)
    with pytest.raises(ValueError):
        mmc(servers=0)
    with pytest.raises(ValueError):
        mmc(buffer=-1)
    with pytest.raises(ValueError):
        producer_consumer(buffer=0)
    with pytest.raises(ValueError):
        retry(ok_weight=0.0)
