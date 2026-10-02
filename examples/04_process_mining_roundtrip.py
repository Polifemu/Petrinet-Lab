"""End-to-end round-trip: design a process, simulate it, rediscover it with PM4Py.

Writes output/mining_report.md and output/metrics.json, plus PNGs of the
hand-designed and discovered models.
"""

import json
from pathlib import Path

from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.simulation import simulate_event_log, traces_from_events
from petrinet_lab.core.viz import draw_net
from petrinet_lab.mining.compare import structural_diff
from petrinet_lab.mining.conformance import (
    log_fitness,
    pm4py_alignments,
    pm4py_token_replay,
    token_replay,
)
from petrinet_lab.mining.discovery import discover, events_to_log, log_stats

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)


def build_rework_process() -> PetriNet:
    net = PetriNet("rework_process")
    for place_id, initial in [("start", 1), ("registered", 0), ("checked", 0),
                              ("approved", 0), ("rejected", 0), ("end", 0)]:
        net.add_place(place_id, initial=initial)
    for transition_id, label in [
        ("t_register", "Register"),
        ("t_check", "Check"),
        ("t_approve", "Approve"),
        ("t_reject", "Reject"),
        ("t_rework", "Rework"),
        ("t_close", "Close"),
        ("t_notify", "Notify"),
    ]:
        net.add_transition(transition_id, name=label)
    net.add_arc("start", "t_register")
    net.add_arc("t_register", "registered")
    net.add_arc("registered", "t_check")
    net.add_arc("t_check", "checked")
    net.add_arc("checked", "t_approve")
    net.add_arc("t_approve", "approved")
    net.add_arc("approved", "t_notify")
    net.add_arc("t_notify", "end")
    net.add_arc("checked", "t_reject")
    net.add_arc("t_reject", "rejected")
    net.add_arc("rejected", "t_rework")
    net.add_arc("t_rework", "checked")
    net.add_arc("rejected", "t_close")
    net.add_arc("t_close", "end")
    return net


def main() -> None:
    import matplotlib.pyplot as plt

    net = build_rework_process()
    print("designed model:", net.summary())

    events = simulate_event_log(net, n_cases=400, seed=21)
    traces = traces_from_events(events)
    log, _ = events_to_log(events)
    stats = log_stats(log)
    print("simulated log:", {key: stats[key] for key in ("cases", "events", "variants")})

    result = discover(log, "inductive")
    print("discovered model:", result.net.summary())

    ours = log_fitness(token_replay(result.net, traces, final_marking={"sink": 1}))
    reference = pm4py_token_replay(log, result)
    alignments = pm4py_alignments(log, result, max_traces=200)
    diff = structural_diff(net, result.net, by="label")

    metrics = {
        "designed": net.summary(),
        "simulated_log": {key: stats[key] for key in ("cases", "events", "variants", "activities")},
        "discovered": result.net.summary(),
        "token_replay_ours": ours,
        "token_replay_pm4py": reference,
        "alignments_pm4py": alignments,
        "structural_diff_by_label": diff,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_net(net, ax=axes[0], title="Designed model")
    draw_net(result.net, ax=axes[1], title="Discovered model (inductive miner)")
    figure.tight_layout()
    figure.savefig(OUTPUT / "04_mining_models.png", dpi=150)
    plt.close(figure)

    report = [
        "# Round-trip process mining — PetriNet Lab",
        "",
        (
            f"Log simulato: **{stats['cases']} casi / {stats['events']} eventi / "
            f"{stats['variants']} varianti** su attività {stats['activities']}."
        ),
        "",
        f"- Modello disegnato: {net.summary()['places']} posti, {net.summary()['transitions']} transizioni.",
        (
            f"- Modello scoperto (inductive miner): {result.net.summary()['places']} posti, "
            f"{result.net.summary()['transitions']} transizioni, "
            f"workflow net: {result.net.summary()['workflow_net']}."
        ),
        (
            f"- Token replay (implementazione interna): log fitness = {ours['log_fitness']:.4f}, "
            f"tracce fitting = {ours['perc_fit_traces']:.1f}%."
        ),
        f"- Token replay (PM4Py): log fitness = {reference['log_fitness']:.4f}.",
        f"- Alignments (PM4Py, 200 tracce): log fitness = {alignments['log_fitness']:.4f}.",
        "",
        "## Confronto strutturale per etichetta",
        "",
        "Nodi in comune: " + ", ".join(str(node) for node in diff["common_nodes"]),
        "",
        "Solo nel modello disegnato: "
        + (", ".join(str(node) for node in diff["only_in_first"]) or "nessuno"),
        "",
        "Solo nel modello scoperto: "
        + (", ".join(str(node) for node in diff["only_in_second"]) or "nessuno"),
        "",
        "![models](04_mining_models.png)",
    ]
    (OUTPUT / "mining_report.md").write_text("\n".join(report), encoding="utf-8")
    print("report written to output/mining_report.md")
    print("fitness ours:", ours["log_fitness"], "| pm4py:", reference["log_fitness"])


if __name__ == "__main__":
    main()
