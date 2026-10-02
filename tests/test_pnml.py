from petrinet_lab.core import pnml
from petrinet_lab.core.model import PetriNet


def test_roundtrip_with_weights_and_names(tmp_path):
    net = PetriNet("weighted")
    net.add_place("p1", name="Source", initial=2)
    net.add_place("p2", name="Sink")
    net.add_transition("t", name="Process")
    net.add_arc("p1", "t", weight=2)
    net.add_arc("t", "p2", weight=3)
    text = pnml.to_pnml(net)
    restored = pnml.from_pnml(text)
    assert restored.to_dict() == net.to_dict()
    path = pnml.save_pnml(net, tmp_path / "net.pnml")
    assert pnml.load_pnml(path).to_dict() == net.to_dict()


def test_roundtrip_without_explicit_names(cycle):
    restored = pnml.from_pnml(pnml.to_pnml(cycle))
    assert restored.to_dict() == cycle.to_dict()


def test_parse_pnml_with_namespace():
    text = """<?xml version="1.0" encoding="UTF-8"?>
    <pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
      <net id="n1" type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel">
        <place id="p1"><name><text>P1</text></name>
          <initialMarking><text>3</text></initialMarking></place>
        <transition id="t1"><name><text>T1</text></name></transition>
        <arc id="a1" source="p1" target="t1">
          <inscription><text>2</text></inscription></arc>
      </net>
    </pnml>"""
    net = pnml.from_pnml(text)
    assert net.name == "n1"
    assert net.places["p1"].initial == 3
    assert net.places["p1"].name == "P1"
    assert net.transitions["t1"].name == "T1"
    assert net.preset("t1") == {"p1": 2}


def test_missing_net_raises():
    try:
        pnml.from_pnml("<pnml></pnml>")
    except ValueError as error:
        assert "no <net>" in str(error)
    else:
        raise AssertionError("expected ValueError")
