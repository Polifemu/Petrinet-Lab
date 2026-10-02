"""PNML (Petri Net Markup Language) import and export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .model import PetriNet

_PNML_TYPE = "http://www.pnml.org/version-2009/grammar/pnmlcoremodel"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def _text_of(element: ET.Element, child_name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == child_name:
            for text in child.iter():
                if _local_name(text.tag) == "text" and text.text is not None:
                    return text.text.strip()
            if child.text:
                return child.text.strip()
    return None


def to_pnml(net: PetriNet) -> str:
    pnml = ET.Element("pnml")
    net_element = ET.SubElement(pnml, "net", {"id": net.name or "net1", "type": _PNML_TYPE})
    for place in net.places.values():
        element = ET.SubElement(net_element, "place", {"id": place.id})
        if place.name and place.name != place.id:
            name = ET.SubElement(element, "name")
            ET.SubElement(name, "text").text = place.name
        if place.initial:
            marking = ET.SubElement(element, "initialMarking")
            ET.SubElement(marking, "text").text = str(place.initial)
    for transition in net.transitions.values():
        element = ET.SubElement(net_element, "transition", {"id": transition.id})
        if transition.name and transition.name != transition.id:
            name = ET.SubElement(element, "name")
            ET.SubElement(name, "text").text = transition.name
    for index, arc in enumerate(net.arcs, start=1):
        element = ET.SubElement(
            net_element, "arc", {"id": f"a{index}", "source": arc.source, "target": arc.target}
        )
        if arc.weight != 1:
            inscription = ET.SubElement(element, "inscription")
            ET.SubElement(inscription, "text").text = str(arc.weight)
    ET.indent(pnml, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(pnml, encoding="unicode")


def from_pnml(text: str) -> PetriNet:
    root = ET.fromstring(text)
    net_element = next((e for e in root.iter() if _local_name(e.tag) == "net"), None)
    if net_element is None:
        raise ValueError("no <net> element found in PNML document")
    net = PetriNet(str(net_element.get("id", "net1")))
    pending_arcs: list[tuple[str, str, int]] = []
    for element in net_element.iter():
        kind = _local_name(element.tag)
        node_id = element.get("id")
        if kind == "place" and node_id:
            initial = _text_of(element, "initialMarking")
            name = _text_of(element, "name")
            net.add_place(node_id, "" if name in (None, node_id) else name, int(initial) if initial else 0)
        elif kind == "transition" and node_id:
            name = _text_of(element, "name")
            net.add_transition(node_id, "" if name in (None, node_id) else name)
        elif kind == "arc":
            source, target = element.get("source"), element.get("target")
            if not source or not target:
                raise ValueError("arc without source/target in PNML document")
            inscription = _text_of(element, "inscription")
            pending_arcs.append((source, target, int(inscription) if inscription else 1))
    for source, target, weight in pending_arcs:
        net.add_arc(source, target, weight)
    return net


def save_pnml(net: PetriNet, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(to_pnml(net), encoding="utf-8")
    return path


def load_pnml(path: str | Path) -> PetriNet:
    return from_pnml(Path(path).read_text(encoding="utf-8"))
