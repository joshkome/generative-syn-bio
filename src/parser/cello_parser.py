"""Stage 2: Parse a Cello UCF JSON file into List[PartSpec].

Each gate structure's cassette device is expanded into one PartSpec per
component. Upstream/downstream context is built from the concatenated
sequences of neighboring parts in the same cassette.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.schema.part_spec import PartSpec, PartType, HostOrganism, PartConstraints


_UCF_TYPE_TO_PART_TYPE: Dict[str, PartType] = {
    "promoter":  PartType.PROMOTER,
    "rbs":       PartType.RBS,
    "cds":       PartType.CDS,
    "terminator": PartType.TERMINATOR,
    "ribozyme":  PartType.SPACER,
    "scar":      PartType.SPACER,
    "operator":  PartType.OPERATOR,
}

_UCF_TYPE_TO_ROLE: Dict[str, str] = {
    "promoter":  "transcription_promoter",
    "rbs":       "ribosome_binding_site",
    "cds":       "coding_sequence",
    "terminator": "transcription_terminator",
    "ribozyme":  "translational_insulator",
    "scar":      "assembly_scar",
    "operator":  "operator_binding_site",
}


def _clean_seq(seq: str) -> str:
    return seq.upper().strip()


def _build_parts_map(ucf_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {p["name"]: p for p in ucf_data if p.get("collection") == "parts"}


def _get_cassette_components(structure: Dict[str, Any]) -> List[str]:
    """Return the ordered part names from the cassette device of a structure."""
    for device in structure.get("devices", []):
        if "cassette" in device["name"]:
            return device["components"]
    return []


def _infer_host(ucf_data: List[Dict[str, Any]]) -> HostOrganism:
    header = next((x for x in ucf_data if x.get("collection") == "header"), {})
    organism = header.get("organism", "").lower()
    if "saccharomyces" in organism:
        return HostOrganism.SACCHAROMYCES
    if "bl21" in organism:
        return HostOrganism.ECOLI_BL21
    return HostOrganism.ECOLI_K12


def _strength_label(ucf_type: Optional[str], params: Dict[str, Any]) -> Optional[str]:
    if ucf_type == "terminator":
        val = params.get("terminator_strength")
        if val is not None:
            if val > 100:
                return "high"
            if val > 10:
                return "medium"
            return "low"
    if ucf_type == "ribozyme":
        val = params.get("ribozyme_efficiency")
        if val is not None:
            if val >= 0.9:
                return "high"
            if val >= 0.5:
                return "medium"
            return "low"
    return None


def _build_constraints(part: Dict[str, Any]) -> PartConstraints:
    params = {p["name"]: p["value"] for p in part.get("parameters", [])}
    return PartConstraints(
        strength_target=_strength_label(part.get("type"), params),
    )


def parse_ucf(ucf_path: str | Path) -> List[PartSpec]:
    """Parse a Cello UCF JSON file into a flat list of PartSpec objects.

    Each gate structure contributes one PartSpec per cassette component.
    The upstream_context and downstream_context fields hold the concatenated
    sequences of all preceding / following parts within the same cassette,
    making them ready for Evo 2 full-context conditioning.

    Args:
        ucf_path: Path to the UCF JSON file (e.g. Eco1C1G1T1.UCF.json).

    Returns:
        List of PartSpec, ordered by gate then cassette position.
    """
    ucf_path = Path(ucf_path)
    with open(ucf_path) as f:
        ucf_data = json.load(f)

    parts_map = _build_parts_map(ucf_data)
    host = _infer_host(ucf_data)
    structures = [x for x in ucf_data if x.get("collection") == "structures"]

    # Collect cassettes and pre-compute circuit_total
    all_cassettes: List[tuple[Dict[str, Any], List[str]]] = []
    for structure in structures:
        components = _get_cassette_components(structure)
        if components:
            all_cassettes.append((structure, components))

    circuit_total = sum(len(comps) for _, comps in all_cassettes)

    specs: List[PartSpec] = []
    global_pos = 0

    for structure, components in all_cassettes:
        gate_name = structure["name"].replace("_structure", "")

        # Resolve and clean sequences once per cassette for context building
        seqs = [
            _clean_seq(parts_map[name]["dnasequence"])
            if name in parts_map else ""
            for name in components
        ]

        for i, comp_name in enumerate(components):
            part = parts_map.get(comp_name)
            if part is None:
                global_pos += 1
                continue

            ucf_type = part.get("type", "")
            specs.append(PartSpec(
                part_id=f"{gate_name}__{comp_name}",
                part_type=_UCF_TYPE_TO_PART_TYPE.get(ucf_type, PartType.SPACER),
                host=host,
                functional_role=_UCF_TYPE_TO_ROLE.get(ucf_type, "unknown"),
                upstream_context="".join(seqs[:i]),
                downstream_context="".join(seqs[i + 1:]),
                reference_seq=_clean_seq(part["dnasequence"]),
                constraints=_build_constraints(part),
                circuit_position=global_pos,
                circuit_total=circuit_total,
                sbol_component_id=comp_name,
            ))
            global_pos += 1

    return specs
