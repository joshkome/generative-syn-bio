"""Tests for Stage 2: Cello UCF parser."""
import pytest
from pathlib import Path

from src.parser.cello_parser import parse_ucf, _infer_host, _strength_label
from src.schema.part_spec import PartSpec, PartType, HostOrganism

UCF_PATH = Path("data/raw/Eco1C1G1T1.UCF.json")


@pytest.fixture(scope="module")
def specs():
    return parse_ucf(UCF_PATH)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

def test_returns_list_of_part_specs(specs):
    assert isinstance(specs, list)
    assert len(specs) > 0
    assert all(isinstance(s, PartSpec) for s in specs)


def test_total_part_count(specs):
    # 20 gate structures × 4 parts per cassette = 80 part slots
    assert len(specs) == 80


def test_circuit_total_consistent(specs):
    assert all(s.circuit_total == 80 for s in specs)


def test_circuit_positions_are_unique_and_sequential(specs):
    positions = [s.circuit_position for s in specs]
    assert positions == list(range(80))


# ---------------------------------------------------------------------------
# Host inference
# ---------------------------------------------------------------------------

def test_host_is_ecoli_k12(specs):
    assert all(s.host == HostOrganism.ECOLI_K12 for s in specs)


# ---------------------------------------------------------------------------
# Part type mapping
# ---------------------------------------------------------------------------

def test_part_types_are_valid(specs):
    valid = set(PartType)
    assert all(s.part_type in valid for s in specs)


def test_known_cds_part_type(specs):
    amtr = next((s for s in specs if s.sbol_component_id == "AmtR"), None)
    assert amtr is not None
    assert amtr.part_type == PartType.CDS


def test_known_rbs_part_type(specs):
    rbs = next((s for s in specs if s.sbol_component_id == "A1"), None)
    assert rbs is not None
    assert rbs.part_type == PartType.RBS


def test_known_terminator_part_type(specs):
    term = next((s for s in specs if s.sbol_component_id == "L3S2P55"), None)
    assert term is not None
    assert term.part_type == PartType.TERMINATOR


def test_ribozyme_maps_to_spacer(specs):
    ribozymes = [s for s in specs if s.sbol_component_id in {"BydvJ", "SarJ"}]
    assert len(ribozymes) > 0
    assert all(s.part_type == PartType.SPACER for s in ribozymes)


# ---------------------------------------------------------------------------
# Sequence integrity
# ---------------------------------------------------------------------------

def test_reference_seq_uppercase_acgt(specs):
    for s in specs:
        assert s.reference_seq is not None
        assert s.reference_seq == s.reference_seq.upper()
        assert set(s.reference_seq) <= set("ACGT")


def test_context_uppercase_acgt(specs):
    for s in specs:
        for ctx in (s.upstream_context, s.downstream_context):
            assert ctx == ctx.upper()
            assert set(ctx) <= set("ACGT")


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def test_first_in_cassette_has_empty_upstream(specs):
    # Cassette order is [ribozyme, rbs, cds, terminator]
    # The ribozyme (position 0 in each cassette) has no upstream within cassette
    first_parts = specs[::4]  # every 4th starting at 0
    assert all(s.upstream_context == "" for s in first_parts)


def test_last_in_cassette_has_empty_downstream(specs):
    last_parts = specs[3::4]  # every 4th starting at 3
    assert all(s.downstream_context == "" for s in last_parts)


def test_middle_part_has_both_contexts(specs):
    # RBS is index 1 in each cassette — has upstream (ribozyme) and downstream
    rbs_parts = specs[1::4]
    assert all(s.upstream_context != "" for s in rbs_parts)
    assert all(s.downstream_context != "" for s in rbs_parts)


def test_context_does_not_include_own_seq(specs):
    for s in specs:
        ref = s.reference_seq or ""
        assert ref not in s.upstream_context
        assert ref not in s.downstream_context


# ---------------------------------------------------------------------------
# part_id and sbol_component_id
# ---------------------------------------------------------------------------

def test_part_id_contains_gate_and_component(specs):
    for s in specs:
        assert "__" in s.part_id
        gate, comp = s.part_id.split("__", 1)
        assert gate  # non-empty gate name
        assert comp == s.sbol_component_id


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_high_strength_terminator_constraint(specs):
    # L3S2P55 has terminator_strength=255.66 → "high"
    term = next(s for s in specs if s.sbol_component_id == "L3S2P55")
    assert term.constraints.strength_target == "high"


def test_high_efficiency_ribozyme_constraint(specs):
    # BydvJ has ribozyme_efficiency=0.95 → "high"
    bydvj = next(s for s in specs if s.sbol_component_id == "BydvJ")
    assert bydvj.constraints.strength_target == "high"


# ---------------------------------------------------------------------------
# Functional roles
# ---------------------------------------------------------------------------

def test_functional_roles_are_strings(specs):
    assert all(isinstance(s.functional_role, str) for s in specs)
    assert all(len(s.functional_role) > 0 for s in specs)


# ---------------------------------------------------------------------------
# Strength label helper (unit tests — no file I/O)
# ---------------------------------------------------------------------------

def test_strength_label_terminator():
    assert _strength_label("terminator", {"terminator_strength": 255.66}) == "high"
    assert _strength_label("terminator", {"terminator_strength": 50.0}) == "medium"
    assert _strength_label("terminator", {"terminator_strength": 1.0}) == "low"
    assert _strength_label("terminator", {}) is None


def test_strength_label_ribozyme():
    assert _strength_label("ribozyme", {"ribozyme_efficiency": 0.95}) == "high"
    assert _strength_label("ribozyme", {"ribozyme_efficiency": 0.7}) == "medium"
    assert _strength_label("ribozyme", {"ribozyme_efficiency": 0.3}) == "low"
    assert _strength_label("ribozyme", {}) is None


def test_strength_label_unknown_type():
    assert _strength_label("cds", {"foo": 99}) is None
    assert _strength_label(None, {}) is None
