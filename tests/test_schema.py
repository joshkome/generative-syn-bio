import pytest
from src.schema.part_spec import PartSpec, PartType, HostOrganism, PartConstraints


VALID_UPSTREAM   = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 40bp
VALID_DOWNSTREAM = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"  # 40bp


def make_test_spec(**kwargs):
    defaults = dict(
        part_id="test_part_001",
        part_type=PartType.PROMOTER,
        host=HostOrganism.ECOLI_K12,
        functional_role="test_promoter",
        upstream_context=VALID_UPSTREAM,
        downstream_context=VALID_DOWNSTREAM,
        circuit_position=0,
        circuit_total=4,
    )
    return PartSpec(**{**defaults, **kwargs})


def test_basic_instantiation():
    spec = make_test_spec()
    assert spec.part_id == "test_part_001"
    assert spec.part_type == PartType.PROMOTER


def test_dna_normalized_to_uppercase():
    spec = make_test_spec(upstream_context="acgtacgtacgtacgtacgtacgtacgtacgtacgtacgt")
    assert spec.upstream_context == VALID_UPSTREAM


def test_invalid_dna_raises():
    with pytest.raises(Exception):
        make_test_spec(upstream_context="ACGTNNN")   # N is invalid


def test_gc_content():
    spec = make_test_spec(reference_seq="GCGCGCGC")
    assert spec.gc_content() == 1.0
    spec2 = make_test_spec(reference_seq="ATATATAT")
    assert spec2.gc_content() == 0.0


def test_all_part_types():
    from src.schema.part_spec import PartType
    for pt in PartType:
        spec = make_test_spec(part_type=pt)
        assert spec.part_type == pt


def test_constraints_optional():
    spec = make_test_spec()
    assert spec.constraints.max_length is None
    spec2 = make_test_spec(
        constraints=PartConstraints(strength_target="high", max_length=200)
    )
    assert spec2.constraints.max_length == 200
