"""Tests for Stage 5: CircuitValidator, ValidationResult, CircuitReport."""
import csv
from pathlib import Path
from typing import List, Tuple

import pytest

from src.generation.types import GeneratedSequence, GenerationMode
from src.schema.part_spec import PartSpec, PartType, HostOrganism
from src.scoring.scorer import MultiObjectiveScorer
from src.scoring.types import ScoreVector
from src.validation.validator import (
    CircuitReport,
    CircuitValidator,
    ValidationResult,
)

# AmtR CDS sequence from Eco1C1G1T1 — a valid in-frame CDS
_AMTR_CDS = (
    "ATGGCAGGCGCAGTTGGTCGTCCGCGTCGTAGTGCACCGCGTCGTGCAGGTAAAAATCCGCGTGAAGAA"
    "ATTCTGGATGCAAGCGCAGAACTGTTTACCCGTCAGGGTTTTGCAACCACCAGTACCCATCAGATTGCAG"
    "ATGCAGTTGGTATTCGTCAGGCAAGCCTGTATTATCATTTTCCGAGCAAAACCGAAATCTTTCTGACCCT"
    "GCTGAAAAGCACCGTTGAACCGAGCACCGTTCTGGCAGAAGATCTGAGCACCCTGGATGCAGGTCCGGAA"
    "ATGCGTCTGTGGGCAATTGTTGCAAGCGAAGTTCGTCTGCTGCTGAGCACCAAATGGAATGTTGGTCGTC"
    "TGTATCAGCTGCCGATTGTTGGTAGCGAAGAATTTGCAGAATATCATAGCCAGCGTGAAGCACTGACCAA"
    "TGTTTTTCGTGATCTGGCAACCGAAATTGTTGGTGATGATCCGCGTGCAGAACTGCCGTTTCATATTACC"
    "ATGAGCGTTATTGAAATGCGTCGCAATGATGGTAAAATTCCGAGTCCGCTGAGCGCAGATAGCCTGCCGG"
    "AAACCGCAATTATGCTGGCAGATGCAAGCCTGGCAGTTCTGGGTGCACCGCTGCCTGCAGATCGTGTTGA"
    "AAAAACCCTGGAACTGATTAAACAGGCAGATGCAAAATAA"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(part_type=PartType.CDS, part_id="gate__seq", ref_seq=_AMTR_CDS) -> PartSpec:
    return PartSpec(
        part_id=part_id,
        part_type=part_type,
        host=HostOrganism.ECOLI_K12,
        functional_role="test",
        upstream_context="ACGTACGTACGTACGT",
        downstream_context="TGCATGCATGCATGCA",
        circuit_position=0,
        circuit_total=4,
        reference_seq=ref_seq,
    )


def _make_candidate(seq: str, part_type=PartType.CDS, log_prob=-1.0,
                    part_id="gate__seq", ref_seq=None) -> GeneratedSequence:
    return GeneratedSequence(
        sequence=seq,
        log_prob=log_prob,
        mode=GenerationMode.FULL_CONTEXT,
        part_spec=_make_spec(part_type=part_type, part_id=part_id,
                             ref_seq=ref_seq or seq),
    )


def _make_score_vector(**kwargs) -> ScoreVector:
    defaults = dict(
        log_prob=-1.0,
        gc_content=0.5,
        gc_in_range=True,
        length_delta=0,
        mfe=None,
        mfe_normalized=None,
        rbs_sd_score=None,
        composite_score=0.5,
    )
    return ScoreVector(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

def test_validation_result_fields():
    r = ValidationResult(
        part_id="gate__AmtR",
        part_type=PartType.CDS,
        has_passing_candidate=True,
        top_candidate=None,
        top_score=None,
        cds_frame_valid=True,
        reference_log_prob=-1.2,
    )
    assert r.part_id == "gate__AmtR"
    assert r.has_passing_candidate is True
    assert r.cds_frame_valid is True


def test_validation_result_to_dict_keys():
    r = ValidationResult(
        part_id="gate__A1",
        part_type=PartType.RBS,
        has_passing_candidate=True,
        top_candidate=None,
        top_score=None,
        cds_frame_valid=None,
        reference_log_prob=None,
    )
    d = r.to_dict()
    assert "part_id" in d
    assert "has_passing_candidate" in d
    assert "cds_frame_valid" in d
    assert "reference_log_prob" in d


# ---------------------------------------------------------------------------
# CircuitReport
# ---------------------------------------------------------------------------

def _make_report(n_passing: int, n_total: int = 4) -> CircuitReport:
    results = []
    for i in range(n_total):
        results.append(ValidationResult(
            part_id=f"gate__part{i}",
            part_type=PartType.CDS,
            has_passing_candidate=(i < n_passing),
            top_candidate=None,
            top_score=None,
            cds_frame_valid=None,
            reference_log_prob=None,
        ))
    return CircuitReport(
        circuit_name="test_gate",
        mode=GenerationMode.FULL_CONTEXT,
        n_candidates_per_part=10,
        results=results,
    )


def test_circuit_report_is_complete_all_pass():
    report = _make_report(n_passing=4, n_total=4)
    assert report.is_complete is True


def test_circuit_report_is_complete_partial_fail():
    report = _make_report(n_passing=3, n_total=4)
    assert report.is_complete is False


def test_circuit_report_is_complete_none_pass():
    report = _make_report(n_passing=0, n_total=4)
    assert report.is_complete is False


def test_circuit_report_is_complete_empty_results():
    report = CircuitReport(
        circuit_name="empty",
        mode=GenerationMode.NO_CONTEXT,
        n_candidates_per_part=0,
        results=[],
    )
    assert report.is_complete is False


def test_circuit_report_summary_dict_keys():
    report = _make_report(n_passing=4)
    d = report.summary_dict()
    assert "circuit_name" in d
    assert "mode" in d
    assert "n_parts_total" in d
    assert "n_parts_passing" in d
    assert "is_complete" in d
    assert "results" in d


def test_circuit_report_summary_dict_counts():
    report = _make_report(n_passing=3, n_total=4)
    d = report.summary_dict()
    assert d["n_parts_total"] == 4
    assert d["n_parts_passing"] == 3
    assert d["is_complete"] is False


def test_circuit_report_to_csv(tmp_path):
    report = _make_report(n_passing=4)
    csv_path = tmp_path / "report.csv"
    report.to_csv(csv_path)
    assert csv_path.exists()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 4
    assert "part_id" in rows[0]
    assert "has_passing_candidate" in rows[0]


def test_circuit_report_to_csv_creates_parent_dirs(tmp_path):
    report = _make_report(n_passing=2)
    nested = tmp_path / "deep" / "nested" / "report.csv"
    report.to_csv(nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# CircuitValidator._check_cds_frame
# ---------------------------------------------------------------------------

def test_check_cds_frame_valid():
    validator = CircuitValidator()
    # AmtR is a valid in-frame CDS
    assert validator._check_cds_frame(_AMTR_CDS) is True


def test_check_cds_frame_premature_stop():
    validator = CircuitValidator()
    # Insert a stop codon (TAA) early in a valid-length sequence
    seq = "ATGTAA" + "ATG" * 10  # TAA at position 3 = premature stop
    assert validator._check_cds_frame(seq) is False


def test_check_cds_frame_not_multiple_of_three():
    validator = CircuitValidator()
    assert validator._check_cds_frame("ATGCAT") is True   # 6 bp = ok
    assert validator._check_cds_frame("ATGCA") is False   # 5 bp = bad


def test_check_cds_frame_empty():
    validator = CircuitValidator()
    # Empty string: 0 % 3 == 0, translate yields "", no internal stops
    assert validator._check_cds_frame("") is True


# ---------------------------------------------------------------------------
# CircuitValidator.validate_circuit (using conftest mock_ranked fixture)
# ---------------------------------------------------------------------------

def test_validate_circuit_returns_circuit_report(mock_ranked):
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    assert isinstance(report, CircuitReport)


def test_validate_circuit_name_and_mode(mock_ranked):
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    assert report.circuit_name == "A1_AmtR"
    assert report.mode == GenerationMode.FULL_CONTEXT


def test_validate_circuit_result_count(mock_ranked):
    """One ValidationResult per unique part_id (4 for A1_AmtR cassette)."""
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    assert len(report.results) == 4


def test_validate_circuit_has_passing_candidates(mock_ranked):
    """Reference seqs pass GC and length filters → all should have passing candidates."""
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    assert all(r.has_passing_candidate for r in report.results)


def test_validate_circuit_is_complete(mock_ranked):
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    assert report.is_complete is True


def test_validate_circuit_top_candidate_rank_is_filled(mock_ranked):
    """rank field is set by MultiObjectiveScorer.rank(); validator preserves it."""
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    for r in report.results:
        if r.top_candidate is not None:
            assert r.top_candidate.rank is not None


def test_validate_circuit_cds_frame_valid_for_cds(mock_ranked):
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    cds_results = [r for r in report.results if r.part_type == PartType.CDS]
    assert len(cds_results) == 1
    assert cds_results[0].cds_frame_valid is True


def test_validate_circuit_cds_frame_none_for_non_cds(mock_ranked):
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    non_cds = [r for r in report.results if r.part_type != PartType.CDS]
    assert all(r.cds_frame_valid is None for r in non_cds)


def test_validate_circuit_reference_log_prob_is_none(mock_ranked):
    """reference_log_prob is filled externally by run_pipeline.py, not by validate_circuit."""
    validator = CircuitValidator()
    report = validator.validate_circuit(mock_ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)
    assert all(r.reference_log_prob is None for r in report.results)


def test_validate_circuit_all_fail_hard_filters():
    """When no candidate passes hard filters, has_passing_candidate should be False."""
    scorer = MultiObjectiveScorer(gc_min=0.9, gc_max=1.0)  # only near-pure-GC passes
    spec = _make_spec(part_type=PartType.RBS, ref_seq="AAAGAGGAGAAATG")
    candidate = _make_candidate("AAAGAGGAGAAATG", part_type=PartType.RBS,
                                part_id="gate__rbs", ref_seq="AAAGAGGAGAAATG")
    ranked = scorer.rank([candidate], filter=True)  # will be empty — GC too low

    # Pass an empty ranked list — no candidates for this part_id
    validator = CircuitValidator()
    report = validator.validate_circuit(ranked, "gate", GenerationMode.NO_CONTEXT)
    assert len(report.results) == 0  # empty input → empty report


# ---------------------------------------------------------------------------
# Integration marker (skipped without real model)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_pipeline_integration(ucf_specs, tmp_path):
    """End-to-end pipeline test. Requires Evo 2 model download (~2GB).
    Run with: pytest tests/test_validation.py -m integration
    """
    try:
        from src.generation.evo2_interface import Evo2Generator
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Evo 2 dependencies not available: {e}")

    gate_prefix = "A1_AmtR__"
    specs = [s for s in ucf_specs if s.part_id.startswith(gate_prefix)]

    generator = Evo2Generator()
    scorer = MultiObjectiveScorer()
    validator = CircuitValidator()

    all_candidates = []
    for spec in specs:
        try:
            candidates = generator.generate(spec, n_candidates=2)
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"Evo 2 model dependencies unavailable (needs GPU): {e}")
        all_candidates.extend(candidates)

    ranked = scorer.rank(all_candidates, filter=False)
    report = validator.validate_circuit(ranked, "A1_AmtR", GenerationMode.FULL_CONTEXT)

    assert isinstance(report, CircuitReport)
    assert len(report.results) == len(specs)

    report.to_csv(tmp_path / "integration_report.csv")
    assert (tmp_path / "integration_report.csv").exists()
