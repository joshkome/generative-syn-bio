"""Tests for Stage 4: ScoreVector, ScorerWeights, and MultiObjectiveScorer."""
import pytest
from src.schema.part_spec import PartSpec, PartType, HostOrganism, PartConstraints
from src.generation.types import GeneratedSequence, GenerationMode
from src.scoring.types import ScoreVector, ScorerWeights
from src.scoring.scorer import (
    MultiObjectiveScorer,
    _gc_fraction,
    _gc_soft_score,
    _normalize_logprob,
    _mfe_score,
    _mfe_soft_score,
    _sd_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UPSTREAM = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
DOWNSTREAM = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"
BALANCED_SEQ = "ATGCATGCATGCATGCATGCATGCATGCATGC"  # GC=0.5


def make_spec(part_type=PartType.CDS, ref_seq=None) -> PartSpec:
    return PartSpec(
        part_id="test__seq",
        part_type=part_type,
        host=HostOrganism.ECOLI_K12,
        functional_role="test",
        upstream_context=UPSTREAM,
        downstream_context=DOWNSTREAM,
        circuit_position=0,
        circuit_total=4,
        reference_seq=ref_seq or BALANCED_SEQ,
    )


def make_candidate(seq=BALANCED_SEQ, log_prob=-1.0, part_type=PartType.CDS, ref_seq=None) -> GeneratedSequence:
    return GeneratedSequence(
        sequence=seq,
        log_prob=log_prob,
        mode=GenerationMode.FULL_CONTEXT,
        part_spec=make_spec(part_type=part_type, ref_seq=ref_seq or seq),
    )


# ---------------------------------------------------------------------------
# ScorerWeights
# ---------------------------------------------------------------------------

def test_default_weights_sum_to_one():
    w = ScorerWeights()
    total = w.log_prob + w.gc + w.structure + w.rbs_proxy
    assert abs(total - 1.0) < 1e-9


def test_custom_weights():
    w = ScorerWeights(log_prob=0.8, gc=0.2, structure=0.0, rbs_proxy=0.0)
    assert w.log_prob == 0.8


# ---------------------------------------------------------------------------
# _gc_fraction
# ---------------------------------------------------------------------------

def test_gc_fraction_all_gc():
    assert _gc_fraction("GCGCGC") == 1.0


def test_gc_fraction_all_at():
    assert _gc_fraction("ATATAT") == 0.0


def test_gc_fraction_balanced():
    assert abs(_gc_fraction("ATGC") - 0.5) < 1e-9


def test_gc_fraction_empty():
    assert _gc_fraction("") == 0.0


# ---------------------------------------------------------------------------
# _gc_soft_score
# ---------------------------------------------------------------------------

def test_gc_soft_score_peak():
    assert _gc_soft_score(0.5) == 1.0


def test_gc_soft_score_extremes():
    assert _gc_soft_score(0.0) == 0.0
    assert _gc_soft_score(1.0) == 0.0


def test_gc_soft_score_monotone():
    scores = [_gc_soft_score(x / 10) for x in range(6)]  # 0.0 → 0.5
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# _normalize_logprob
# ---------------------------------------------------------------------------

def test_normalize_logprob_zero():
    assert _normalize_logprob(0.0) == 1.0


def test_normalize_logprob_floor():
    assert _normalize_logprob(-4.0) == 0.0


def test_normalize_logprob_clamp_positive():
    assert _normalize_logprob(5.0) == 1.0


def test_normalize_logprob_clamp_very_negative():
    assert _normalize_logprob(-100.0) == 0.0


def test_normalize_logprob_midpoint():
    assert abs(_normalize_logprob(-2.0) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _mfe_score (calls ViennaRNA — fast, no model needed)
# ---------------------------------------------------------------------------

def test_mfe_score_hairpin_is_negative():
    # Strong GC-rich hairpin should have very negative MFE
    hairpin = "GCGCGCGCTTTTGCGCGCGC"
    mfe, mfe_norm = _mfe_score(hairpin)
    assert mfe < 0.0
    assert mfe_norm < 0.0


def test_mfe_score_normalized_by_length():
    seq = "GCGCGCGCTTTTGCGCGCGC"
    mfe, mfe_norm = _mfe_score(seq)
    assert abs(mfe_norm - mfe / len(seq)) < 1e-6


def test_mfe_soft_score_structured():
    # mfe_normalized = -0.5 → score = 1.0
    assert _mfe_soft_score(-0.5) == 1.0


def test_mfe_soft_score_unstructured():
    # mfe_normalized = 0.0 → score = 0.0
    assert _mfe_soft_score(0.0) == 0.0


def test_mfe_soft_score_clamps():
    assert _mfe_soft_score(-10.0) == 1.0  # capped at 1.0


# ---------------------------------------------------------------------------
# _sd_score
# ---------------------------------------------------------------------------

def test_sd_score_strong_sd():
    # Contains AGGAGG
    rbs = "AAAGAGGAGAAATG"
    score = _sd_score(rbs)
    assert score > 0.0


def test_sd_score_no_sd():
    # No SD-like motifs
    score = _sd_score("CCCCCCCCCCCCCCCCCCCC")
    assert score == 0.0


def test_sd_score_empty():
    assert _sd_score("") == 0.0


def test_sd_score_capped_at_one():
    # Many hits — should not exceed 1.0
    score = _sd_score("AGGAGGAGGAGGAGGAGGAGGAGG")
    assert score <= 1.0


# ---------------------------------------------------------------------------
# ScoreVector
# ---------------------------------------------------------------------------

def make_score_vector(**kwargs) -> ScoreVector:
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


def test_passes_hard_filters_happy():
    sv = make_score_vector()
    assert sv.passes_hard_filters()


def test_passes_hard_filters_gc_out_of_range():
    sv = make_score_vector(gc_in_range=False)
    assert not sv.passes_hard_filters()


def test_passes_hard_filters_length_exceeded():
    sv = make_score_vector(length_delta=21)
    assert not sv.passes_hard_filters()


def test_to_dict_keys():
    sv = make_score_vector()
    d = sv.to_dict()
    expected = {"log_prob", "gc_content", "gc_in_range", "length_delta",
                "mfe", "mfe_normalized", "rbs_sd_score", "composite_score", "passes_filters"}
    assert set(d.keys()) == expected


def test_to_dict_passes_filters_reflects_hard_filters():
    sv = make_score_vector(gc_in_range=False)
    assert sv.to_dict()["passes_filters"] is False


# ---------------------------------------------------------------------------
# MultiObjectiveScorer.score()
# ---------------------------------------------------------------------------

def test_score_returns_score_vector():
    scorer = MultiObjectiveScorer()
    candidate = make_candidate()
    sv = scorer.score(candidate)
    assert isinstance(sv, ScoreVector)


def test_score_gc_content_correct():
    scorer = MultiObjectiveScorer()
    sv = scorer.score(make_candidate(seq="GCGCGCGC"))
    assert sv.gc_content == 1.0


def test_score_gc_out_of_range_flagged():
    scorer = MultiObjectiveScorer(gc_min=0.35, gc_max=0.65)
    sv = scorer.score(make_candidate(seq="GCGCGCGCGCGCGCGCGCGC"))  # GC=1.0
    assert not sv.gc_in_range


def test_score_length_delta_exact():
    seq = BALANCED_SEQ
    scorer = MultiObjectiveScorer()
    sv = scorer.score(make_candidate(seq=seq, ref_seq=seq))
    assert sv.length_delta == 0


def test_score_length_delta_mismatch():
    scorer = MultiObjectiveScorer()
    candidate = make_candidate(seq="ATGC", ref_seq="ATGCATGCATGC")  # delta = 8
    sv = scorer.score(candidate)
    assert sv.length_delta == 8


def test_score_mfe_populated_for_terminator():
    scorer = MultiObjectiveScorer()
    seq = "GCGCGCGCTTTTGCGCGCGC"
    candidate = make_candidate(seq=seq, part_type=PartType.TERMINATOR, ref_seq=seq)
    sv = scorer.score(candidate)
    assert sv.mfe is not None
    assert sv.mfe_normalized is not None


def test_score_mfe_none_for_cds():
    scorer = MultiObjectiveScorer()
    sv = scorer.score(make_candidate(part_type=PartType.CDS))
    assert sv.mfe is None


def test_score_rbs_sd_populated_for_rbs():
    scorer = MultiObjectiveScorer()
    seq = "AAAGAGGAGAAATG"
    candidate = make_candidate(seq=seq, part_type=PartType.RBS, ref_seq=seq)
    sv = scorer.score(candidate)
    assert sv.rbs_sd_score is not None


def test_score_rbs_sd_none_for_promoter():
    scorer = MultiObjectiveScorer()
    sv = scorer.score(make_candidate(part_type=PartType.PROMOTER))
    assert sv.rbs_sd_score is None


def test_composite_score_bounded():
    scorer = MultiObjectiveScorer()
    sv = scorer.score(make_candidate(log_prob=-1.0))
    assert 0.0 <= sv.composite_score <= 1.0


# ---------------------------------------------------------------------------
# MultiObjectiveScorer.rank()
# ---------------------------------------------------------------------------

def test_rank_sorts_by_composite_descending():
    scorer = MultiObjectiveScorer()
    # Higher log_prob → higher composite score
    low = make_candidate(log_prob=-3.0)
    high = make_candidate(log_prob=-0.1)
    ranked = scorer.rank([low, high], filter=False)
    assert ranked[0][0] is high
    assert ranked[1][0] is low


def test_rank_fills_rank_field():
    scorer = MultiObjectiveScorer()
    candidates = [make_candidate(log_prob=-1.0), make_candidate(log_prob=-2.0)]
    ranked = scorer.rank(candidates, filter=False)
    assert ranked[0][0].rank == 1
    assert ranked[1][0].rank == 2


def test_rank_filter_removes_gc_failures():
    scorer = MultiObjectiveScorer(gc_min=0.35, gc_max=0.65)
    bad_gc = make_candidate(seq="GCGCGCGCGCGCGCGCGCGC")  # GC=1.0
    good = make_candidate(seq=BALANCED_SEQ)
    ranked = scorer.rank([bad_gc, good], filter=True)
    assert len(ranked) == 1
    assert ranked[0][0] is good


def test_rank_no_filter_keeps_all():
    scorer = MultiObjectiveScorer(gc_min=0.35, gc_max=0.65)
    bad_gc = make_candidate(seq="GCGCGCGCGCGCGCGCGCGC")
    good = make_candidate(seq=BALANCED_SEQ)
    ranked = scorer.rank([bad_gc, good], filter=False)
    assert len(ranked) == 2


def test_rank_empty_input():
    scorer = MultiObjectiveScorer()
    assert scorer.rank([]) == []


def test_score_batch_length_matches_input():
    scorer = MultiObjectiveScorer()
    candidates = [make_candidate() for _ in range(5)]
    scores = scorer.score_batch(candidates)
    assert len(scores) == 5
    assert all(isinstance(sv, ScoreVector) for sv in scores)
