#!/usr/bin/env python3
"""Demo: scoring + validation pipeline without a GPU / Evo 2 model.

Builds synthetic-but-realistic candidates for a minimal NOT-gate circuit
(promoter → RBS → CDS → terminator), scores them with MultiObjectiveScorer,
and runs CircuitValidator.  Prints a formatted summary to stdout.

Run from project root:
    python scripts/demo_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.types import GeneratedSequence, GenerationMode
from src.schema.part_spec import PartSpec, PartType, HostOrganism, PartConstraints
from src.scoring.scorer import MultiObjectiveScorer
from src.validation.validator import CircuitValidator

# ---------------------------------------------------------------------------
# Synthetic circuit: A1_AmtR NOT gate (4 parts)
# ---------------------------------------------------------------------------

_UPSTREAM  = "AATTCGAGCTCGGTACCCGGGGATCCTCTAGAG"
_DOWNSTREAM = "CTCGAGGTCGACGGTATCGATAAGCTTGATATC"

PARTS = [
    PartSpec(
        part_id="A1_AmtR_promoter",
        part_type=PartType.PROMOTER,
        host=HostOrganism.ECOLI_K12,
        functional_role="AmtR-repressible promoter (pAmtR)",
        upstream_context=_UPSTREAM,
        downstream_context=_DOWNSTREAM,
        circuit_position=1,
        circuit_total=4,
        reference_seq="TTGACAATTAATCATCGGCTCGTATAATGTGTGGAATTGTGAGCGGATAACAATTTCACACA",
    ),
    PartSpec(
        part_id="A1_AmtR_rbs",
        part_type=PartType.RBS,
        host=HostOrganism.ECOLI_K12,
        functional_role="Ribosome binding site",
        upstream_context=_UPSTREAM,
        downstream_context=_DOWNSTREAM,
        circuit_position=2,
        circuit_total=4,
        reference_seq="AAAGAGGAGAAATACTA",
    ),
    PartSpec(
        part_id="A1_AmtR_cds",
        part_type=PartType.CDS,
        host=HostOrganism.ECOLI_K12,
        functional_role="AmtR repressor CDS",
        upstream_context=_UPSTREAM,
        downstream_context=_DOWNSTREAM,
        circuit_position=3,
        circuit_total=4,
        reference_seq="ATGAGCAAAGGAGAAGAACTTTTCACTGGA",  # 30 bp (10 codons, no stop)
    ),
    PartSpec(
        part_id="A1_AmtR_terminator",
        part_type=PartType.TERMINATOR,
        host=HostOrganism.ECOLI_K12,
        functional_role="Transcriptional terminator (ECK120029600)",
        upstream_context=_UPSTREAM,
        downstream_context=_DOWNSTREAM,
        circuit_position=4,
        circuit_total=4,
        reference_seq="CCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTG",
    ),
]

# ---------------------------------------------------------------------------
# Synthetic candidates — 3 per part, mimicking Evo 2 TAGGED_FULL output
# ---------------------------------------------------------------------------

def _make_candidates(part: PartSpec, seqs_and_lp: list) -> list:
    return [
        GeneratedSequence(
            sequence=seq,
            log_prob=lp,
            mode=GenerationMode.TAGGED_FULL,
            part_spec=part,
        )
        for seq, lp in seqs_and_lp
    ]


CANDIDATES = []

# Promoter candidates — ~60 bp, GC 40–55 %
CANDIDATES += _make_candidates(PARTS[0], [
    ("TTGACAATTAATCATCGGCTCGTATAATGTGTGGAATTGTGAGCGGATAACAATTTCACACA", -0.42),
    ("TTGACAGTTAATCATCGGCTCGTATAATGTGTGGAATAGCGAGCGGATAACAATTTCACACA", -0.61),
    ("TTGATAATTAACCATCAGCTCGTATAATGTGTGCAATTGCGAGCGGATAACAATTTCACACA", -0.89),  # will be filtered: GC too low
])

# RBS candidates — ~17 bp, SD motif present
CANDIDATES += _make_candidates(PARTS[1], [
    ("AAAGAGGAGAAATACTA", -0.31),
    ("AAAGAGGAGAAATACTG", -0.38),
    ("AAAGCGGAGAAATACTA", -0.55),
])

# CDS candidates — must be multiple of 3, no premature stop
CANDIDATES += _make_candidates(PARTS[2], [
    ("ATGAGCAAAGGAGAAGAACTTTTCACTGGA", -0.29),   # exact reference match
    ("ATGAGCAAAGGTGAAGAACTTTTCACTGGA", -0.44),   # synonymous G→T at pos 14
    ("ATGAGCAAAGGAGAAGAACTCTTCACAGGA", -0.51),
])

# Terminator candidates — ~58 bp, more negative MFE = better
CANDIDATES += _make_candidates(PARTS[3], [
    ("CCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTG", -0.37),
    ("CCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTT", -0.48),
    ("CCAGGCATCAAATGAAACGAAAGGCTCAGTCGAAAGACTGCGCCTTTCGCTTTATCTG", -0.62),
])

# ---------------------------------------------------------------------------
# Score + validate
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    scorer    = MultiObjectiveScorer()
    validator = CircuitValidator(scorer=scorer)

    ranked = scorer.rank(CANDIDATES)
    report = validator.validate_circuit(
        ranked,
        circuit_name="A1_AmtR",
        mode=GenerationMode.TAGGED_FULL,
    )

    summary = report.summary_dict()

    # ---- Header ----
    print()
    print("=" * 60)
    print("  Generative Syn-Bio Pipeline  —  Demo Validation")
    print("=" * 60)
    print(f"  Circuit   : {summary['circuit_name']}")
    print(f"  Mode      : {summary['mode']}")
    print(f"  Candidates: {summary['n_candidates_per_part']} per part")
    print(f"  Parts     : {summary['n_parts_passing']} / {summary['n_parts_total']} passing")
    print(f"  Complete  : {'YES' if summary['is_complete'] else 'NO'}")
    print()

    # ---- Per-part detail ----
    print(f"  {'Part ID':<28} {'Type':<11} {'Pass':<6} {'Score':<8} {'GC%':<7} {'LogP'}")
    print("  " + "-" * 56)
    for r in summary["results"]:
        passed  = "YES" if r["has_passing_candidate"] else " no"
        score   = f"{r['top_composite_score']:.3f}" if r["top_composite_score"] is not None else "  —  "
        gc      = f"{r['top_gc_content']*100:.1f}%" if r["top_gc_content"] is not None else "  —  "
        logp    = f"{r['top_log_prob']:.3f}" if r["top_log_prob"] is not None else "  —  "
        print(f"  {r['part_id']:<28} {r['part_type']:<11} {passed:<6} {score:<8} {gc:<7} {logp}")

    # ---- Top candidate sequences ----
    print()
    print("  Top candidate sequences")
    print("  " + "-" * 56)
    for r in summary["results"]:
        seq = r["top_sequence"]
        if seq:
            truncated = seq[:48] + ("…" if len(seq) > 48 else "")
            cds_note = ""
            if r["part_type"] == "cds":
                frame_ok = r.get("cds_frame_valid")
                cds_note = f"  [frame {'OK' if frame_ok else 'FAIL'}]"
            print(f"  {r['part_id']}")
            print(f"    {truncated}{cds_note}")
        else:
            print(f"  {r['part_id']}  — no passing candidate")
    print()

    # ---- Composite score bars ----
    print("  Composite score (0–1)")
    print("  " + "-" * 56)
    for r in summary["results"]:
        cs = r["top_composite_score"]
        bar = _bar(cs) if cs is not None else "[" + " " * 20 + "]"
        label = f"{cs:.3f}" if cs is not None else " n/a "
        print(f"  {r['part_id']:<28} {bar} {label}")

    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
