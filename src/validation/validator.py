"""Stage 5: Circuit-level validation.

Takes pre-scored ranked candidates (from Stage 4) and produces a
CircuitReport summarising whether every part slot in a gate has at
least one viable candidate, along with CDS frame integrity checks.

The validator has no dependency on the Evo 2 model — it only consumes
the outputs of MultiObjectiveScorer.rank(), making it fully testable
without a GPU.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio.Seq import Seq

from src.schema.part_spec import PartType
from src.generation.types import GeneratedSequence, GenerationMode
from src.scoring.types import ScoreVector
from src.scoring.scorer import MultiObjectiveScorer


@dataclass
class ValidationResult:
    """Validation outcome for a single part slot."""
    part_id:               str
    part_type:             PartType
    has_passing_candidate: bool
    top_candidate:         Optional[GeneratedSequence]  # None if nothing passed filters
    top_score:             Optional[ScoreVector]
    cds_frame_valid:       Optional[bool]    # None for non-CDS parts
    reference_log_prob:    Optional[float]   # filled externally by run_pipeline.py

    def to_dict(self) -> dict:
        return {
            "part_id":               self.part_id,
            "part_type":             self.part_type.value,
            "has_passing_candidate": self.has_passing_candidate,
            "top_sequence":          self.top_candidate.sequence if self.top_candidate else None,
            "top_log_prob":          self.top_score.log_prob if self.top_score else None,
            "top_gc_content":        self.top_score.gc_content if self.top_score else None,
            "top_composite_score":   self.top_score.composite_score if self.top_score else None,
            "cds_frame_valid":       self.cds_frame_valid,
            "reference_log_prob":    self.reference_log_prob,
        }


@dataclass
class CircuitReport:
    """Validation summary for all part slots of one gate in one generation mode."""
    circuit_name:          str
    mode:                  GenerationMode
    n_candidates_per_part: int
    results:               List[ValidationResult] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True only when every part slot has at least one passing candidate."""
        return bool(self.results) and all(r.has_passing_candidate for r in self.results)

    def summary_dict(self) -> dict:
        n_pass = sum(1 for r in self.results if r.has_passing_candidate)
        return {
            "circuit_name":          self.circuit_name,
            "mode":                  self.mode.value,
            "n_candidates_per_part": self.n_candidates_per_part,
            "n_parts_total":         len(self.results),
            "n_parts_passing":       n_pass,
            "is_complete":           self.is_complete,
            "results":               [r.to_dict() for r in self.results],
        }

    def to_csv(self, path: Path) -> None:
        """Write one row per ValidationResult to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in self.results]
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


class CircuitValidator:
    """Validate ranked candidates against circuit-level requirements.

    Parameters
    ----------
    scorer : MultiObjectiveScorer, optional
        Used only if you call validate_circuit with unscored candidates.
        In normal usage, candidates come pre-ranked from Stage 4 and the
        scorer is not needed here.
    """

    def __init__(self, scorer: Optional[MultiObjectiveScorer] = None):
        self.scorer = scorer or MultiObjectiveScorer()

    def validate_circuit(
        self,
        ranked: List[Tuple[GeneratedSequence, ScoreVector]],
        circuit_name: str,
        mode: GenerationMode,
    ) -> CircuitReport:
        """Produce a CircuitReport from a pre-scored, ranked candidate list.

        The ranked list may contain multiple candidates per part_id (as
        produced by MultiObjectiveScorer.rank over all candidates for a gate).
        For each unique part_id, the first (highest-ranked) entry is taken as
        the top candidate.

        Parameters
        ----------
        ranked : output of MultiObjectiveScorer.rank() — already sorted
                 descending by composite_score, with .rank filled in.
        circuit_name : gate name, e.g. "A1_AmtR"
        mode : GenerationMode used to produce these candidates
        """
        # Group by part_id, preserving rank order (first = best)
        by_part: Dict[str, List[Tuple[GeneratedSequence, ScoreVector]]] = {}
        for candidate, sv in ranked:
            pid = candidate.part_spec.part_id
            by_part.setdefault(pid, []).append((candidate, sv))

        # Determine n_candidates_per_part from largest group
        n_candidates = max((len(v) for v in by_part.values()), default=0)

        results: List[ValidationResult] = []
        for part_id, entries in by_part.items():
            spec = entries[0][0].part_spec
            top_candidate, top_score = entries[0]

            has_passing = top_score.passes_hard_filters()

            cds_frame_valid: Optional[bool] = None
            if spec.part_type == PartType.CDS and has_passing:
                cds_frame_valid = self._check_cds_frame(top_candidate.sequence)

            results.append(ValidationResult(
                part_id=part_id,
                part_type=spec.part_type,
                has_passing_candidate=has_passing,
                top_candidate=top_candidate if has_passing else None,
                top_score=top_score if has_passing else None,
                cds_frame_valid=cds_frame_valid,
                reference_log_prob=None,
            ))

        # Sort results by circuit_position for stable output ordering
        results.sort(key=lambda r: r.top_candidate.part_spec.circuit_position
                     if r.top_candidate else float("inf"))

        return CircuitReport(
            circuit_name=circuit_name,
            mode=mode,
            n_candidates_per_part=n_candidates,
            results=results,
        )

    def _check_cds_frame(self, seq: str) -> bool:
        """Return True if seq is in-frame with no premature stop codons.

        Criteria:
        - Length must be a multiple of 3
        - No stop codon (*) in translated sequence except possibly at the very end
        """
        seq = seq.upper()
        if len(seq) % 3 != 0:
            return False
        protein = str(Seq(seq).translate(to_stop=False))
        # Stop codon in translated position before the final codon = premature
        internal_stops = protein[:-1].count("*")
        return internal_stops == 0
