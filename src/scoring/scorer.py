"""Stage 4: MultiObjectiveScorer.

Scores GeneratedSequence candidates on multiple biophysical axes and
returns a ranked list ready for Stage 5 validation.

Scoring axes
------------
1. log_prob      (primary)  — Evo 2 per-token log-prob, already on GeneratedSequence
2. gc_content    (filter)   — hard-filter at [gc_min, gc_max]; soft score peaks at 0.5
3. length_delta  (filter)   — hard-filter at max_length_delta bp from reference
4. mfe           (soft)     — ViennaRNA MFE for TERMINATOR / SPACER parts only
5. rbs_sd_score  (soft)     — Shine-Dalgarno complementarity proxy for RBS parts only
"""
import re
from typing import List, Optional, Tuple

from src.schema.part_spec import PartType
from src.generation.types import GeneratedSequence
from src.scoring.types import ScoreVector, ScorerWeights

# Shine-Dalgarno consensus: AGGAGG and close relatives
# We search for 4-mers with >= 3 matches to the AGGAGG hexamer core
_SD_HEXAMER = "AGGAGG"
_SD_4MER_PATTERNS = re.compile(
    r"(?:AGGA|GGAG|GAGG|AGGG|AAGG|AGAG)",
    re.IGNORECASE,
)

# Part types that benefit from secondary-structure scoring
_STRUCTURED_TYPES = {PartType.TERMINATOR, PartType.SPACER}


# ---------------------------------------------------------------------------
# Internal sub-scorers
# ---------------------------------------------------------------------------

def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def _gc_soft_score(gc: float) -> float:
    """Unimodal score peaking at gc=0.5, falling linearly to 0 at 0 or 1."""
    return 1.0 - abs(gc - 0.5) * 2.0


def _normalize_logprob(log_prob: float) -> float:
    """Map per-token log-prob to [0, 1].

    Typical Evo 2 per-token log-probs sit in [-2, 0]; we clip and rescale.
    Positive values (extremely confident) are capped at 1.0.
    """
    # Clamp to a reasonable range for DNA language models
    clamped = max(-4.0, min(0.0, log_prob))
    return 1.0 + clamped / 4.0  # maps [-4, 0] → [0, 1]


def _mfe_score(seq: str) -> Tuple[float, float]:
    """Return (mfe_kcal, mfe_normalized) via ViennaRNA fold.

    More negative MFE = more stable structure. We return the raw value
    and length-normalized value; interpretation is caller's responsibility.
    """
    import RNA  # imported here so the module loads without ViennaRNA in test envs
    _, mfe = RNA.fold(seq)
    return float(mfe), float(mfe) / len(seq) if seq else 0.0


def _mfe_soft_score(mfe_normalized: float) -> float:
    """Convert normalized MFE to a [0, 1] score.

    For terminators / ribozymes, more negative MFE is better. We map
    [-0.5, 0.0] kcal/mol/nt → [1.0, 0.0]. Values beyond -0.5 are capped.
    """
    # mfe_normalized is <= 0; more negative = more structured = better
    clamped = max(-0.5, min(0.0, mfe_normalized))
    return abs(clamped) / 0.5


def _sd_score(seq: str) -> float:
    """Shine-Dalgarno complementarity proxy for RBS sequences.

    Searches the full RBS sequence for SD-like 4-mers (AGGA, GGAG, …).
    Returns fraction of possible SD sites that match, normalized to [0, 1].
    """
    if not seq:
        return 0.0
    matches = len(_SD_4MER_PATTERNS.findall(seq))
    # Normalize: expect ~0–3 hits in a typical 30 bp RBS
    return min(1.0, matches / 3.0)


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------

class MultiObjectiveScorer:
    """Score and rank GeneratedSequence candidates.

    Parameters
    ----------
    weights : ScorerWeights
        Relative weights for composite score computation.
    gc_min, gc_max : float
        Hard GC content bounds (inclusive). Default 35–65 % for E. coli.
    max_length_delta : int
        Hard limit on absolute length deviation from reference_seq.
        Candidates exceeding this fail passes_hard_filters().
    """

    def __init__(
        self,
        weights: ScorerWeights = None,
        gc_min: float = 0.35,
        gc_max: float = 0.65,
        max_length_delta: int = 20,
    ):
        self.weights = weights or ScorerWeights()
        self.gc_min = gc_min
        self.gc_max = gc_max
        self.max_length_delta = max_length_delta

    def score(self, candidate: GeneratedSequence) -> ScoreVector:
        """Score a single candidate and return its ScoreVector."""
        seq = candidate.sequence.upper()
        spec = candidate.part_spec

        # --- GC ---
        gc = _gc_fraction(seq)
        gc_in_range = self.gc_min <= gc <= self.gc_max

        # --- Length ---
        ref_len = len(spec.reference_seq) if spec.reference_seq else len(seq)
        length_delta = abs(len(seq) - ref_len)

        # --- Structure (TERMINATOR / SPACER only) ---
        mfe: Optional[float] = None
        mfe_norm: Optional[float] = None
        mfe_soft: float = 0.0
        if spec.part_type in _STRUCTURED_TYPES and seq:
            mfe, mfe_norm = _mfe_score(seq)
            mfe_soft = _mfe_soft_score(mfe_norm)

        # --- RBS proxy ---
        sd: Optional[float] = None
        sd_soft: float = 0.0
        if spec.part_type == PartType.RBS and seq:
            sd = _sd_score(seq)
            sd_soft = sd

        # --- Composite ---
        w = self.weights
        composite = (
            w.log_prob  * _normalize_logprob(candidate.log_prob)
            + w.gc      * _gc_soft_score(gc)
            + w.structure * mfe_soft
            + w.rbs_proxy * sd_soft
        )

        return ScoreVector(
            log_prob=candidate.log_prob,
            gc_content=gc,
            gc_in_range=gc_in_range,
            length_delta=length_delta,
            mfe=mfe,
            mfe_normalized=mfe_norm,
            rbs_sd_score=sd,
            composite_score=composite,
            _gc_min=self.gc_min,
            _gc_max=self.gc_max,
            _max_length_delta=self.max_length_delta,
        )

    def score_batch(
        self, candidates: List[GeneratedSequence]
    ) -> List[ScoreVector]:
        """Score all candidates. Order matches input list."""
        return [self.score(c) for c in candidates]

    def rank(
        self,
        candidates: List[GeneratedSequence],
        *,
        filter: bool = True,
    ) -> List[Tuple[GeneratedSequence, ScoreVector]]:
        """Score, optionally filter, and sort candidates by composite_score.

        Parameters
        ----------
        candidates : list of GeneratedSequence
        filter : bool
            If True (default), drop candidates that fail passes_hard_filters().
            Pass False to include all candidates (useful for ablation analysis).

        Returns
        -------
        List of (GeneratedSequence, ScoreVector) sorted descending by
        composite_score, with GeneratedSequence.rank filled in (1-indexed).
        """
        scored = [(c, self.score(c)) for c in candidates]

        if filter:
            scored = [(c, sv) for c, sv in scored if sv.passes_hard_filters()]

        scored.sort(key=lambda pair: pair[1].composite_score, reverse=True)

        for i, (candidate, _) in enumerate(scored):
            candidate.rank = i + 1

        return scored
