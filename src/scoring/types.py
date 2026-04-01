"""Stage 4 scoring types: ScoreVector and ScorerWeights."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ScorerWeights:
    """Relative weights for the composite score.

    log_prob is the thesis's primary signal. gc and structure serve as
    soft biophysical scores; rbs_proxy only activates for RBS parts.
    Weights need not sum to 1 — composite_score is computed as a
    weighted sum of 0–1 normalized subscores.
    """
    log_prob:  float = 0.6
    gc:        float = 0.2
    structure: float = 0.1
    rbs_proxy: float = 0.1


@dataclass
class ScoreVector:
    """All scoring axes for one generated candidate.

    Hard filters (GC range, length delta) gate entry into the ranked set.
    Soft scores (mfe, rbs_sd_score) contribute to composite_score but do
    not disqualify candidates on their own.
    """
    # --- Primary thesis signal ---
    log_prob: float          # mean per-token log-prob from Evo 2

    # --- Biophysical filters ---
    gc_content: float        # fraction 0.0–1.0
    gc_in_range: bool        # True when gc_min <= gc_content <= gc_max
    length_delta: int        # abs(len(seq) - len(reference)); 0 = exact

    # --- Structure scoring (TERMINATOR / SPACER only) ---
    mfe: Optional[float]            # ViennaRNA MFE in kcal/mol; None if skipped
    mfe_normalized: Optional[float] # mfe / len(seq); None if skipped

    # --- RBS scoring (RBS parts only) ---
    rbs_sd_score: Optional[float]   # Shine-Dalgarno complementarity 0–1; None if skipped

    # --- Aggregate ---
    composite_score: float   # weighted sum used for ranking

    # Hard-filter thresholds stored for reference / serialization
    _gc_min: float = field(default=0.35, repr=False, compare=False)
    _gc_max: float = field(default=0.65, repr=False, compare=False)
    _max_length_delta: int = field(default=20, repr=False, compare=False)

    def passes_hard_filters(self) -> bool:
        """Return True only if this candidate clears all hard filters."""
        return self.gc_in_range and self.length_delta <= self._max_length_delta

    def to_dict(self) -> dict:
        return {
            "log_prob":       self.log_prob,
            "gc_content":     self.gc_content,
            "gc_in_range":    self.gc_in_range,
            "length_delta":   self.length_delta,
            "mfe":            self.mfe,
            "mfe_normalized": self.mfe_normalized,
            "rbs_sd_score":   self.rbs_sd_score,
            "composite_score": self.composite_score,
            "passes_filters": self.passes_hard_filters(),
        }
