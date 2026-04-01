"""Shared pytest fixtures used across multiple test modules."""
from pathlib import Path
from typing import List, Tuple

import pytest

from src.generation.types import GeneratedSequence, GenerationMode
from src.parser.cello_parser import parse_ucf
from src.schema.part_spec import PartSpec
from src.scoring.scorer import MultiObjectiveScorer
from src.scoring.types import ScoreVector

UCF_PATH = Path("data/raw/Eco1C1G1T1.UCF.json")


@pytest.fixture(scope="session")
def ucf_specs() -> List[PartSpec]:
    """All 80 PartSpecs parsed from Eco1C1G1T1.UCF.json (parsed once per session)."""
    return parse_ucf(UCF_PATH)


@pytest.fixture(scope="session")
def amtr_specs(ucf_specs: List[PartSpec]) -> List[PartSpec]:
    """The 4 part slots of the A1_AmtR gate cassette (ribozyme, rbs, cds, terminator)."""
    return [s for s in ucf_specs if s.part_id.startswith("A1_AmtR__")]


@pytest.fixture
def mock_ranked(amtr_specs: List[PartSpec]) -> List[Tuple[GeneratedSequence, ScoreVector]]:
    """Pre-scored candidates using each part's reference_seq as the 'generated' sequence.

    Provides deterministic, GPU-free input for Stage 5 tests. log_prob is set
    to -1.0 (a plausible mid-range value) so composite scores are non-trivial.
    """
    scorer = MultiObjectiveScorer()
    candidates = [
        GeneratedSequence(
            sequence=spec.reference_seq,
            log_prob=-1.0,
            mode=GenerationMode.FULL_CONTEXT,
            part_spec=spec,
        )
        for spec in amtr_specs
        if spec.reference_seq  # all UCF parts have reference seqs
    ]
    return scorer.rank(candidates, filter=False)
