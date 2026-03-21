from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from src.schema.part_spec import PartSpec


class GenerationMode(str, Enum):
    NO_CONTEXT    = "no_context"     # Condition A — baseline (no flanking seq)
    UPSTREAM_ONLY = "upstream_only"  # Condition B — upstream context only
    FULL_CONTEXT  = "full_context"   # Condition C — full flanking context
    TAGGED_FULL   = "tagged_full"    # Condition D — tags + full context


@dataclass
class GeneratedSequence:
    sequence:   str
    log_prob:   float                # mean log-prob over sequence tokens
    mode:       GenerationMode
    part_spec:  PartSpec
    rank:       Optional[int] = None # filled in after scoring

    def __len__(self):
        return len(self.sequence)

    def gc_content(self) -> float:
        seq = self.sequence.upper()
        if not seq:
            return 0.0
        return (seq.count("G") + seq.count("C")) / len(seq)
