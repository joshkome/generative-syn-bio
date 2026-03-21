from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class PartType(str, Enum):
    PROMOTER   = "promoter"
    RBS        = "rbs"
    CDS        = "cds"
    TERMINATOR = "terminator"
    OPERATOR   = "operator"
    SPACER     = "spacer"


class HostOrganism(str, Enum):
    ECOLI_K12     = "ecoli_k12"
    ECOLI_BL21    = "ecoli_bl21"
    SACCHAROMYCES = "saccharomyces_cerevisiae"


class PartConstraints(BaseModel):
    min_length:        Optional[int]   = None
    max_length:        Optional[int]   = None
    strength_target:   Optional[str]   = None   # 'low' | 'medium' | 'high'
    inducible_by:      Optional[str]   = None
    repressible_by:    Optional[str]   = None
    gc_content_min:    Optional[float] = None   # 0.0 – 1.0
    gc_content_max:    Optional[float] = None


class PartSpec(BaseModel):
    part_id:            str
    part_type:          PartType
    host:               HostOrganism
    functional_role:    str
    upstream_context:   str
    downstream_context: str
    constraints:        PartConstraints = Field(default_factory=PartConstraints)
    circuit_position:   int
    circuit_total:      int
    reference_seq:      Optional[str] = None
    sbol_component_id:  Optional[str] = None

    @field_validator("upstream_context", "downstream_context", "reference_seq", mode="before")
    @classmethod
    def validate_and_normalize_dna(cls, v):
        if v is None:
            return v
        v = v.upper().strip()
        invalid = set(v) - set("ACGT")
        if invalid:
            raise ValueError(f"Invalid DNA characters: {invalid}")
        return v

    def gc_content(self, sequence: Optional[str] = None) -> float:
        seq = sequence or self.reference_seq or ""
        if not seq:
            return 0.0
        return (seq.count("G") + seq.count("C")) / len(seq)
