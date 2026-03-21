"""
Evo 2 generation interface.

Uses the ARC Institute evo2 package (pip install evo2).
Supports 4 generation modes for the context window ablation study.

Local dev:  uses evo2_1b_base (runs on Mac CPU/MPS, ~2GB)
Cloud runs: set EVO2_MODEL=arcinstitute/evo2_7b in .env
"""

import torch
import numpy as np
from typing import List, Optional
from src.schema.part_spec import PartSpec, PartType
from src.generation.types import GenerationMode, GeneratedSequence
from src.config import EVO2_MODEL_ID, DEFAULT_N_CANDIDATES, DEFAULT_TEMPERATURE, DEFAULT_TOP_K


# Functional tags prepended in TAGGED_FULL mode
PART_TYPE_TAGS = {
    PartType.PROMOTER:   "<promoter>",
    PartType.RBS:        "<rbs>",
    PartType.CDS:        "<cds>",
    PartType.TERMINATOR: "<terminator>",
    PartType.OPERATOR:   "<operator>",
    PartType.SPACER:     "<spacer>",
}

STRENGTH_TAGS = {
    "low":    "<strength_low>",
    "medium": "<strength_medium>",
    "high":   "<strength_high>",
}


class Evo2Generator:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or EVO2_MODEL_ID
        self.model = None  # lazy load — only download when generate() is first called
        print(f"Evo2Generator initialized (model: {self.model_id})")
        print("Model will be downloaded on first generate() call.")

    def _load_model(self):
        """Lazy load the model — avoids downloading on import."""
        if self.model is None:
            print(f"Loading {self.model_id} ... (this may take a few minutes on first run)")
            from evo2 import Evo2
            # evo2 package handles HF auth via huggingface_hub automatically
            self.model = Evo2(self.model_id.split("/")[-1])  # e.g. 'evo2_1b_base'
            print("Model loaded.")
        return self.model

    def build_prompt(self, spec: PartSpec, mode: GenerationMode) -> str:
        """
        Build the prompt string for Evo 2 based on generation mode.
        This is the core of the context window ablation — each mode
        provides a different amount of sequence context.
        """
        upstream   = spec.upstream_context
        downstream = spec.downstream_context

        if mode == GenerationMode.NO_CONTEXT:
            # Condition A: no flanking context, just a seed
            # Use first 4bp of upstream as minimal seed
            return upstream[-4:] if upstream else "ACGT"

        elif mode == GenerationMode.UPSTREAM_ONLY:
            # Condition B: upstream context only
            return upstream

        elif mode == GenerationMode.FULL_CONTEXT:
            # Condition C: upstream context as prompt
            # downstream context is used for log-prob scoring only (not generation)
            return upstream

        elif mode == GenerationMode.TAGGED_FULL:
            # Condition D: semantic tags + upstream context
            tag = PART_TYPE_TAGS.get(spec.part_type, "")
            strength_tag = ""
            if spec.constraints.strength_target:
                strength_tag = STRENGTH_TAGS.get(spec.constraints.strength_target, "")
            host_tag = f"<{spec.host.value}>"
            # Prepend tags to upstream context as a few-shot hint
            return f"{host_tag}{tag}{strength_tag}{upstream}"

        else:
            raise ValueError(f"Unknown generation mode: {mode}")

    def generate(
        self,
        spec: PartSpec,
        mode: GenerationMode = GenerationMode.FULL_CONTEXT,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[GeneratedSequence]:
        """
        Generate n_candidates sequences for a given PartSpec.
        Returns GeneratedSequence objects with log-prob scores.
        """
        model = self._load_model()

        # Determine target length from reference or constraints
        if spec.reference_seq:
            n_tokens = len(spec.reference_seq)
        elif spec.constraints.max_length:
            n_tokens = spec.constraints.max_length
        else:
            # Default lengths by part type
            defaults = {
                PartType.PROMOTER:   100,
                PartType.RBS:        30,
                PartType.CDS:        600,
                PartType.TERMINATOR: 80,
            }
            n_tokens = defaults.get(spec.part_type, 150)

        prompt = self.build_prompt(spec, mode)
        results = []

        for i in range(n_candidates):
            output = model.generate(
                prompt_seqs=[prompt],
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            seq = output.sequences[0]

            # Strip the prompt prefix from the output
            if seq.startswith(prompt):
                seq = seq[len(prompt):]

            # Compute log-prob of this sequence given full context
            log_prob = self.score_sequence(seq, spec, mode)

            results.append(GeneratedSequence(
                sequence  = seq.upper(),
                log_prob  = log_prob,
                mode      = mode,
                part_spec = spec,
            ))

        return results

    def score_sequence(
        self,
        sequence: str,
        spec: PartSpec,
        mode: GenerationMode = GenerationMode.FULL_CONTEXT,
    ) -> float:
        """
        Compute log P(sequence | context) under Evo 2.
        This is the primary metric for the ablation study.

        For NO_CONTEXT: score sequence with minimal context
        For UPSTREAM_ONLY: score with upstream context prepended
        For FULL_CONTEXT / TAGGED_FULL: score with full flanking context
        """
        model = self._load_model()

        if mode == GenerationMode.NO_CONTEXT:
            context_seq = sequence
        elif mode == GenerationMode.UPSTREAM_ONLY:
            context_seq = spec.upstream_context + sequence
        else:
            # Full context: upstream + sequence + downstream
            context_seq = spec.upstream_context + sequence + spec.downstream_context

        # Use the model's built-in scoring
        # evo2 package scores via log-likelihood of the full sequence
        score_output = model.score_sequences(sequences=[context_seq])

        if score_output and len(score_output) > 0:
            full_logprob = float(score_output[0])
            # Normalize by sequence length to get per-token log-prob
            return full_logprob / max(len(sequence), 1)
        return 0.0
