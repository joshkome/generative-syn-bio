#!/usr/bin/env python3
"""Ablation study runner: compare all 4 generation modes on a single gate.

For each GenerationMode × each part slot in the chosen gate, generates
n_candidates sequences, scores them (without hard-filtering so all candidates
appear in the output), and saves a comparison CSV.

The per-mode mean log_prob columns form the basis of the thesis ablation table:
  NO_CONTEXT → UPSTREAM_ONLY → FULL_CONTEXT → TAGGED_FULL

Usage
-----
    python scripts/run_ablation.py \\
        --ucf data/raw/Eco1C1G1T1.UCF.json \\
        --gate A1_AmtR \\
        --n-candidates 10 \\
        --output results/ablation
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEFAULT_N_CANDIDATES, DEFAULT_TEMPERATURE, DEFAULT_TOP_K
from src.generation.evo2_interface import Evo2Generator
from src.generation.types import GenerationMode
from src.parser.cello_parser import parse_ucf
from src.scoring.scorer import MultiObjectiveScorer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the 4-mode ablation study for one gate."
    )
    p.add_argument("--ucf", required=True, help="Path to UCF JSON file")
    p.add_argument("--gate", required=True,
                   help="Gate name prefix, e.g. A1_AmtR")
    p.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES,
                   help=f"Candidates per part per mode (default: {DEFAULT_N_CANDIDATES})")
    p.add_argument("--output", default="results/ablation",
                   help="Output directory (created if needed)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return p


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_prefix = args.gate + "__"

    # --- Parse UCF ---
    print(f"Parsing UCF: {args.ucf}")
    all_specs = parse_ucf(args.ucf)
    specs = [s for s in all_specs if s.part_id.startswith(gate_prefix)]
    if not specs:
        print(f"ERROR: No parts found for gate '{args.gate}'.")
        sys.exit(1)
    print(f"Gate '{args.gate}': {len(specs)} part(s)\n")

    generator = Evo2Generator()
    scorer = MultiObjectiveScorer()
    rows = []

    for mode in GenerationMode:
        print(f"Mode: {mode.value}")
        for spec in specs:
            candidates = generator.generate(
                spec,
                mode=mode,
                n_candidates=args.n_candidates,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            # Score without hard-filter so every candidate appears in the CSV
            score_vectors = scorer.score_batch(candidates)
            for candidate, sv in zip(candidates, score_vectors):
                rows.append({
                    "mode":            mode.value,
                    "part_id":         spec.part_id,
                    "part_type":       spec.part_type.value,
                    "log_prob":        candidate.log_prob,
                    "gc_content":      sv.gc_content,
                    "gc_in_range":     sv.gc_in_range,
                    "length_delta":    sv.length_delta,
                    "mfe":             sv.mfe,
                    "rbs_sd_score":    sv.rbs_sd_score,
                    "composite_score": sv.composite_score,
                })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "ablation_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows → {csv_path}")

    # Print per-mode log_prob summary (thesis Table 1 preview)
    print("\n=== Log-prob by mode (mean ± std) ===")
    summary = df.groupby("mode")["log_prob"].agg(["mean", "std"]).reindex(
        [m.value for m in GenerationMode]
    )
    print(summary.to_string())
    summary.to_csv(output_dir / "ablation_summary.csv")


if __name__ == "__main__":
    main(build_parser().parse_args())
