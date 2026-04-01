#!/usr/bin/env python3
"""Stage 5 pipeline runner: UCF → generate → score → validate → report.

Runs one gate × one generation mode end-to-end and writes a CircuitReport.
Requires the Evo 2 model (downloaded on first run, ~2 GB for evo2_1b_base).

Usage
-----
    python scripts/run_pipeline.py \\
        --ucf data/raw/Eco1C1G1T1.UCF.json \\
        --gate A1_AmtR \\
        --mode FULL_CONTEXT \\
        --n-candidates 10 \\
        --output results/not_gate_run1
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEFAULT_N_CANDIDATES, DEFAULT_TEMPERATURE, DEFAULT_TOP_K
from src.generation.evo2_interface import Evo2Generator
from src.generation.types import GenerationMode
from src.parser.cello_parser import parse_ucf
from src.scoring.scorer import MultiObjectiveScorer
from src.validation.validator import CircuitValidator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the generative syn-bio pipeline for one gate and one generation mode."
    )
    p.add_argument("--ucf", required=True, help="Path to UCF JSON file")
    p.add_argument("--gate", required=True,
                   help="Gate name prefix, e.g. A1_AmtR")
    p.add_argument("--mode", required=True,
                   choices=[m.value for m in GenerationMode],
                   help="Generation mode for context window ablation")
    p.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES,
                   help=f"Candidates per part (default: {DEFAULT_N_CANDIDATES})")
    p.add_argument("--output", default="results/pipeline_run",
                   help="Output directory (created if needed)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return p


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = GenerationMode(args.mode)
    gate_prefix = args.gate + "__"

    # --- Stage 2: parse ---
    print(f"[1/4] Parsing UCF: {args.ucf}")
    all_specs = parse_ucf(args.ucf)
    specs = [s for s in all_specs if s.part_id.startswith(gate_prefix)]
    if not specs:
        print(f"ERROR: No parts found for gate '{args.gate}'. "
              f"Available gates: {sorted({s.part_id.split('__')[0] for s in all_specs})}")
        sys.exit(1)
    print(f"  Found {len(specs)} part(s) for gate '{args.gate}'")

    # --- Stage 3: generate ---
    print(f"[2/4] Generating {args.n_candidates} candidate(s) per part "
          f"(mode={mode.value}) ...")
    generator = Evo2Generator()
    all_candidates = []
    for spec in specs:
        candidates = generator.generate(
            spec,
            mode=mode,
            n_candidates=args.n_candidates,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        all_candidates.extend(candidates)
    print(f"  Generated {len(all_candidates)} total candidate(s)")

    # --- Stage 4: score and rank ---
    print("[3/4] Scoring and ranking ...")
    scorer = MultiObjectiveScorer()
    ranked = scorer.rank(all_candidates, filter=True)
    print(f"  {len(ranked)} candidate(s) passed hard filters")

    # Optionally score reference sequences for comparison
    for result_spec in specs:
        if result_spec.reference_seq:
            ref_lp = generator.score_sequence(result_spec.reference_seq, result_spec, mode)
            print(f"  Reference log_prob for {result_spec.part_id}: {ref_lp:.4f}")

    # --- Stage 5: validate ---
    print("[4/4] Validating circuit ...")
    validator = CircuitValidator()
    report = validator.validate_circuit(ranked, args.gate, mode)

    status = "COMPLETE" if report.is_complete else "INCOMPLETE"
    print(f"\n  Circuit status: {status}")
    for r in report.results:
        top_score = f"{r.top_score.composite_score:.3f}" if r.top_score else "N/A"
        frame = f"  cds_frame={r.cds_frame_valid}" if r.cds_frame_valid is not None else ""
        print(f"  {r.part_id}: pass={r.has_passing_candidate}  "
              f"composite={top_score}{frame}")

    # --- Save outputs ---
    csv_path = output_dir / "report.csv"
    json_path = output_dir / "summary.json"

    report.to_csv(csv_path)
    with open(json_path, "w") as f:
        json.dump(report.summary_dict(), f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  {csv_path.name}  — per-part results")
    print(f"  {json_path.name} — JSON summary")


if __name__ == "__main__":
    main(build_parser().parse_args())
