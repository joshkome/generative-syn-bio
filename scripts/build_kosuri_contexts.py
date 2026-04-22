"""
Build a CSV of full circuit DNA sequences with each Kosuri RBS swapped in.

Output columns:
    rbs_name      — RBS identifier (from Kosuri dataset)
    rbs_sequence  — cleaned RBS sequence (spaces stripped, CATATG removed)
    full_sequence — upstream_context + rbs_sequence + downstream_context
    mean_xlat     — measured relative translation rate

Usage:
    python scripts/build_kosuri_contexts.py
    python scripts/build_kosuri_contexts.py --ucf data/raw/Eco1C1G1T1.UCF.json \
        --rbs data/reference/kosuri_rbs.csv --gate A1_AmtR \
        --output data/processed/kosuri_contexts.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.schema.part_spec import PartType
from src.parser.cello_parser import parse_ucf


def clean_rbs_sequence(seq: str, strip_start_codon: bool = True) -> str:
    seq = seq.replace(" ", "").upper()
    if strip_start_codon and seq.endswith("CATATG"):
        seq = seq[:-6]
    return seq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ucf",    default="data/raw/Eco1C1G1T1.UCF.json",       help="Path to Cello UCF JSON")
    parser.add_argument("--rbs",    default="data/reference/kosuri_rbs.csv",       help="Path to Kosuri RBS CSV")
    parser.add_argument("--gate",   default="A1_AmtR",                             help="Gate name to use for context")
    parser.add_argument("--output",          default="data/processed/kosuri_contexts.csv", help="Output CSV path")
    parser.add_argument("--upstream-trim",   type=int, default=20, help="Upstream bp to include in trimmed_sequence (default 20)")
    args = parser.parse_args()

    # ── Parse UCF and extract RBS context ────────────────────────────────────
    print(f"Parsing UCF: {args.ucf}")
    parts = parse_ucf(args.ucf)

    rbs_parts = [p for p in parts if p.part_type == PartType.RBS and args.gate in p.part_id]
    if not rbs_parts:
        sys.exit(f"No RBS part found for gate '{args.gate}' in {args.ucf}")

    rbs_part = rbs_parts[0]
    upstream   = rbs_part.upstream_context
    downstream = rbs_part.downstream_context
    print(f"Gate: {args.gate}  |  upstream={len(upstream)} bp  downstream={len(downstream)} bp")
    print(f"Native RBS ({rbs_part.sbol_component_id}): {rbs_part.reference_seq}")

    # ── Load Kosuri dataset ───────────────────────────────────────────────────
    print(f"\nLoading RBS dataset: {args.rbs}")
    df = pd.read_csv(args.rbs)

    # Strip triple-quote artifacts produced by some CSV exports
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].str.strip('"').str.strip()

    df["rbs_sequence"] = df["Sequence"].map(clean_rbs_sequence)

    # Drop rows with invalid sequences or missing translation data
    valid = df["rbs_sequence"].str.fullmatch(r"[ACGT]+", na=False) & df["mean.xlat"].notna()
    dropped = (~valid).sum()
    if dropped:
        print(f"Dropped {dropped} rows with invalid/empty sequences or missing mean.xlat.")
    df = df[valid].reset_index(drop=True)

    # ── Build sequences ───────────────────────────────────────────────────────
    trimmed_upstream = upstream[-args.upstream_trim:]

    # full_sequence:    complete circuit context (upstream + RBS + downstream)
    # upstream_rbs:     full upstream + RBS, no downstream — RBS is ~20% of tokens
    # trimmed_sequence: last N bp of upstream + RBS — maximises RBS signal fraction
    df["full_sequence"]    = upstream + df["rbs_sequence"] + downstream
    df["upstream_rbs"]     = upstream + df["rbs_sequence"]
    df["trimmed_sequence"] = trimmed_upstream + df["rbs_sequence"]

    out = df[["RBS", "rbs_sequence", "full_sequence", "upstream_rbs", "trimmed_sequence", "mean.xlat"]].rename(
        columns={"RBS": "rbs_name", "mean.xlat": "mean_xlat"}
    )

    # ── Write output ──────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows to {args.output}")
    example = out.iloc[0]
    print(f"Sequence lengths for '{example['rbs_name']}':")
    print(f"  full_sequence:    {len(example['full_sequence'])} bp  (RBS is {len(example['rbs_sequence'])}/{len(example['full_sequence'])} = {len(example['rbs_sequence'])/len(example['full_sequence']):.1%} of tokens)")
    print(f"  upstream_rbs:     {len(example['upstream_rbs'])} bp  (RBS is {len(example['rbs_sequence'])}/{len(example['upstream_rbs'])} = {len(example['rbs_sequence'])/len(example['upstream_rbs']):.1%} of tokens)")
    print(f"  trimmed_sequence: {len(example['trimmed_sequence'])} bp  (RBS is {len(example['rbs_sequence'])}/{len(example['trimmed_sequence'])} = {len(example['rbs_sequence'])/len(example['trimmed_sequence']):.1%} of tokens)")


if __name__ == "__main__":
    main()
