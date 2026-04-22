# CLAUDE.md

## Project Overview
Investigation of Evo 2 as a scoring function for synthetic genetic circuit design.
Uses Cello circuit outputs (UCF format) as sequence context for two validation experiments.
This is a research/analysis project — not a generation pipeline.

## Research Questions
1. Does Evo 2 log-likelihood correlate with measured RBS translation strength when scored
   in circuit context? (Kosuri validation)
2. Does providing more circuit context improve Evo 2 log-probability scores for generated
   sequences? (4-mode context ablation via NVIDIA NIM)

## Repository Structure
    notebooks/
      ablation_local_NVIDIA.ipynb   — context ablation study (NVIDIA NIM API, A1_AmtR gate)
      rbs_kosuri_correlation.ipynb  — exploratory correlation notebook (superseded by script)
      rbs_scoring_kosuri.ipynb      — Colab scoring notebook (requires GPU)
    scripts/
      build_kosuri_contexts.py      — generates kosuri_contexts.csv from UCF + RBS dataset
      analyze_rbs_correlation.py    — correlation analysis + saves all plots to results/
    data/
      raw/                          — Cello UCF + input/output JSON (gitignored)
      reference/kosuri_rbs.csv      — Kosuri et al. 2013 RBS library (111 sequences + xlat rates)
      processed/
        kosuri_contexts.csv         — full circuit sequences with each Kosuri RBS swapped in
        kosuri_contexts_labeled.csv — same + manually scored Evo2 NLL via online interface
    results/
      ablation/                     — ablation plots and score CSVs (A1_AmtR, 4 modes)
      rbs_kosuri_scoring/           — correlation plots and stats CSV
    old/                            — archived pipeline code (src/, tests/, prior scripts)

## Key Scripts
    # Build the kosuri context CSV from UCF + RBS dataset
    python scripts/build_kosuri_contexts.py

    # Run correlation analysis and save all plots
    python scripts/analyze_rbs_correlation.py

    # Both accept --help for path overrides

## Environment
    Python 3.11 — activate with: source .venv/bin/activate
    Install: pip install pandas scipy matplotlib seaborn

## Key Data Files
    data/raw/Eco1C1G1T1.UCF.json            — Cello E. coli gate library (gitignored)
    data/reference/kosuri_rbs.csv           — source RBS sequences and xlat measurements
    data/processed/kosuri_contexts_labeled.csv — primary analysis input

## Cello Parser
    src/parser/cello_parser.py — parse_ucf() is still used by build_kosuri_contexts.py
    src/schema/part_spec.py    — PartSpec model used by the parser
    These are the only src/ modules actively used; the rest is in old/

## Known Issues / Gotchas
    - Evo 2 online interface (https://evo2.arcinstitute.org) was used to score sequences
      manually — no local GPU available. Scores stored in kosuri_contexts_labeled.csv.
    - NVIDIA NIM API key required for ablation_local_NVIDIA.ipynb
      (set NVIDIA_API_KEY in .env or notebook config cell)
    - Kosuri CSV uses triple-quoted strings — build_kosuri_contexts.py strips these
    - Sequence scores are mean per-token NLL; higher NLL = lower model likelihood
    - trimmed_sequence column (20 bp upstream + RBS) gives best signal-to-noise for
      scoring short RBS elements; full_sequence dilutes RBS signal with 726 bp fixed CDS
