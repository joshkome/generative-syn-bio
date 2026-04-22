# generative-syn-bio

Investigation of [Evo 2](https://github.com/arcinstitute/evo2) as a scoring function for synthetic genetic circuit design. Uses [Cello](https://github.com/CIDARLAB/cello) circuit outputs as sequence context to test whether Evo 2 log-likelihood is a meaningful fitness proxy for circuit parts.

## Research Questions

1. **Kosuri validation** — Does Evo 2 log-likelihood correlate with experimentally measured RBS translation strength when sequences are scored in their circuit context?
2. **Context ablation** — Does providing more flanking circuit context improve Evo 2 log-probability scores for generated sequences across four conditioning modes?

## Key Results

**Kosuri RBS validation (n=111):**
- Pearson r = +0.42, Spearman ρ = +0.41 (p < 0.0001) between Evo 2 log-likelihood and measured translation rate
- Correlation is robust to removal of highly synthetic sequences (DeadRBS, salis-1-21, salis-2-3)
- Scored using `trimmed_sequence` (last 20 bp upstream + RBS) via Evo 2 online interface

**Context ablation (A1_AmtR gate, NVIDIA NIM evo2-7b):**
- `tagged_full` context outperforms `no_context` baseline (p = 0.024)
- Results in `results/ablation/`

## Repository Structure

```
notebooks/
  ablation_local_NVIDIA.ipynb     context ablation study (NVIDIA NIM API)
  rbs_scoring_kosuri.ipynb        Colab scoring notebook (requires GPU)

scripts/
  build_kosuri_contexts.py        build circuit sequences with each Kosuri RBS swapped in
  analyze_rbs_correlation.py      correlation analysis + plots

data/
  raw/                            Cello UCF + circuit JSON (gitignored)
  reference/kosuri_rbs.csv        Kosuri et al. 2013 — 111 RBS sequences + translation rates
  processed/
    kosuri_contexts.csv           full circuit sequences (upstream + RBS + downstream)
    kosuri_contexts_labeled.csv   + Evo 2 NLL scores (manual, via online interface)

results/
  ablation/                       context ablation plots and CSVs
  rbs_kosuri_scoring/             correlation plots and stats

old/                              archived pipeline code (not used)
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install pandas scipy matplotlib seaborn pydantic
```

For the ablation notebook, add your NVIDIA NIM API key to `.env`:

```
NVIDIA_API_KEY=your_key_here
```

## Reproducing the Kosuri Correlation

```bash
# Step 1 — rebuild the context CSV from the UCF and RBS dataset
python scripts/build_kosuri_contexts.py

# Step 2 — score trimmed_sequence column via https://evo2.arcinstitute.org
#           and save results to data/processed/kosuri_contexts_labeled.csv

# Step 3 — run correlation analysis and save plots
python scripts/analyze_rbs_correlation.py
```

Plots are saved to `results/rbs_kosuri_scoring/`.

## Notes

- GPU limitations prevented local Evo 2 inference; sequences were scored manually via the Evo 2 online interface
- `trimmed_sequence` (20 bp upstream + RBS, ~29 bp total) was used for scoring — the full 815 bp circuit sequence dilutes the RBS signal to ~1% of tokens
- The Cello UCF parser (`src/parser/cello_parser.py`) is still used by `build_kosuri_contexts.py` to extract upstream/downstream context for the A1_AmtR gate
