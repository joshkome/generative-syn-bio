# CLAUDE.md

## Project Overview
Scaffolding pipeline connecting Cello circuit design outputs to Evo 2 DNA language
model generation. Replaces part library lookup with context-aware sequence generation.

## Thesis
Large-context DNA language models (Evo 2) have a structural advantage for
circuit-aware part generation when full flanking sequence context is provided.
The key ablation compares Evo 2 log-probability as a fitness proxy across 4
conditioning modes (no context → upstream only → full context → tagged full).

## Repository Structure
    src/
      schema/         Stage 1 — PartSpec Pydantic model (DONE)
      parser/         Stage 2 — Cello UCF → List[PartSpec] (DONE)
      generation/     Stage 3 — Evo 2 interface + GenerationMode enum (DONE)
      scoring/        Stage 4 — MultiObjectiveScorer + ScoreVector (DONE)
      validation/     Stage 5 — CircuitValidator + CircuitReport (DONE)
    tests/            104 passing tests (all stages); 1 integration skipped (needs GPU)
    scripts/          run_pipeline.py, run_ablation.py
    data/
      raw/            Gitignored. Run scripts/download_datasets.sh to populate.
      processed/      Gitignored. Pipeline outputs go here.
      reference/      Gitignored. iGEM characterized parts.
    notebooks/        Jupyter notebooks, one per stage
    scripts/          Runners and download scripts
    results/          Figures and reports (gitignored except explicit commits)

## Current Stage
All 5 stages complete. The pipeline is fully connected:
  parse_ucf() → Evo2Generator.generate() → MultiObjectiveScorer.rank() → CircuitValidator.validate_circuit()

Next steps:
  - Run scripts on GPU to generate actual candidates (run_pipeline.py, run_ablation.py)
  - Collect ablation study data (4 modes × A1_AmtR gate)
  - Build notebooks for analysis/visualization
  - Write thesis results section

## Key Data Contracts
    src/schema/part_spec.py      → PartSpec            (all stages communicate via this)
    src/parser/cello_parser.py   → parse_ucf()         (UCF JSON → List[PartSpec])
    src/generation/types.py      → GeneratedSequence, GenerationMode
    src/scoring/types.py         → ScoreVector, ScorerWeights  (Stage 4)

## Environment
    Python 3.11 — activate with: source .venv/bin/activate
    Install deps: pip install -e ".[dev]"
    Token storage: .env file in project root (never commit this)
    Evo 2 model: set EVO2_MODEL in .env (defaults to arcinstitute/evo2_1b_base)

## Running Tests
    pytest tests/ -v                          # all tests
    pytest tests/test_schema.py -v            # stage 1 only
    pytest tests/test_parser.py -v            # stage 2 only
    pytest tests/test_scoring.py -v           # stage 4 only
    pytest tests/test_validation.py -v        # stage 5 only
    pytest tests/ -k "not integration" -v     # skip GPU-requiring Evo 2 model tests
    pytest tests/ -m integration -v           # run GPU integration tests only

## Running the Pipeline (Stage 5 — not yet built)
    python scripts/run_pipeline.py \
        --circuit data/processed/not_gate \
        --mode TAGGED_FULL \
        --n-candidates 100 \
        --output results/not_gate_run1

## Running the Ablation (Stage 5 — not yet built)
    python scripts/run_ablation.py \
        --dataset data/reference/mutalik_rbs.csv \
        --output results/ablation

## Package Notes
- evo2 package (pip install evo2) wraps model loading and generation
  Model IDs: evo2_1b_base (dev/Mac), evo2_7b (GPU runs)
  Model is lazy-loaded — not downloaded until generate() is first called
- python-dotenv loads .env automatically via src/config.py
- All modules should import config via: from src.config import ...

## Known Issues / Gotchas
- pytest needs pythonpath = ["."] in pyproject.toml (already set) to resolve src.*
- evo2 package expects short model names e.g. 'evo2_1b_base' not full HF path
- ViennaRNA installed via pip (2.7.2) — no conda needed on Apple Silicon
- NUPACK 4.0 still needs separate install from nupack.org (Stage 4, deferred)
- RBS Calculator needs separate install from salis-lab (Stage 4, deferred — use proxy)
- data/raw/ is gitignored — teammates must run download_datasets.sh after cloning
- ViennaRNA Python bindings import as `RNA` (from `ViennaRNA` pip package)

## Stage Status
    Stage 1 — PartSpec schema          DONE  (6 tests)
    Stage 2 — Cello parser             DONE  (23 tests)
    Stage 3 — Evo 2 interface          DONE  (7 tests; model tests need GPU)
    Stage 4 — Scoring pipeline         DONE  (44 tests)
    Stage 5 — Validation + scripts     DONE  (24 tests + 1 integration skipped)
    Total: 104 passing, 1 skipped
