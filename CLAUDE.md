# CLAUDE.md

## Project Overview
Scaffolding pipeline connecting Cello circuit design outputs to Evo 2 DNA language
model generation. Replaces part library lookup with context-aware sequence generation.

## Thesis
Large-context DNA language models (Evo 2) have a structural advantage for
circuit-aware part generation when full flanking sequence context is provided.

## Architecture
Five sequential stages, each in src/<stage_name>/
- Stage 1: schema/      — PartSpec data model (Pydantic)
- Stage 2: parser/      — Cello output → List[PartSpec]
- Stage 3: generation/  — PartSpec → Evo 2 → List[GeneratedSequence]
- Stage 4: scoring/     — GeneratedSequence → ScoreVector (multi-objective)
- Stage 5: validation/  — End-to-end NOT gate run + ablation

## Current Stage
Stage 1 — Implementing PartSpec schema in src/schema/part_spec.py

## Key Data Contracts
- src/schema/part_spec.py   → PartSpec (primary exchange object between all stages)
- src/generation/types.py   → GeneratedSequence
- src/scoring/types.py      → ScoreVector

## Running Tests
pytest tests/ -v                      # all tests
pytest tests/test_schema.py -v        # stage 1 only
pytest tests/ -k "not integration"    # skip slow Evo2 model tests

## Environment
Python 3.11+
Activate: source .venv/bin/activate
Evo 2 model: arcinstitute/evo2-7b (Hugging Face)
BLAST db: data/raw/blast_db/ecoli_k12 (download in Stage 4)

## Known Issues
- NUPACK 4.0 requires a separate install via nupack.org (free academic license)
- RBS Calculator requires shell subprocess interface (wrap in src/scoring/rbs_calc.py)
- Evo 2 weights require Hugging Face access request at hf.co/arcinstitute/evo2

## Running the Pipeline (Stage 5, not yet built)
python scripts/run_pipeline.py \
    --circuit data/processed/not_gate \
    --mode TAGGED_FULL \
    --n-candidates 100 \
    --output results/not_gate_run1
