# generative-syn-bio

Scaffolding pipeline that connects [Cello](https://github.com/CIDARLAB/cello) circuit design outputs to the [Evo 2](https://github.com/arcinstitute/evo2) DNA language model for context-aware part sequence generation.

The core thesis: Evo 2's large context window gives it a structural advantage for circuit-aware part generation when full flanking sequence context is provided. The key ablation compares Evo 2 log-probability as a fitness proxy across four conditioning modes:

```
no context → upstream only → full context → tagged full context
```

## Pipeline Overview

```
parse_ucf()  →  Evo2Generator.generate()  →  MultiObjectiveScorer.rank()  →  CircuitValidator.validate_circuit()
  Stage 2             Stage 3                       Stage 4                          Stage 5
```

| Stage | Module            | Description                                         |
| ----- | ----------------- | --------------------------------------------------- |
| 1     | `src/schema/`     | `PartSpec` — shared data contract across all stages |
| 2     | `src/parser/`     | Cello UCF JSON → `List[PartSpec]`                   |
| 3     | `src/generation/` | Evo 2 interface + `GenerationMode` enum             |
| 4     | `src/scoring/`    | `MultiObjectiveScorer` + `ScoreVector`              |
| 5     | `src/validation/` | `CircuitValidator` + `CircuitReport`                |

## Setup

**Requirements:** Python 3.11, a `.env` file in the project root.

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install package + dev dependencies
pip install -e ".[dev]"

# Download raw data (Cello UCF, E. coli genome, iGEM parts)
bash scripts/download_datasets.sh
```

**.env file:**

```
EVO2_MODEL=evo2_1b_base   # or evo2_7b for GPU runs
```

The Evo 2 model is lazy-loaded — not downloaded until `generate()` is first called.

## Running Tests

```bash
pytest tests/ -v                        # all tests (104 passing, 1 skipped)
pytest tests/ -k "not integration" -v   # skip GPU-requiring tests (local dev)
pytest tests/ -m integration -v         # GPU integration tests only
```

Individual stages:

```bash
pytest tests/test_schema.py -v      # Stage 1 (6 tests)
pytest tests/test_parser.py -v      # Stage 2 (23 tests)
pytest tests/test_scoring.py -v     # Stage 4 (44 tests)
pytest tests/test_validation.py -v  # Stage 5 (24 tests)
```

## Running the Pipeline

Generate candidate sequences for a circuit (requires GPU for full Evo 2 runs):

```bash
python scripts/run_pipeline.py \
    --circuit data/processed/not_gate \
    --mode TAGGED_FULL \
    --n-candidates 100 \
    --output results/not_gate_run1
```

`--mode` options: `NO_CONTEXT`, `UPSTREAM_ONLY`, `FULL_CONTEXT`, `TAGGED_FULL`

## Running the Ablation Study

Compare the four conditioning modes on a characterized RBS dataset:

```bash
python scripts/run_ablation.py \
    --dataset data/reference/mutalik_rbs.csv \
    --output results/ablation
```

## Data

Raw data is not committed. After cloning, run:

```bash
bash scripts/download_datasets.sh
```

| Path                           | Contents                                 |
| ------------------------------ | ---------------------------------------- |
| `data/raw/Eco1C1G1T1.UCF.json` | Cello E. coli gate library               |
| `data/raw/ecoli_k12.fna`       | E. coli K-12 genome (BLAST db)           |
| `data/reference/`              | iGEM characterized parts (via BioPython) |

## Key Interfaces

```python
from src.parser.cello_parser import parse_ucf
from src.generation.evo2_generator import Evo2Generator
from src.generation.types import GenerationMode
from src.scoring.scorer import MultiObjectiveScorer
from src.validation.validator import CircuitValidator

parts = parse_ucf("data/raw/Eco1C1G1T1.UCF.json")
generator = Evo2Generator()
candidates = generator.generate(parts[0], mode=GenerationMode.TAGGED_FULL, n=10)
scores = MultiObjectiveScorer().rank(candidates)
report = CircuitValidator().validate_circuit(scores)
```

## Known Issues

- NUPACK 4.0 requires separate install from [nupack.org](https://nupack.org) (Stage 4, deferred)
- RBS Calculator requires separate install from Salis Lab (Stage 4, deferred — proxy used)
- `evo2` package expects short model names (`evo2_1b_base`), not full HuggingFace paths
