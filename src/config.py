from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")

# Paths
DATA_DIR       = _PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
REFERENCE_DIR  = DATA_DIR / "reference"
RESULTS_DIR    = _PROJECT_ROOT / "results"

# Model IDs
# Use DEV model locally (Mac CPU/MPS) — fast, small, good for testing pipeline code
# Switch to PROD model on a cloud GPU for actual generation runs
EVO2_MODEL_DEV  = "arcinstitute/evo2_1b_base"   # ~2GB, runs on Mac
EVO2_MODEL_PROD = "arcinstitute/evo2_7b"          # ~14GB, needs GPU

# Read from .env or default to dev
EVO2_MODEL_ID = os.getenv("EVO2_MODEL", EVO2_MODEL_DEV)

# Generation defaults
DEFAULT_N_CANDIDATES = 100
DEFAULT_TEMPERATURE  = 0.8
DEFAULT_TOP_K        = 4      # evo2 package uses top_k, not top_p
DEFAULT_CONTEXT_BP   = 80    # flanking context window in base pairs


def validate_config():
    """Call at startup to catch missing env vars early."""
    missing = []
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            f"Add them to your .env file at {_PROJECT_ROOT / '.env'}"
        )
    print(f"Config OK — using model: {EVO2_MODEL_ID}")
    return True
