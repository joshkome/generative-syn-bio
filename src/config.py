from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root, no matter where the script is run from
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

# Model
EVO2_MODEL_ID  = "arcinstitute/evo2-7b"

# Generation defaults
DEFAULT_N_CANDIDATES  = 100
DEFAULT_TEMPERATURE   = 0.8
DEFAULT_CONTEXT_BP    = 80   # flanking context window size

def validate_config():
    """Call this at startup to catch missing env vars early."""
    missing = []
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            f"Add them to your .env file at {_PROJECT_ROOT / '.env'}"
        )
    return True
