from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_ROOT.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHALLENGE_DATA_DIR = DATA_DIR / "challenge"
DERIVED_DATA_DIR = DATA_DIR / "derived"
MODEL_DIR = PROJECT_ROOT / "models"