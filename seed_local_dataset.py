from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tess_classifier.data import write_sample_curated_dataset


if __name__ == "__main__":
    written = write_sample_curated_dataset()
    print(f"Wrote {len(written)} sample CSV files to the curated dataset folder.")
