from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tess_classifier.config import DATASET_SUMMARY_FILE, TESS_MANIFEST_FILE
from tess_classifier.data import write_dataset_manifest, write_dataset_summary


if __name__ == "__main__":
    manifest = write_dataset_manifest()
    summary = write_dataset_summary()
    print(f"Wrote manifest with {len(manifest)} rows to: {TESS_MANIFEST_FILE}")
    print(f"Wrote dataset summary to: {DATASET_SUMMARY_FILE}")
    print(f"Labels found: {summary['label_counts']}")
