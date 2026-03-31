from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
TESS_CURATED_DIR = DATA_DIR / "tess_curated"
TESS_MANIFEST_FILE = TESS_CURATED_DIR / "manifest.csv"
DATASET_SUMMARY_FILE = DATA_DIR / "dataset_summary.json"
MODEL_FILE = MODELS_DIR / "baseline_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
SYNTHETIC_FEATURE_DATA_FILE = DATA_DIR / "demo_features.csv"
LOCAL_TESS_FEATURE_DATA_FILE = DATA_DIR / "tess_features.csv"
