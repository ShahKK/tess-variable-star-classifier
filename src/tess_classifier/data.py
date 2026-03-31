from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_SUMMARY_FILE, TESS_CURATED_DIR, TESS_MANIFEST_FILE
from .features import FEATURE_COLUMNS, extract_features


LABELS = ("quiet_star", "delta_scuti", "eclipsing_binary")
TIME_COLUMNS = ("time", "TIME")
FLUX_COLUMNS = ("flux", "FLUX", "pdcsap_flux", "PDCSAP_FLUX", "sap_flux", "SAP_FLUX")


def generate_synthetic_light_curve(
    label: str,
    num_points: int = 200,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple synthetic light curve for one variability class."""
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 10.0, num_points)
    noise = rng.normal(0.0, 0.01, num_points)

    if label == "quiet_star":
        flux = 1.0 + noise
    elif label == "delta_scuti":
        amplitude = rng.uniform(0.03, 0.12)
        cycles = rng.uniform(6.0, 14.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal = amplitude * np.sin((2.0 * np.pi * cycles * time_axis / time_axis.max()) + phase)
        flux = 1.0 + signal + noise
    elif label == "eclipsing_binary":
        period = rng.uniform(1.2, 3.0)
        depth = rng.uniform(0.12, 0.25)
        width = rng.uniform(0.05, 0.1)
        offset = rng.uniform(0.0, 1.0)
        phase = ((time_axis / period) + offset) % 1.0
        primary_eclipse = (phase < width) | (phase > 1.0 - width)
        secondary_eclipse = np.abs(phase - 0.5) < (width * 0.6)
        flux = (
            1.0
            - depth * primary_eclipse.astype(float)
            - (depth * 0.45) * secondary_eclipse.astype(float)
            + noise
        )
    else:
        raise ValueError(f"Unknown label: {label}")

    return time_axis, flux


def build_demo_dataset(
    samples_per_class: int = 250,
    num_points: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a tabular feature dataset from synthetic light curves."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    for label in LABELS:
        for _ in range(samples_per_class):
            sample_seed = int(rng.integers(0, 1_000_000))
            time_axis, flux = generate_synthetic_light_curve(
                label=label,
                num_points=num_points,
                seed=sample_seed,
            )
            feature_row = extract_features(time_axis, flux)
            feature_row["label"] = label
            rows.append(feature_row)

    frame = pd.DataFrame(rows)
    ordered_columns = list(FEATURE_COLUMNS) + ["label"]
    return frame[ordered_columns]


def clean_light_curve(
    time_axis: np.ndarray,
    flux: np.ndarray,
    min_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop invalid values, sort by time, and normalize flux around unity."""
    frame = pd.DataFrame({"time": time_axis, "flux": flux}).dropna()
    frame = frame.drop_duplicates(subset="time").sort_values("time")

    if len(frame) < min_points:
        raise ValueError(f"Need at least {min_points} valid points, found {len(frame)}.")

    median_flux = float(frame["flux"].median())
    if not np.isfinite(median_flux) or median_flux == 0.0:
        raise ValueError("Median flux must be finite and non-zero.")

    frame["flux"] = frame["flux"] / median_flux

    std_flux = float(frame["flux"].std())
    if np.isfinite(std_flux) and std_flux > 0.0:
        center = float(frame["flux"].median())
        clipped = frame[(frame["flux"] >= center - (5.0 * std_flux)) & (frame["flux"] <= center + (5.0 * std_flux))]
        if len(clipped) >= min_points:
            frame = clipped

    return frame["time"].to_numpy(dtype=float), frame["flux"].to_numpy(dtype=float)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find a {kind} column. Expected one of: {', '.join(candidates)}")


def load_light_curve_csv(path: Path, min_points: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalize a light curve CSV file."""
    frame = pd.read_csv(path)
    time_column = _pick_column(frame, TIME_COLUMNS, "time")
    flux_column = _pick_column(frame, FLUX_COLUMNS, "flux")
    return clean_light_curve(frame[time_column].to_numpy(), frame[flux_column].to_numpy(), min_points=min_points)


def list_local_light_curve_files(dataset_dir: Path = TESS_CURATED_DIR) -> list[Path]:
    """Return all CSV light curves from a curated local dataset directory."""
    if not dataset_dir.exists():
        return []
    return sorted(
        path
        for path in dataset_dir.rglob("*.csv")
        if path.is_file() and path.name.lower() != TESS_MANIFEST_FILE.name.lower()
    )


def load_existing_manifest(manifest_file: Path = TESS_MANIFEST_FILE) -> pd.DataFrame:
    """Load an existing dataset manifest if present."""
    if not manifest_file.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_file)


def build_dataset_manifest(dataset_dir: Path = TESS_CURATED_DIR) -> pd.DataFrame:
    """Build a lightweight manifest for the curated light-curve folder."""
    existing_manifest = load_existing_manifest(TESS_MANIFEST_FILE)
    existing_rows: dict[str, dict] = {}
    if not existing_manifest.empty and "relative_path" in existing_manifest.columns:
        for row in existing_manifest.to_dict(orient="records"):
            existing_rows[str(row["relative_path"])] = row

    rows: list[dict[str, str | int | bool]] = []
    for csv_file in list_local_light_curve_files(dataset_dir):
        relative_path = csv_file.relative_to(dataset_dir)
        label = csv_file.parent.name
        sample_name = csv_file.stem
        existing = existing_rows.get(str(relative_path).replace("\\", "/"), {})
        rows.append(
            {
                "relative_path": str(relative_path).replace("\\", "/"),
                "label": label,
                "sample_name": sample_name,
                "object_id": existing.get("object_id", ""),
                "sector": existing.get("sector", ""),
                "source": existing.get("source", "manual"),
                "is_real_data": bool(existing.get("is_real_data", False)),
                "notes": existing.get("notes", ""),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "relative_path",
            "label",
            "sample_name",
            "object_id",
            "sector",
            "source",
            "is_real_data",
            "notes",
        ],
    )


def write_dataset_manifest(dataset_dir: Path = TESS_CURATED_DIR, manifest_file: Path = TESS_MANIFEST_FILE) -> pd.DataFrame:
    """Write the curated dataset manifest to disk."""
    manifest = build_dataset_manifest(dataset_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, index=False)
    return manifest


def summarize_local_dataset(dataset_dir: Path = TESS_CURATED_DIR) -> dict:
    """Return a compact summary of the curated dataset contents."""
    csv_files = list_local_light_curve_files(dataset_dir)
    label_counts: dict[str, int] = {}
    for csv_file in csv_files:
        label = csv_file.parent.name
        label_counts[label] = label_counts.get(label, 0) + 1

    manifest_exists = TESS_MANIFEST_FILE.exists()
    manifest = load_existing_manifest(TESS_MANIFEST_FILE) if manifest_exists else pd.DataFrame()
    real_data_files = 0
    sources: dict[str, int] = {}
    if not manifest.empty:
        if "is_real_data" in manifest.columns:
            real_data_files = int(pd.Series(manifest["is_real_data"]).fillna(False).astype(bool).sum())
        if "source" in manifest.columns:
            for source_name, count in manifest["source"].fillna("unknown").value_counts().items():
                sources[str(source_name)] = int(count)

    summary = {
        "dataset_dir": str(dataset_dir),
        "num_files": len(csv_files),
        "num_labels": len(label_counts),
        "label_counts": label_counts,
        "manifest_exists": manifest_exists,
        "manifest_path": str(TESS_MANIFEST_FILE),
        "real_data_files": real_data_files,
        "sources": sources,
    }
    return summary


def validate_local_dataset(dataset_dir: Path = TESS_CURATED_DIR, min_points: int = 80) -> list[dict[str, str | int]]:
    """Validate curated local light curves and return per-file status rows."""
    results: list[dict[str, str | int]] = []
    for csv_file in list_local_light_curve_files(dataset_dir):
        try:
            time_axis, flux = load_light_curve_csv(csv_file, min_points=min_points)
            results.append(
                {
                    "relative_path": str(csv_file.relative_to(dataset_dir)).replace("\\", "/"),
                    "label": csv_file.parent.name,
                    "status": "ok",
                    "num_points": len(time_axis),
                    "message": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "relative_path": str(csv_file.relative_to(dataset_dir)).replace("\\", "/"),
                    "label": csv_file.parent.name,
                    "status": "error",
                    "num_points": 0,
                    "message": str(exc),
                }
            )
    return results


def write_dataset_summary(dataset_dir: Path = TESS_CURATED_DIR, summary_file: Path = DATASET_SUMMARY_FILE) -> dict:
    """Persist a summary and validation report for the curated dataset."""
    summary = summarize_local_dataset(dataset_dir)
    validation_rows = validate_local_dataset(dataset_dir)
    summary["validation"] = validation_rows
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(summary).to_json(summary_file, indent=2)
    return summary


def build_local_tess_dataset(
    dataset_dir: Path = TESS_CURATED_DIR,
    min_points: int = 80,
) -> pd.DataFrame:
    """Build a feature dataset from a curated folder of labeled light-curve CSVs."""
    csv_files = list_local_light_curve_files(dataset_dir)
    manifest = load_existing_manifest(TESS_MANIFEST_FILE)
    if not manifest.empty and "is_real_data" in manifest.columns:
        real_rows = manifest[pd.Series(manifest["is_real_data"]).fillna(False).astype(bool)]
        real_relative_paths = set(real_rows["relative_path"].astype(str).tolist())
        if real_relative_paths:
            csv_files = [
                path
                for path in csv_files
                if str(path.relative_to(dataset_dir)).replace("\\", "/") in real_relative_paths
            ]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_dir}. Add labeled files under "
            "data/tess_curated/<label>/sample.csv"
        )

    rows: list[dict[str, float | str]] = []
    for csv_file in csv_files:
        label = csv_file.parent.name
        time_axis, flux = load_light_curve_csv(csv_file, min_points=min_points)
        feature_row = extract_features(time_axis, flux)
        feature_row["label"] = label
        feature_row["source_file"] = str(csv_file)
        feature_row["relative_path"] = str(csv_file.relative_to(dataset_dir)).replace("\\", "/")
        rows.append(feature_row)

    frame = pd.DataFrame(rows)
    ordered_columns = list(FEATURE_COLUMNS) + ["label", "source_file", "relative_path"]
    return frame[ordered_columns]


def write_sample_curated_dataset(
    dataset_dir: Path = TESS_CURATED_DIR,
    samples_per_class: int = 4,
    num_points: int = 200,
    seed: int = 42,
    overwrite: bool = False,
) -> list[Path]:
    """Write small example CSVs in the curated-dataset format for local testing."""
    rng = np.random.default_rng(seed)
    written_files: list[Path] = []

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        label_dir = dataset_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for index in range(1, samples_per_class + 1):
            output_file = label_dir / f"{label}_{index:03d}.csv"
            if output_file.exists() and not overwrite:
                written_files.append(output_file)
                continue

            sample_seed = int(rng.integers(0, 1_000_000))
            time_axis, flux = generate_synthetic_light_curve(
                label=label,
                num_points=num_points,
                seed=sample_seed,
            )
            frame = pd.DataFrame({"time": time_axis, "flux": flux})
            frame.to_csv(output_file, index=False)
            written_files.append(output_file)

    return written_files
