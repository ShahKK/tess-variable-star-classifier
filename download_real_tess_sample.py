from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Catalogs, Observations

from tess_classifier.config import TESS_CURATED_DIR, TESS_MANIFEST_FILE
from tess_classifier.data import write_dataset_manifest, write_dataset_summary


TARGETS = [
    {
        "label": "eclipsing_binary",
        "candidates": [
            {
                "object_name": "Algol",
                "tic_id": "346783960",
            }
        ],
        "notes": "Real TESS light curves for Algol, a well-known eclipsing binary.",
    },
    {
        "label": "delta_scuti",
        "candidates": [
            {
                "object_name": "Altair",
                "tic_id": "471012052",
            },
            {
                "object_name": "WASP-33",
                "query_name": "WASP-33",
            },
            {
                "object_name": "FG Vir",
                "query_name": "FG Vir",
            },
            {
                "object_name": "AI Vel",
                "query_name": "AI Vel",
            },
        ],
        "notes": "Real TESS light curves for a delta Scuti variable.",
    },
    {
        "label": "quiet_star",
        "candidates": [
            {
                "object_name": "HD 209458",
                "tic_id": "420814525",
            },
            {
                "object_name": "18 Sco",
                "query_name": "18 Sco",
            },
        ],
        "notes": "Relatively quiet comparison target with shallow transit signatures.",
    },
]


def extract_light_curve_from_fits(fits_path: Path) -> pd.DataFrame:
    """Read time and flux columns from a downloaded TESS FITS light curve."""
    with fits.open(fits_path, memmap=False) as hdul:
        table = hdul[1].data
        columns = {name.upper(): name for name in table.columns.names}

        time_column = columns.get("TIME")
        flux_column = (
            columns.get("PDCSAP_FLUX")
            or columns.get("SAP_FLUX")
            or columns.get("FLUX")
        )
        if time_column is None or flux_column is None:
            raise ValueError(f"Could not find TIME/FLUX columns in {fits_path.name}")

        frame = pd.DataFrame(
            {
                "time": np.asarray(table[time_column], dtype=float),
                "flux": np.asarray(table[flux_column], dtype=float),
            }
        ).dropna()

    return frame


def downsample_frame(frame: pd.DataFrame, max_points: int = 1200) -> pd.DataFrame:
    """Keep file sizes small while preserving the overall light-curve shape."""
    if len(frame) <= max_points:
        return frame
    indices = pd.Series(np.linspace(0, len(frame) - 1, num=max_points).round().astype(int))
    return frame.iloc[indices.to_numpy()].reset_index(drop=True)


def resolve_tic_id(candidate: dict[str, str]) -> tuple[str, str]:
    """Resolve a candidate object to a TIC identifier."""
    if candidate.get("tic_id"):
        return candidate["tic_id"], candidate["object_name"]

    query_name = candidate["query_name"]
    catalog_row = Catalogs.query_object(query_name, catalog="TIC", radius=0.001)
    if len(catalog_row) == 0:
        raise ValueError(f"No TIC row found for {query_name}")
    return str(catalog_row[0]["ID"]), candidate["object_name"]


def query_target_observations(tic_id: str) -> pd.DataFrame:
    """Query TESS observations for a TIC target."""
    observations = Observations.query_criteria(
        obs_collection="TESS",
        target_name=f"TIC {tic_id}",
    )
    if len(observations) > 0:
        return observations

    catalog_row = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.001)
    if len(catalog_row) == 0:
        raise ValueError(f"No TIC row found for TIC {tic_id}")

    ra = float(catalog_row[0]["ra"])
    dec = float(catalog_row[0]["dec"])
    observations = Observations.query_region(
        f"{ra} {dec}",
        radius="0.001 deg",
    )
    return observations[observations["obs_collection"] == "TESS"]


def select_light_curve_products(observations: pd.DataFrame, max_files: int = 4) -> pd.DataFrame:
    """Select a small set of downloadable TESS light-curve FITS products."""
    products = Observations.get_product_list(observations)
    product_frame = products.to_pandas()

    mask = product_frame["productFilename"].fillna("").str.contains(r"lc\.fits$", case=False, regex=True)
    product_frame = product_frame.loc[mask].copy()
    if product_frame.empty:
        raise ValueError("No light-curve FITS products found for this target.")

    product_frame["sector_key"] = product_frame["productFilename"].str.extract(r"(s\d{4})", expand=False).fillna("")
    product_frame["priority"] = 1
    product_frame.loc[
        product_frame["productFilename"].str.contains("hlsp_halo-tess", case=False, regex=False),
        "priority",
    ] = 0
    product_frame = product_frame.sort_values(["priority", "sector_key", "productFilename"])
    product_frame = product_frame.drop_duplicates(subset=["sector_key"], keep="first")
    return product_frame.head(max_files)


def sector_from_path(path: Path) -> str:
    """Infer a sector number from a downloaded product filename."""
    match = re.search(r"-s(\d{4})-", path.name.lower())
    if match:
        return str(int(match.group(1)))

    match = re.search(r"-(\d+)_tess_v\d+_lc\.fits", path.name.lower())
    if match:
        return match.group(1)

    return ""


def update_manifest_rows(entries: list[dict[str, str]]) -> None:
    """Merge downloaded-file metadata into the manifest."""
    manifest = write_dataset_manifest(TESS_CURATED_DIR, TESS_MANIFEST_FILE)
    manifest_rows = manifest.to_dict(orient="records")
    entry_map = {entry["relative_path"]: entry for entry in entries}

    updated_rows: list[dict] = []
    for row in manifest_rows:
        relative_path = str(row["relative_path"])
        merged = dict(row)
        if relative_path in entry_map:
            merged.update(entry_map[relative_path])
        updated_rows.append(merged)

    pd.DataFrame(updated_rows).to_csv(TESS_MANIFEST_FILE, index=False)


def download_target_samples(max_files_per_target: int = 4) -> list[dict[str, str]]:
    """Download a small real TESS sample and convert it to curated CSV files."""
    written_entries: list[dict[str, str]] = []

    for target in TARGETS:
        selected_table = None
        object_name = ""
        tic_id = ""
        last_error = ""

        for candidate in target["candidates"]:
            try:
                tic_id, object_name = resolve_tic_id(candidate)
                print(f"Querying {object_name} (TIC {tic_id})...")
                observations = query_target_observations(tic_id)
                selected_products = select_light_curve_products(observations, max_files=max_files_per_target)
                selected_table = selected_products.head(max_files_per_target)
                break
            except Exception as exc:
                last_error = str(exc)
                print(f"Skipping {candidate.get('object_name', candidate.get('query_name', 'candidate'))}: {exc}")

        if selected_table is None:
            raise ValueError(f"Could not find downloadable TESS light curves for {target['label']}: {last_error}")

        manifest = Observations.download_products(Table.from_pandas(selected_table), mrp_only=False)
        manifest_frame = manifest.to_pandas()

        label_dir = TESS_CURATED_DIR / target["label"]
        label_dir.mkdir(parents=True, exist_ok=True)
        object_prefix = object_name.lower().replace(" ", "_").replace("-", "_")
        for existing_file in label_dir.glob(f"{object_prefix}_sector_*.csv"):
            existing_file.unlink()

        for _, row in manifest_frame.iterrows():
            downloaded_path = Path(row["Local Path"])
            if not downloaded_path.exists():
                continue

            frame = extract_light_curve_from_fits(downloaded_path)
            frame = downsample_frame(frame)
            sector = sector_from_path(downloaded_path)
            sample_name = f"{object_prefix}_sector_{sector or 'unknown'}"
            output_file = label_dir / f"{sample_name}.csv"
            frame.to_csv(output_file, index=False)

            written_entries.append(
                {
                    "relative_path": str(output_file.relative_to(TESS_CURATED_DIR)).replace("\\", "/"),
                    "label": target["label"],
                    "sample_name": output_file.stem,
                    "object_id": tic_id,
                    "sector": sector,
                    "source": "MAST TESS official sample",
                    "is_real_data": True,
                    "notes": target["notes"],
                }
            )

            print(f"Saved {output_file.name}")

    return written_entries


if __name__ == "__main__":
    entries = download_target_samples()
    update_manifest_rows(entries)
    summary = write_dataset_summary(TESS_CURATED_DIR)
    print(f"Downloaded {len(entries)} real light-curve CSV files.")
    print(f"Real data files in summary: {summary['real_data_files']}")
