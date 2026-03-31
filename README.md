# TESS Variable Star Classifier

A small machine-learning project that combines astrophysics, computer science, and AI by classifying stellar variability from TESS light curves.

This repository is intentionally scoped like a strong student portfolio project:

- small enough to finish
- technical enough to discuss in interviews
- grounded in real astronomy data
- structured like a reproducible ML pipeline instead of a single notebook

## Project Summary

The goal is to classify light curves into a few interpretable stellar-variability categories:

- `quiet_star`
- `delta_scuti`
- `eclipsing_binary`

The project supports two workflows:

1. A synthetic workflow for quick local testing
2. A curated real-data workflow using official TESS light curves downloaded from MAST

The current repo includes a tiny official TESS sample converted into local CSV files so the full pipeline can be demonstrated end to end.

## Why This Project Works Well On a Resume

- It uses real mission data from TESS.
- It has a clear AI component: supervised classification.
- It has a clear astrophysics component: stellar variability and light-curve interpretation.
- It has a clear software-engineering component: data prep, feature extraction, model training, saved artifacts, and an interactive app.
- It is honest about scope and limitations, which makes it easier to explain well.

## What The Pipeline Does

1. Load light curves from a curated local dataset.
2. Clean and normalize flux values.
3. Extract lightweight time-series features.
4. Train and compare baseline classifiers.
5. Save the best model and metrics.
6. Serve a Streamlit app for interactive inspection and prediction.

## Current Real Dataset

The current real-data sample includes official TESS light curves for:

- Algol as the eclipsing-binary example
- WASP-33 as the delta-Scuti example
- HD 209458 as a relatively quiet comparison target

These files are tracked in:

- [manifest.csv](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\data\tess_curated\manifest.csv)
- [dataset_summary.json](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\data\dataset_summary.json)

This is intentionally a tiny benchmark, not a publication-grade dataset. It is useful because it demonstrates the real-data workflow:

- download official TESS products
- convert FITS light curves into a consistent CSV format
- validate and catalog the dataset
- train a baseline model on real files
- inspect predictions in a small app

## Quick Start

If PowerShell activation is blocked on your machine, use the virtual-environment Python directly.

```powershell
cd c:\Users\Admin\Desktop\repos\tess-variable-star-classifier
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Common Workflows

Train on synthetic data:

```powershell
.\.venv\Scripts\python.exe train.py
```

Generate placeholder CSVs for format testing:

```powershell
.\.venv\Scripts\python.exe seed_local_dataset.py
.\.venv\Scripts\python.exe prepare_dataset.py
```

Download the tiny official TESS sample used by this repo:

```powershell
.\.venv\Scripts\python.exe download_real_tess_sample.py
.\.venv\Scripts\python.exe prepare_dataset.py
```

Train on the curated real dataset:

```powershell
.\.venv\Scripts\python.exe train.py --dataset-source local_tess
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Repository Layout

```text
tess-variable-star-classifier/
|-- app.py
|-- train.py
|-- seed_local_dataset.py
|-- prepare_dataset.py
|-- download_real_tess_sample.py
|-- data/
|-- docs/
|-- models/
`-- src/
    `-- tess_classifier/
        |-- config.py
        |-- data.py
        |-- features.py
        `-- train.py
```

## Key Files

- [app.py](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\app.py): Streamlit interface for exploring predictions and curated samples
- [train.py](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\train.py): root training entrypoint
- [download_real_tess_sample.py](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\download_real_tess_sample.py): downloads and converts a tiny official TESS sample
- [src/tess_classifier/data.py](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\src\tess_classifier\data.py): dataset loading, cleaning, manifest handling, and CSV preparation
- [src/tess_classifier/features.py](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\src\tess_classifier\features.py): engineered feature extraction
- [src/tess_classifier/train.py](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\src\tess_classifier\train.py): model selection and metric generation
- [docs/PROJECT_OVERVIEW.md](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\docs\PROJECT_OVERVIEW.md): short interview/demo notes

## Data Format

Curated light curves live under `data/tess_curated/<label>/`.

Each CSV should include:

- `time` or `TIME`
- `flux`, `FLUX`, `PDCSAP_FLUX`, or `SAP_FLUX`

Example:

```text
data/
`-- tess_curated/
    |-- delta_scuti/
    |   `-- star_001.csv
    |-- eclipsing_binary/
    |   `-- star_101.csv
    `-- quiet_star/
        `-- star_201.csv
```

After adding or changing curated files, run:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

That command:

- regenerates the dataset manifest
- validates each file
- writes a dataset summary for quick inspection

## Results And Limitations

The current real-data training run is only a smoke test because the official sample is tiny. That means the metrics are useful for checking that the workflow functions correctly, but they should not be presented as a meaningful scientific benchmark.

That limitation is okay for a portfolio project. The real value here is showing that you can:

- work with mission data
- build a clean local dataset
- design a repeatable ML pipeline
- present results through an interactive interface

