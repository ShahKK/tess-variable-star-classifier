# Data Directory

This folder contains generated artifacts and curated dataset files used by the project.

## What Lives Here

- `demo_features.csv`: feature table generated from synthetic light curves
- `tess_features.csv`: feature table generated from curated local TESS files
- `dataset_summary.json`: summary and validation report for the curated dataset
- `tess_curated/`: labeled light-curve CSV files plus the dataset manifest

## Typical Workflow

1. Add or download curated light-curve CSVs into `tess_curated/`
2. Run `prepare_dataset.py`
3. Train with `train.py --dataset-source local_tess`

The curated dataset documentation is in [data/tess_curated/README.md](c:\Users\Admin\Desktop\repos\tess-variable-star-classifier\data\tess_curated\README.md).

