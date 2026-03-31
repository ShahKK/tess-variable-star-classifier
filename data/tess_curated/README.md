# Curated TESS Dataset

This folder stores the local light-curve dataset used by the real-data workflow.

## Folder Structure

Use one subdirectory per class label:

```text
data/tess_curated/
|-- delta_scuti/
|   `-- sample.csv
|-- eclipsing_binary/
|   `-- sample.csv
`-- quiet_star/
    `-- sample.csv
```

## Required CSV Columns

Each file should contain:

- `time` or `TIME`
- `flux`, `FLUX`, `PDCSAP_FLUX`, or `SAP_FLUX`

## Project Metadata

This folder also contains:

- `manifest.csv`: one row per curated file
- validation and source information managed by `prepare_dataset.py`

Rows in the manifest can identify whether a file is:

- a synthetic placeholder
- a downloaded official TESS sample
- a manually added real light curve

## Useful Commands

Generate placeholder files:

```powershell
.\.venv\Scripts\python.exe seed_local_dataset.py
```

Download the tiny official sample:

```powershell
.\.venv\Scripts\python.exe download_real_tess_sample.py
```

Refresh manifest and validation output:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

Train on curated real files:

```powershell
.\.venv\Scripts\python.exe train.py --dataset-source local_tess
```

