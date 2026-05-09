# borewell

Baseline solution for the horizontal well TVT prediction task.

## What is included

- `baseline_tvt_model.py`: train-and-predict pipeline using per-row features from `__horizontal_well.csv` files.

## Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run with real competition files

```bash
python baseline_tvt_model.py --data-dir /path/to/data --output submission.csv
```

Expected layout:

```text
<data-dir>/
  train/
    <WELL>__horizontal_well.csv
  test/
    <WELL>__horizontal_well.csv
```

## Run a reproducible demo (no external files)

```bash
python baseline_tvt_model.py --demo --output demo_submission.csv
```

This prints GroupKFold RMSE and writes output in the required format:

```text
id,tvt
000d7d20_1442,0.0
```
