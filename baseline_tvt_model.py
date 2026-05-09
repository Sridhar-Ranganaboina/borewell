#!/usr/bin/env python3
"""Baseline TVT prediction pipeline for horizontal wells.

Usage:
  python baseline_tvt_model.py --data-dir /path/to/competition --output submission.csv
  python baseline_tvt_model.py --demo --output demo_submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["MD", "X", "Y", "Z", "GR", "TVT_input", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline TVT model and create submission.")
    p.add_argument("--data-dir", type=Path, help="Path containing train/ and test/ folders")
    p.add_argument("--output", type=Path, default=Path("submission.csv"), help="Submission output path")
    p.add_argument("--n-splits", type=int, default=5, help="GroupKFold split count")
    p.add_argument("--demo", action="store_true", help="Run fully reproducible demo with synthetic wells")
    return p.parse_args()


def load_horizontal_files(folder: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for file in sorted(folder.glob("*__horizontal_well.csv")):
        well = file.name.split("__", maxsplit=1)[0]
        df = pd.read_csv(file)
        df["well"] = well
        df["row_index"] = np.arange(len(df), dtype=int)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No horizontal well csv files found in: {folder}")
    return pd.concat(rows, ignore_index=True)


def build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)

    def one_well(well: str, n: int, hide_last: int = 0) -> pd.DataFrame:
        md = np.arange(n)
        gr = 70 + 20 * np.sin(md / 15) + rng.normal(0, 2, n)
        z = -8500 - md * 0.8 + rng.normal(0, 3, n)
        tvt = 10 + 0.03 * md + 0.08 * gr + rng.normal(0, 0.8, n)
        df = pd.DataFrame({
            "MD": md,
            "X": 10000 + md * 3,
            "Y": 5000 + md * 1.5,
            "Z": z,
            "GR": gr,
            "ANCC": z + 30,
            "ASTNU": z + 20,
            "ASTNL": z + 10,
            "EGFDU": z + 5,
            "EGFDL": z - 5,
            "BUDA": z - 15,
            "TVT": tvt,
            "TVT_input": tvt,
            "well": well,
            "row_index": np.arange(n, dtype=int),
        })
        if hide_last > 0:
            df.loc[n - hide_last :, "TVT"] = np.nan
            df.loc[n - hide_last :, "TVT_input"] = np.nan
        return df

    train = pd.concat([one_well("a1b2c3d4", 220), one_well("b2c3d4e5", 210), one_well("c3d4e5f6", 230)], ignore_index=True)
    test = pd.concat([one_well("d4e5f6g7", 180, hide_last=80), one_well("e5f6g7h8", 170, hide_last=70)], ignore_index=True)
    return train, test


def make_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    numeric = [c for c in NUMERIC_FEATURES if c in columns]
    categorical = [c for c in ["well"] if c in columns]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def main() -> None:
    args = parse_args()
    if args.demo:
        train_df, test_df = build_demo_data()
    else:
        if not args.data_dir:
            raise ValueError("--data-dir is required unless --demo is used")
        train_df = load_horizontal_files(args.data_dir / "train")
        test_df = load_horizontal_files(args.data_dir / "test")

    if "TVT" not in train_df.columns:
        raise KeyError("Training files must include TVT column")

    train_labeled = train_df[train_df["TVT"].notna()].copy()
    features = [c for c in train_labeled.columns if c != "TVT"]
    X, y, groups = train_labeled[features], train_labeled["TVT"].values, train_labeled["well"].values

    pipe = Pipeline([
        ("prep", make_preprocessor(X.columns)),
        ("model", HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, max_iter=300, random_state=42)),
    ])

    n_splits = min(args.n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least 2 wells with labels for GroupKFold")
    rmses = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[va])
        rmse = mean_squared_error(y[va], pred, squared=False)
        rmses.append(rmse)
        print(f"Fold {fold}: RMSE={rmse:.4f}")
    print(f"CV RMSE mean={np.mean(rmses):.4f} std={np.std(rmses):.4f}")

    pipe.fit(X, y)
    X_test = test_df.reindex(columns=features)
    test_pred = pipe.predict(X_test)
    submission = pd.DataFrame({"id": test_df["well"] + "_" + test_df["row_index"].astype(str), "tvt": test_pred})
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
