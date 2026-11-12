"""Train and test 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import FT_FEATURES, MWD_FEATURES, Config, TunnelStrengthModel

BASE_DIR = Path(__file__).resolve().parent


def split_groups(
    data: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split whole construction groups once"""
    fraction_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fraction_sum, 1.0):
        raise ValueError("train/validation/test fractions must sum to 1")
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("all split fractions must be positive")

    group_stats = (
        data.groupby("group_id", as_index=False)["Rc"]
        .mean()
        .sort_values("Rc")
        .reset_index(drop=True)
    )
    n_strata = min(5, len(group_stats))
    group_stats["stratum"] = (
        pd.qcut(
            group_stats["Rc"].rank(method="first"),
            q=n_strata,
            labels=False,
        )
        .astype(int)
    )
    rng = np.random.default_rng(seed)
    split_groups_map: dict[str, str] = {}

    for _, stratum in group_stats.groupby("stratum"):
        group_names = stratum["group_id"].to_numpy(copy=True)
        rng.shuffle(group_names)
        n_groups = len(group_names)
        n_test = max(1, int(round(n_groups * test_fraction)))
        n_val = max(1, int(round(n_groups * val_fraction)))
        if n_test + n_val >= n_groups:
            n_test = max(1, int(n_groups * test_fraction))
            n_val = max(1, int(n_groups * val_fraction))
        if n_test + n_val >= n_groups:
            raise ValueError("too few groups for the requested three-way split")
        for name in group_names[:n_test]:
            split_groups_map[str(name)] = "test"
        for name in group_names[n_test : n_test + n_val]:
            split_groups_map[str(name)] = "validation"
        for name in group_names[n_test + n_val :]:
            split_groups_map[str(name)] = "train"

    split_labels = data["group_id"].astype(str).map(split_groups_map)
    if split_labels.isna().any():
        raise RuntimeError("some groups were not assigned to a split")
    return {
        split: np.flatnonzero(split_labels.to_numpy() == split)
        for split in ("train", "validation", "test")
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "R2": 1.0 - ss_res / ss_tot,
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def load_data(
    ft_path: Path, mwd_path: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ft_data = pd.read_csv(ft_path)
    mwd_data = pd.read_csv(mwd_path)
    required_ft = {"sample_id", "group_id", "Rc", *FT_FEATURES}
    required_mwd = {"sample_id", *MWD_FEATURES}
    missing = sorted(required_ft.difference(ft_data.columns))
    if missing:
        raise ValueError(f"F/T data is missing columns: {missing}")
    missing = sorted(required_mwd.difference(mwd_data.columns))
    if missing:
        raise ValueError(f"MWD data is missing columns: {missing}")
    if ft_data["sample_id"].duplicated().any() or mwd_data["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in both input tables")
    if ft_data[["sample_id", "group_id", "Rc"]].isna().any().any():
        raise ValueError("sample_id, group_id or Rc contains missing values")

    data = ft_data.merge(mwd_data, on="sample_id", how="inner", validate="one_to_one")
    if len(data) != len(ft_data) or len(data) != len(mwd_data):
        raise ValueError("F/T and MWD samples do not match one-to-one by sample_id")

    for family in (
        "derivative_Curve_F",
        "derivative_Curve_T",
        "param_Curve_F",
        "param_Curve_T",
    ):
        xf2 = data[f"{family}_XF2"].to_numpy(float)
        xf3 = data[f"{family}_XF3"].to_numpy(float)
        xf6 = data[f"{family}_XF6"].to_numpy(float)
        xf7 = data[f"{family}_XF7"].to_numpy(float)
        xf8 = data[f"{family}_XF8"].to_numpy(float)
        if not np.allclose(xf8, xf3 - xf2, atol=1e-8, equal_nan=True):
            raise ValueError(f"{family}: XF8 != XF3 - XF2")
        if not np.allclose(xf8, xf6 + xf7, atol=1e-8, equal_nan=True):
            raise ValueError(f"{family}: XF8 != XF6 + XF7")

    x_ft = data.loc[:, FT_FEATURES].to_numpy(float)
    x_mwd = data.loc[:, MWD_FEATURES].to_numpy(float)
    y = data["Rc"].to_numpy(float)
    groups = data["group_id"].astype(str).to_numpy()
    return data, x_mwd, x_ft, y, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft", type=Path, default=BASE_DIR / "data.csv")
    parser.add_argument("--mwd", type=Path, default=BASE_DIR / "mwd_features.csv")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    ft_path = Path(args.ft)
    mwd_path = Path(args.mwd)
    missing_paths = [str(path) for path in (ft_path, mwd_path) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Pre-generated demonstration data are missing: "
            + ", ".join(missing_paths)
        )

    data, x_mwd, x_ft, y, groups = load_data(ft_path, mwd_path)
    split_index = split_groups(
        data,
        train_fraction=args.train_fraction,
        val_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    model = TunnelStrengthModel(Config(epochs=args.epochs, seed=args.seed))
    model.fit(
        x_mwd[split_index["train"]],
        x_ft[split_index["train"]],
        y[split_index["train"]],
        mwd_val=x_mwd[split_index["validation"]],
        ft_val=x_ft[split_index["validation"]],
        y_val=y[split_index["validation"]],
    )

    split_metrics: dict[str, dict[str, float]] = {}
    split_groups_count: dict[str, int] = {}
    prediction_rows = []
    for split_name in ("train", "validation", "test"):
        index = split_index[split_name]
        output = model.predict(x_mwd[index], x_ft[index])
        prediction = output["prediction"].reshape(-1)
        split_metrics[split_name] = metrics(y[index], prediction)
        split_groups_count[split_name] = int(
            pd.Series(groups[index]).nunique()
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "sample_id": data.iloc[index]["sample_id"].to_numpy(),
                    "group_id": groups[index],
                    "split": split_name,
                    "observed_Rc": y[index],
                    "predicted_Rc": prediction,
                    "physics_baseline": output["physics"].reshape(-1),
                    "dynamic_residual": output["residual"].reshape(-1),
                }
            )
        )
        print(
            f"{split_name.capitalize()}: "
            f"R2={split_metrics[split_name]['R2']:.4f}, "
            f"MAE={split_metrics[split_name]['MAE']:.4f}, "
            f"RMSE={split_metrics[split_name]['RMSE']:.4f}"
        )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )

    summary = {
        "status": "passed",
        "samples": len(data),
        "ft_features": len(FT_FEATURES),
        "mwd_features": len(MWD_FEATURES),
        "groups": int(data["group_id"].nunique()),
        "max_epochs": args.epochs,
        "seed": args.seed,
        "split_protocol": {
            "method": "single stratified group-held-out split",
            "fractions": {
                "train": args.train_fraction,
                "validation": args.validation_fraction,
                "test": args.test_fraction,
            },
            "groups": split_groups_count,
            "samples": {
                name: int(len(index)) for name, index in split_index.items()
            },
        },
        "metrics": split_metrics,
        "training_control": {
            "optimizer": "AdamW",
            "learning_rate": Config().learning_rate,
            "batch_size": Config().batch_size,
            "early_stopping": "validation MSE",
            "patience": Config().early_stopping_patience,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nFINAL TEST RESULT")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
