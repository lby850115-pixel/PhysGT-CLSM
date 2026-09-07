"""Kruskal-Wallis and Dunn-BH tests for the manuscript 33-image analysis."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, norm, rankdata

ROOT = Path(__file__).resolve().parent
MODELS = ["physegt_clsm", "cellpose", "mitosegnet", "modl", "nellie", "mitometer"]
METRICS = [
    "mean_area",
    "mean_aspect_ratio",
    "mean_eccentricity",
    "mean_solidity",
    "mean_tortuosity",
    "mean_thickness",
]


def bh_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result.tolist()


def significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def dunn_test(groups: dict[str, np.ndarray]) -> list[dict]:
    names = list(groups)
    values = np.concatenate([groups[name] for name in names])
    labels = np.concatenate([[name] * len(groups[name]) for name in names])
    ranks = rankdata(values)
    n_total = len(values)
    _, counts = np.unique(values, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    rank_variance = n_total * (n_total + 1) / 12
    if n_total > 1:
        rank_variance -= tie_term / (12 * (n_total - 1))
    mean_ranks = {name: float(ranks[labels == name].mean()) for name in names}

    rows = []
    raw_p = []
    for left, right in combinations(names, 2):
        denominator = np.sqrt(
            rank_variance * (1 / len(groups[left]) + 1 / len(groups[right]))
        )
        z_value = abs(mean_ranks[left] - mean_ranks[right]) / denominator
        p_value = float(2 * norm.sf(z_value))
        rows.append({"model_a": left, "model_b": right, "z": z_value, "p_raw": p_value})
        raw_p.append(p_value)

    for row, adjusted in zip(rows, bh_adjust(raw_p)):
        row["p_bh"] = adjusted
        row["significance"] = significance(adjusted)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "reference_results" / "all_models_summary_33_images.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "statistics_33"),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise SystemExit(f"Morphology summary not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        if row["model"] == "physegt_clsm_p3":
            row["model"] = "physegt_clsm"

    observed = {row["model"] for row in records}
    missing_models = [model for model in MODELS if model not in observed]
    if missing_models:
        raise SystemExit(f"Missing models in morphology summary: {', '.join(missing_models)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    omnibus_rows = []
    physgt_rows = []

    for metric in METRICS:
        groups = {
            model: np.asarray(
                [float(row[metric]) for row in records if row["model"] == model],
                dtype=float,
            )
            for model in MODELS
        }
        sizes = {model: len(values) for model, values in groups.items()}
        if set(sizes.values()) != {33}:
            raise SystemExit(f"{metric}: expected 33 observations per model, found {sizes}")

        h_value, p_value = kruskal(*(groups[model] for model in MODELS))
        omnibus_rows.append(
            {
                "metric": metric,
                "H": float(h_value),
                "df": 5,
                "p": float(p_value),
                "significance": significance(p_value),
            }
        )

        dunn_rows = dunn_test(groups)
        for row in dunn_rows:
            row["metric"] = metric
        write_csv(output_dir / f"dunn_{metric}.csv", dunn_rows)
        for row in dunn_rows:
            if "physegt_clsm" in (row["model_a"], row["model_b"]):
                baseline = (
                    row["model_b"] if row["model_a"] == "physegt_clsm" else row["model_a"]
                )
                physgt_rows.append(
                    {
                        "metric": metric,
                        "baseline": baseline,
                        "p_bh": row["p_bh"],
                        "significance": row["significance"],
                    }
                )

    write_csv(output_dir / "kruskal_wallis.csv", omnibus_rows)
    write_csv(output_dir / "physgt_vs_baselines_dunn_bh.csv", physgt_rows)
    print(f"Saved statistical results to {output_dir}")


if __name__ == "__main__":
    main()
