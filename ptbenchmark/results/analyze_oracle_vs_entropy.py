import argparse
import csv
import json
import os

import h5py
import numpy as np


METRICS = ("mse", "psnr", "ssim", "time", "niter")


def _read_method(group):
    params = json.loads(group.attrs.get("params", "{}"))
    return {
        "mse": float(group.attrs.get("mse", np.nan)),
        "psnr": float(group.attrs.get("psnr", np.nan)),
        "ssim": float(group.attrs.get("ssim", np.nan)),
        "time": float(group.attrs.get("time", np.nan)),
        "niter": float(params.get("niter", np.nan)),
        "stop_reason": params.get("stop_reason", ""),
    }


def analyze(input_file, output_dir, oracle_method="perstree", entropy_method="perstree_rec_sec"):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    with h5py.File(input_file, "r") as f:
        for dataset_name, dataset_group in f.items():
            for subset_name, subset_group in dataset_group.items():
                if subset_name.endswith("_stats"):
                    continue
                for image_name, image_group in subset_group.items():
                    if image_name.endswith("_stats"):
                        continue
                    if oracle_method not in image_group or entropy_method not in image_group:
                        continue

                    oracle = _read_method(image_group[oracle_method])
                    entropy = _read_method(image_group[entropy_method])
                    row = {
                        "dataset": dataset_name,
                        "subset": subset_name,
                        "image": image_name,
                        "entropy_stop_reason": entropy["stop_reason"],
                    }
                    for metric in METRICS:
                        row[f"oracle_{metric}"] = oracle[metric]
                        row[f"entropy_{metric}"] = entropy[metric]
                        row[f"delta_{metric}"] = entropy[metric] - oracle[metric]
                    rows.append(row)

    detail_path = os.path.join(output_dir, "oracle_vs_entropy_detail.csv")
    fieldnames = [
        "dataset",
        "subset",
        "image",
        "entropy_stop_reason",
        *[f"{prefix}_{metric}" for metric in METRICS for prefix in ("oracle", "entropy", "delta")],
    ]
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    keys = sorted({(row["dataset"], row["subset"]) for row in rows})
    for dataset_name, subset_name in keys:
        subset_rows = [row for row in rows if row["dataset"] == dataset_name and row["subset"] == subset_name]
        summary = {
            "dataset": dataset_name,
            "subset": subset_name,
            "n_images": len(subset_rows),
        }
        for metric in METRICS:
            for prefix in ("oracle", "entropy", "delta"):
                values = np.array([row[f"{prefix}_{metric}"] for row in subset_rows], dtype=float)
                summary[f"mean_{prefix}_{metric}"] = float(np.nanmean(values))
                summary[f"std_{prefix}_{metric}"] = float(np.nanstd(values))
        summary_rows.append(summary)

    summary_path = os.path.join(output_dir, "oracle_vs_entropy_summary.csv")
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else ["dataset", "subset", "n_images"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return detail_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Compare oracle MSE stopping against entropy stopping.")
    parser.add_argument("--input", required=True, help="Merged HDF5 result file.")
    parser.add_argument("--output-dir", default="paper/analysis", help="Directory for CSV outputs.")
    parser.add_argument("--oracle-method", default="perstree")
    parser.add_argument("--entropy-method", default="perstree_rec_sec")
    args = parser.parse_args()

    detail_path, summary_path = analyze(
        args.input,
        args.output_dir,
        oracle_method=args.oracle_method,
        entropy_method=args.entropy_method,
    )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
