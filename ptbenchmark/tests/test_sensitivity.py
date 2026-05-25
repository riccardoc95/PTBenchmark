import csv
import itertools
import json
import os
import time

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from ptbenchmark.src import Dataset, pt_denoise, pt_denoise_sec


def _parse_number_list(value, cast=float):
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def _format_float(value):
    return f"{value:.0e}".replace("+", "")


def _sensitivity_method_name(method, base_relax_scale, guided_radius, epsilon_scale, stop_threshold):
    name = (
        f"{method}_sens_br{_format_float(base_relax_scale)}"
        f"_r{guided_radius}_eps{_format_float(epsilon_scale)}"
    )
    if method == "entropy":
        name += f"_thr{_format_float(stop_threshold)}"
    return name


def _save_to_h5(output_file, dataset_name, subset, image_name, method_name, data_dict):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "a") as f:
        img_group = f.require_group(dataset_name).require_group(subset).require_group(image_name)
        if method_name in img_group:
            del img_group[method_name]
        method_group = img_group.create_group(method_name)
        method_group.create_dataset("rec", data=data_dict["rec"], compression="gzip")
        method_group.attrs["mse"] = data_dict["mse"]
        method_group.attrs["psnr"] = data_dict["psnr"]
        method_group.attrs["ssim"] = data_dict["ssim"]
        method_group.attrs["time"] = data_dict["time"]
        method_group.attrs["params"] = json.dumps(data_dict["params"])


def _save_summary(output_file, dataset_name, subset, method_name, rows):
    metrics = {
        "mean_mse": np.mean([row["mse"] for row in rows]),
        "var_mse": np.var([row["mse"] for row in rows]),
        "mean_psnr": np.mean([row["psnr"] for row in rows]),
        "var_psnr": np.var([row["psnr"] for row in rows]),
        "mean_ssim": np.mean([row["ssim"] for row in rows]),
        "var_ssim": np.var([row["ssim"] for row in rows]),
        "mean_time": np.mean([row["time"] for row in rows]),
        "var_time": np.var([row["time"] for row in rows]),
        "mean_niter": np.mean([row["niter"] for row in rows]),
        "var_niter": np.var([row["niter"] for row in rows]),
    }
    with h5py.File(output_file, "a") as f:
        stats_group = f.require_group(dataset_name).require_group(subset).require_group(f"{method_name}_stats")
        for key, value in metrics.items():
            stats_group.attrs[key] = float(value)
    return metrics


def _append_csv(csv_file, rows):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    fieldnames = [
        "dataset",
        "subset",
        "image",
        "method",
        "method_name",
        "base_relax_scale",
        "guided_radius",
        "epsilon_scale",
        "stop_threshold",
        "mse",
        "psnr",
        "ssim",
        "time",
        "niter",
        "stop_reason",
    ]
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_sensitivity(
    dataset_name,
    datasets_dir,
    output_file,
    csv_file,
    method="perstree",
    base_relax_scales="1e-6,1e-5,1e-4",
    guided_radii="1,2,3",
    epsilon_scales="1e-4,5e-4,1e-3",
    stop_thresholds="1e-5,1e-4,1e-3",
    max_iter=100,
    max_images=None,
):
    if method not in {"perstree", "entropy"}:
        raise ValueError("Sensitivity method must be 'perstree' or 'entropy'.")

    base_relax_scales = _parse_number_list(base_relax_scales, float)
    guided_radii = _parse_number_list(guided_radii, int)
    epsilon_scales = _parse_number_list(epsilon_scales, float)
    stop_thresholds = _parse_number_list(stop_thresholds, float)
    if method == "perstree":
        stop_thresholds = [np.nan]

    dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
    all_rows = []

    for subset in dataset.get_subsets():
        dataset.set_subset(subset)
        subset_items = list(dataset)
        if max_images is not None:
            subset_items = subset_items[:max_images]

        grid = itertools.product(base_relax_scales, guided_radii, epsilon_scales, stop_thresholds)
        for base_relax_scale, guided_radius, epsilon_scale, stop_threshold in grid:
            method_name = _sensitivity_method_name(
                method, base_relax_scale, guided_radius, epsilon_scale, stop_threshold
            )
            rows = []

            for image_name, img, gth in tqdm(subset_items, desc=f"{dataset_name}/{subset}/{method_name}"):
                t0 = time.time()
                if method == "entropy":
                    rec, mse, params = pt_denoise_sec(
                        img,
                        gth,
                        max_iter=max_iter,
                        stop_threshold=stop_threshold,
                        cut=False,
                        base_relax_scale=base_relax_scale,
                        guided_radius=guided_radius,
                        epsilon_scale=epsilon_scale,
                    )
                else:
                    rec, mse, params = pt_denoise(
                        img,
                        gth,
                        max_iter=max_iter,
                        cut=False,
                        base_relax_scale=base_relax_scale,
                        guided_radius=guided_radius,
                        epsilon_scale=epsilon_scale,
                    )
                elapsed = time.time() - t0
                psnr_val = float(psnr(gth, rec, data_range=gth.max() - gth.min()))
                ssim_val = float(ssim(gth, rec, data_range=gth.max() - gth.min()))
                params["total_time"] = elapsed

                row = {
                    "dataset": dataset_name,
                    "subset": subset,
                    "image": image_name,
                    "method": method,
                    "method_name": method_name,
                    "base_relax_scale": base_relax_scale,
                    "guided_radius": guided_radius,
                    "epsilon_scale": epsilon_scale,
                    "stop_threshold": stop_threshold,
                    "mse": float(mse),
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "time": float(params.get("time", elapsed)),
                    "niter": int(params.get("niter", -1)),
                    "stop_reason": params.get("stop_reason", "oracle_mse"),
                }
                rows.append(row)
                all_rows.append(row)

                _save_to_h5(
                    output_file,
                    dataset_name,
                    subset,
                    image_name,
                    method_name,
                    {
                        "rec": rec,
                        "mse": float(mse),
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "time": float(params.get("time", elapsed)),
                        "params": params,
                    },
                )

            _save_summary(output_file, dataset_name, subset, method_name, rows)
            _append_csv(csv_file, rows)

    return all_rows
