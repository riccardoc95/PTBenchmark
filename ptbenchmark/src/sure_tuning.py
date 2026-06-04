from __future__ import annotations

import csv
import os
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .dataset import Dataset
from .perstrees import PersTree
from .perstrees_anisodiff import lifetime_denoise_tree


def _as_list(values, cast):
    if isinstance(values, str):
        return [cast(v.strip()) for v in values.split(",") if v.strip()]
    return [cast(v) for v in values]


def _estimate_noise_sd(img):
    img = np.asarray(img, dtype=float)
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    diffs = np.concatenate([dx.ravel(), dy.ravel()])
    return float(np.median(np.abs(diffs - np.median(diffs))) / 0.6745 / np.sqrt(2))


def _image_metrics(clean, rec):
    data_range = float(clean.max() - clean.min())
    if data_range == 0.0:
        data_range = 1.0
    return {
        "mse": float(np.mean((clean - rec) ** 2)),
        "psnr": float(psnr(clean, rec, data_range=data_range)),
        "ssim": float(ssim(clean, rec, data_range=data_range)),
    }


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def summarize_sure_oracle_differences(rows):
    """
    Compare per-image SURE-selected niter against oracle-MSE-selected niter.

    Differences are computed as ``SURE choice - oracle choice``.
    """
    image_keys = sorted({(row["dataset"], row["subset"], row["image"]) for row in rows})
    comparisons = []

    for key in image_keys:
        image_rows = [
            row
            for row in rows
            if (row["dataset"], row["subset"], row["image"]) == key
        ]
        sure_row = min(image_rows, key=lambda row: row["sure"])
        oracle_row = min(image_rows, key=lambda row: row["mse"])
        comparisons.append(
            {
                "dataset": key[0],
                "subset": key[1],
                "image": key[2],
                "niter_sure": int(sure_row["niter"]),
                "niter_oracle": int(oracle_row["niter"]),
                "delta_niter": int(sure_row["niter"] - oracle_row["niter"]),
                "mse_sure": float(sure_row["mse"]),
                "mse_oracle": float(oracle_row["mse"]),
                "delta_mse": float(sure_row["mse"] - oracle_row["mse"]),
                "psnr_sure": float(sure_row["psnr"]),
                "psnr_oracle": float(oracle_row["psnr"]),
                "delta_psnr": float(sure_row["psnr"] - oracle_row["psnr"]),
                "ssim_sure": float(sure_row["ssim"]),
                "ssim_oracle": float(oracle_row["ssim"]),
                "delta_ssim": float(sure_row["ssim"] - oracle_row["ssim"]),
            }
        )

    summary = {"n_images": len(comparisons)}
    for metric in ("niter", "mse", "psnr", "ssim"):
        values = [row[f"delta_{metric}"] for row in comparisons]
        mean_value, std_value = _mean_std(values)
        summary[f"delta_{metric}_mean"] = mean_value
        summary[f"delta_{metric}_std"] = std_value

    return summary, comparisons


def perstree_anisodiff_fixed_iter(
    img,
    niter,
    lifetime_t=None,
    cut=False,
    cut_mode="nearest",
    base_relax_scale=1e-5,
    guided_radius=1,
    epsilon_scale=5e-4,
):
    """Run PerSTree anisotropic diffusion for exactly ``niter`` iterations."""
    img = np.asarray(img, dtype=float)
    tree = PersTree(img, lifetime_t=lifetime_t, cut=cut, cut_mode=cut_mode)

    values = tree.features[:, 1].copy()
    lifetimes = tree.features[:, -1].copy()
    rec = img.copy()

    for _ in range(int(niter)):
        rec = lifetime_denoise_tree(
            values,
            lifetimes,
            tree.parent,
            tree.child_index,
            tree.children_all,
            tree.rows,
            tree.cols,
            num_iter=1,
            base_relax_scale=base_relax_scale,
            guided_radius=int(guided_radius),
            epsilon_scale=epsilon_scale,
        )
        values = rec.ravel()

    return np.clip(rec, 0.0, 1.0)


def perstree_anisodiff_sure(
    img,
    noise_sd=None,
    niter=10,
    h=1e-3,
    seed=123,
    mc_samples=30,
    lifetime_t=None,
    cut=False,
    cut_mode="nearest",
    base_relax_scale=1e-5,
    guided_radius=1,
    epsilon_scale=5e-4,
):
    """Monte Carlo SURE estimate for one fixed PerSTree anisodiff iteration count."""
    sure, _ = _perstree_anisodiff_sure_with_rec(
        img,
        noise_sd=noise_sd,
        niter=niter,
        h=h,
        seed=seed,
        mc_samples=mc_samples,
        lifetime_t=lifetime_t,
        cut=cut,
        cut_mode=cut_mode,
        base_relax_scale=base_relax_scale,
        guided_radius=guided_radius,
        epsilon_scale=epsilon_scale,
    )
    return sure


def _perstree_anisodiff_sure_with_rec(
    img,
    noise_sd=None,
    niter=10,
    h=1e-3,
    seed=123,
    mc_samples=30,
    lifetime_t=None,
    cut=False,
    cut_mode="nearest",
    base_relax_scale=1e-5,
    guided_radius=1,
    epsilon_scale=5e-4,
):
    img = np.asarray(img, dtype=float)
    sigma = _estimate_noise_sd(img) if noise_sd is None else float(noise_sd)
    rng = np.random.default_rng(seed)

    denoised = perstree_anisodiff_fixed_iter(
        img,
        niter=niter,
        lifetime_t=lifetime_t,
        cut=cut,
        cut_mode=cut_mode,
        base_relax_scale=base_relax_scale,
        guided_radius=guided_radius,
        epsilon_scale=epsilon_scale,
    )

    divergences = []
    for _ in range(int(mc_samples)):
        z = rng.normal(size=img.shape)
        denoised_perturbed = perstree_anisodiff_fixed_iter(
            img + h * z,
            niter=niter,
            lifetime_t=lifetime_t,
            cut=cut,
            cut_mode=cut_mode,
            base_relax_scale=base_relax_scale,
            guided_radius=guided_radius,
            epsilon_scale=epsilon_scale,
        )
        divergences.append(float(np.sum(z * (denoised_perturbed - denoised)) / h))

    divergence = float(np.mean(divergences))
    residual = float(np.sum((denoised - img) ** 2))
    sure = residual + 2.0 * sigma**2 * divergence - img.size * sigma**2
    return sure / img.size, denoised


def tune_perstree_anisodiff_niter_sure(
    dataset_names,
    datasets_dir="datasets",
    subsets=None,
    niter_grid=(1, 2, 3, 5, 10, 15, 20),
    max_images=None,
    noise_sd=None,
    h=1e-3,
    seed=123,
    mc_samples=30,
    lifetime_t=None,
    cut=False,
    cut_mode="nearest",
    base_relax_scale=1e-5,
    guided_radius=1,
    epsilon_scale=5e-4,
    csv_file=None,
):
    """
    Tune only the number of PerSTree anisodiff iterations using Monte Carlo SURE.

    The returned ``best`` entry is selected by mean SURE over the requested
    datasets/subsets/images. Ground truth is loaded only to report MSE diagnostics.
    """
    if isinstance(dataset_names, str):
        selected_datasets = [d.strip() for d in dataset_names.split(",") if d.strip()]
    else:
        selected_datasets = list(dataset_names)
    niter_grid = _as_list(niter_grid, int)

    rows = []
    for dataset_name in selected_datasets:
        dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
        available_subsets = dataset.get_subsets()
        if subsets is None:
            selected_subsets = available_subsets[:1]
        elif isinstance(subsets, str):
            selected_subsets = [s.strip() for s in subsets.split(",") if s.strip()]
        else:
            selected_subsets = list(subsets)

        missing = sorted(set(selected_subsets) - set(available_subsets))
        if missing:
            raise ValueError(f"Unknown subsets for {dataset_name}: {missing}")

        for subset in selected_subsets:
            dataset.set_subset(subset)
            items = [dataset[i] for i in range(len(dataset))]
            if max_images is not None:
                items = items[: int(max_images)]

            for image_index, (image_name, img, gth) in enumerate(items):
                sigma = _estimate_noise_sd(img) if noise_sd is None else float(noise_sd)
                for niter in niter_grid:
                    t0 = time.time()
                    sure, rec = _perstree_anisodiff_sure_with_rec(
                        img,
                        noise_sd=sigma,
                        niter=niter,
                        h=h,
                        seed=seed + image_index,
                        mc_samples=mc_samples,
                        lifetime_t=lifetime_t,
                        cut=cut,
                        cut_mode=cut_mode,
                        base_relax_scale=base_relax_scale,
                        guided_radius=guided_radius,
                        epsilon_scale=epsilon_scale,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "subset": subset,
                            "image": image_name,
                            "niter": int(niter),
                            "sure": float(sure),
                            "mse": float(np.mean((gth - rec) ** 2)),
                            "noise_sd": sigma,
                            "time": time.time() - t0,
                        }
                    )

    if not rows:
        raise ValueError("No images were selected for SURE tuning.")

    summary = []
    for niter in niter_grid:
        niter_rows = [row for row in rows if row["niter"] == niter]
        summary.append(
            {
                "niter": int(niter),
                "mean_sure": float(np.mean([row["sure"] for row in niter_rows])),
                "std_sure": float(np.std([row["sure"] for row in niter_rows])),
                "mean_mse": float(np.mean([row["mse"] for row in niter_rows])),
                "std_mse": float(np.std([row["mse"] for row in niter_rows])),
                "n_images": len(niter_rows),
            }
        )

    best = dict(min(summary, key=lambda row: row["mean_sure"]))

    if csv_file is not None:
        directory = os.path.dirname(csv_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(csv_file, "w", newline="") as f:
            fieldnames = ["dataset", "subset", "image", "niter", "sure", "mse", "noise_sd", "time"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return best, summary, rows


def tune_perstree_anisodiff_niter_sure_gaussian(
    dataset_names,
    datasets_dir="datasets",
    niter_grid=(1, 2, 3, 5, 10, 15, 20),
    noise_mean=0.0,
    noise_sd=0.1,
    max_images=None,
    h=1e-3,
    seed=123,
    mc_samples=30,
    lifetime_t=None,
    cut=False,
    cut_mode="nearest",
    base_relax_scale=1e-5,
    guided_radius=1,
    epsilon_scale=5e-4,
    csv_file=None,
):
    """
    Tune PerSTree anisodiff iterations with SURE on synthetic Gaussian noise.

    Clean images are read from the ``original`` group of each dataset, Gaussian
    noise with known mean/std is added, and SURE receives the known ``noise_sd``.
    """
    if isinstance(dataset_names, str):
        selected_datasets = [d.strip() for d in dataset_names.split(",") if d.strip()]
    else:
        selected_datasets = list(dataset_names)
    niter_grid = _as_list(niter_grid, int)
    rng = np.random.default_rng(seed)

    rows = []
    for dataset_name in selected_datasets:
        dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
        image_names = dataset.get_labels()
        if max_images is not None:
            image_names = image_names[: int(max_images)]

        for image_index, image_name in enumerate(image_names):
            clean = dataset.get_gth_from_label(image_name)
            noisy = clean + rng.normal(noise_mean, noise_sd, size=clean.shape)
            noisy = np.clip(noisy, 0.0, 1.0)
            noisy_metrics = _image_metrics(clean, noisy)

            for niter in niter_grid:
                t0 = time.time()
                sure, rec = _perstree_anisodiff_sure_with_rec(
                    noisy,
                    noise_sd=noise_sd,
                    niter=niter,
                    h=h,
                    seed=seed + image_index,
                    mc_samples=mc_samples,
                    lifetime_t=lifetime_t,
                    cut=cut,
                    cut_mode=cut_mode,
                    base_relax_scale=base_relax_scale,
                    guided_radius=guided_radius,
                    epsilon_scale=epsilon_scale,
                )
                rec_metrics = _image_metrics(clean, rec)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "subset": "synthetic_gaussian",
                        "image": image_name,
                        "niter": int(niter),
                        "sure": float(sure),
                        "mse": rec_metrics["mse"],
                        "psnr": rec_metrics["psnr"],
                        "ssim": rec_metrics["ssim"],
                        "noisy_mse": noisy_metrics["mse"],
                        "noisy_psnr": noisy_metrics["psnr"],
                        "noisy_ssim": noisy_metrics["ssim"],
                        "noise_mean": float(noise_mean),
                        "noise_sd": float(noise_sd),
                        "time": time.time() - t0,
                    }
                )

    if not rows:
        raise ValueError("No images were selected for Gaussian SURE tuning.")

    comparison_summary, comparisons = summarize_sure_oracle_differences(rows)

    summary = []
    for niter in niter_grid:
        niter_rows = [row for row in rows if row["niter"] == niter]
        summary.append(
            {
                "niter": int(niter),
                "mean_sure": float(np.mean([row["sure"] for row in niter_rows])),
                "std_sure": float(np.std([row["sure"] for row in niter_rows])),
                "mean_mse": float(np.mean([row["mse"] for row in niter_rows])),
                "std_mse": float(np.std([row["mse"] for row in niter_rows])),
                "mean_psnr": float(np.mean([row["psnr"] for row in niter_rows])),
                "std_psnr": float(np.std([row["psnr"] for row in niter_rows])),
                "mean_ssim": float(np.mean([row["ssim"] for row in niter_rows])),
                "std_ssim": float(np.std([row["ssim"] for row in niter_rows])),
                "mean_noisy_mse": float(np.mean([row["noisy_mse"] for row in niter_rows])),
                "mean_noisy_psnr": float(np.mean([row["noisy_psnr"] for row in niter_rows])),
                "mean_noisy_ssim": float(np.mean([row["noisy_ssim"] for row in niter_rows])),
                "n_images": len(niter_rows),
            }
        )

    best = dict(min(summary, key=lambda row: row["mean_sure"]))
    best["sure_vs_oracle"] = comparison_summary

    if csv_file is not None:
        directory = os.path.dirname(csv_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(csv_file, "w", newline="") as f:
            fieldnames = [
                "dataset",
                "subset",
                "image",
                "niter",
                "sure",
                "mse",
                "psnr",
                "ssim",
                "noisy_mse",
                "noisy_psnr",
                "noisy_ssim",
                "noise_mean",
                "noise_sd",
                "time",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

            comparison_file = os.path.splitext(csv_file)[0] + "_sure_vs_oracle.csv"
            with open(comparison_file, "w", newline="") as comparison_f:
                comparison_fieldnames = [
                    "dataset",
                    "subset",
                    "image",
                    "niter_sure",
                    "niter_oracle",
                    "delta_niter",
                    "mse_sure",
                    "mse_oracle",
                    "delta_mse",
                    "psnr_sure",
                    "psnr_oracle",
                    "delta_psnr",
                    "ssim_sure",
                    "ssim_oracle",
                    "delta_ssim",
                ]
                comparison_writer = csv.DictWriter(
                    comparison_f, fieldnames=comparison_fieldnames
                )
                comparison_writer.writeheader()
                comparison_writer.writerows(comparisons)

    return best, summary, rows
