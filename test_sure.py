from ptbenchmark.src import tune_perstree_anisodiff_niter_sure_gaussian


DATASET = "CBSD68"
DATASETS_DIR = "datasets"
NOISE_LEVELS_255 = [5, 10, 15, 25, 35, 50]
NITER_GRID = [1, 2, 3, 5, 10, 15, 20, 25, 30]
MAX_IMAGES = 12
SEED = 123


def mean_std(summary, metric):
    return (
        summary[f"delta_{metric}_mean"],
        summary[f"delta_{metric}_std"],
    )


def format_mean_std(mean, std, precision=4):
    return f"{mean:.{precision}g} +/- {std:.{precision}g}"


def print_table(rows):
    headers = [
        "noise",
        "sigma",
        "sure_niter",
        "oracle_niter",
        "d_niter",
        "d_mse",
        "d_psnr",
        "d_ssim",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def run_noise_level(noise_level_255):
    noise_sd = noise_level_255 / 255.0
    best, summary, rows = tune_perstree_anisodiff_niter_sure_gaussian(
        DATASET,
        datasets_dir=DATASETS_DIR,
        niter_grid=NITER_GRID,
        noise_mean=0.0,
        noise_sd=noise_sd,
        max_images=MAX_IMAGES,
        h=1e-3,
        seed=SEED,
        csv_file=f"results/sure_niter_{DATASET}_gaussian_sigma{noise_level_255}.csv",
    )

    oracle_best = min(summary, key=lambda row: row["mean_mse"])
    comparison = best["sure_vs_oracle"]
    d_niter = mean_std(comparison, "niter")
    d_mse = mean_std(comparison, "mse")
    d_psnr = mean_std(comparison, "psnr")
    d_ssim = mean_std(comparison, "ssim")

    return {
        "noise": noise_level_255,
        "sigma": f"{noise_sd:.5f}",
        "sure_niter": best["niter"],
        "oracle_niter": oracle_best["niter"],
        "d_niter": format_mean_std(*d_niter, precision=3),
        "d_mse": format_mean_std(*d_mse, precision=3),
        "d_psnr": format_mean_std(*d_psnr, precision=3),
        "d_ssim": format_mean_std(*d_ssim, precision=3),
    }


if __name__ == "__main__":
    print(f"Dataset: {DATASET}")
    print(f"Images per noise level: {MAX_IMAGES}")
    print(f"Iteration grid: {NITER_GRID}")
    print("Images are normalized to [0, 1], so sigma = noise / 255.\n")

    table_rows = []
    for noise_level in NOISE_LEVELS_255:
        print(f"Running noise={noise_level} (sigma={noise_level / 255.0:.5f})...")
        table_rows.append(run_noise_level(noise_level))

    print("\nSURE - ORACLE, per image")
    print_table(table_rows)
