# ============================================================
# build_results_and_best_images.py
# ============================================================
import os
import re
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

# === Percorsi ===
H5_PATH = "sbatch/merged.h5"
OUTPUT_JSON = "sbatch/results_summary.json"
OUTPUT_IMG_DIR = "paper/plots_best"
os.makedirs("sbatch", exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# === Parametri globali ===
methods = [
    "perstree", "perstree_cut", "perstree_rec_sec", "peronamalik", "median",
    "gaussian", "wavelet", "nlmeans", "bm3d", "unet", "dncnn",
    "nafnet", "hirdiff", "restormer"
]
zscaler = ZScaleInterval()


def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def build_results_and_images(h5_path, output_json, output_img_dir):
    results = {}

    with h5py.File(h5_path, "r") as f:
        for dataset_name in f.keys():
            dataset_group = f[dataset_name]
            results[dataset_name] = {}

            subsets = sorted(
                dataset_group.keys(),
                key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
            )

            for subset_name in subsets:
                subset_group = dataset_group[subset_name]
                if not isinstance(subset_group, h5py.Group):
                    continue

                results[dataset_name][subset_name] = {}

                # --- Metriche standard ---
                for method in methods:
                    stats_path = f"{dataset_name}/{subset_name}/{method}_stats"
                    if stats_path not in f:
                        continue
                    grp = f[stats_path]
                    attrs = grp.attrs

                    mean_vals = {}
                    std_vals = {}
                    for key, val in attrs.items():
                        if key.startswith("mean_"):
                            metric = key.replace("mean_", "").upper()
                            mean_vals[metric] = float(val)
                        elif key.startswith("var_"):
                            metric = key.replace("var_", "").upper()
                            std_vals[metric] = float(np.sqrt(val))

                    for m in ["MSE", "PSNR", "SSIM", "TIME", "TRAINING_TIME"]:
                        mean_vals.setdefault(m, np.nan)
                        std_vals.setdefault(m, np.nan)

                    results[dataset_name][subset_name][method] = {
                        "mean": mean_vals,
                        "std": std_vals,
                    }

                # --- DISTANCES ---
                for dist in ["RD", "RF"]:
                    dist_path = f"{dataset_name}/{subset_name}/pt_distance_{dist}_stats"
                    if dist_path not in f:
                        continue
                    grp = f[dist_path]
                    attrs = grp.attrs
                    results[dataset_name][subset_name][f"distance_{dist}"] = {
                        "mean": {
                            "NOISY": float(attrs.get("noisy_mean", np.nan)),
                            "MJUMP": float(attrs.get("mjump_mean", np.nan)),
                            "OPTUNA": float(attrs.get("optuna_mean", np.nan))
                        },
                        "std": {
                            "NOISY": float(np.sqrt(attrs.get("noisy_var", 0))),
                            "MJUMP": float(np.sqrt(attrs.get("mjump_var", 0))),
                            "OPTUNA": float(np.sqrt(attrs.get("optuna_var", 0)))
                        }
                    }

                # --- BEST IMAGE (PerSTree) ---
                best_image = None
                best_score = -np.inf
                for img_name, img_group in subset_group.items():
                    if "perstree" in img_group:
                        psnr_val = img_group["perstree"].attrs.get("psnr", np.nan)
                        if psnr_val > best_score:
                            best_score = psnr_val
                            best_image = img_name

                if best_image and "perstree" in subset_group[best_image]:
                    rec = np.array(subset_group[best_image]["perstree"]["rec"])
                    rec = normalize(rec)
                    fig_path = os.path.join(output_img_dir, f"{dataset_name}_{subset_name}_{best_image}.png")
                    plt.imshow(rec, cmap="gray", vmin=0, vmax=1)
                    plt.axis("off")
                    plt.title(f"{dataset_name} - {subset_name} - {best_image}")
                    plt.tight_layout()
                    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    results[dataset_name][subset_name]["best_image"] = {
                        "name": best_image,
                        "psnr": float(best_score),
                        "path": fig_path
                    }
                else:
                    results[dataset_name][subset_name]["best_image"] = {
                        "name": "N/A", "psnr": np.nan, "path": None
                    }

    with open(output_json, "w") as f_json:
        json.dump(results, f_json, indent=4)

    print(f"\nRisultati e immagini salvati in:\n  - JSON: {output_json}\n  - Images: {output_img_dir}")
    return results


if __name__ == "__main__":
    build_results_and_images(H5_PATH, OUTPUT_JSON, OUTPUT_IMG_DIR)
