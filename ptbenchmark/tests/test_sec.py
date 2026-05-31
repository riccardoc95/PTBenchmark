import os
import h5py
import json
import time
import numpy as np
from tqdm import tqdm

from ptbenchmark.src import Dataset
from ptbenchmark.src import pt_denoise_sec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def save_to_h5(output_file, dataset_name, subset, image_name, method_name, data_dict):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "a") as f:
        ds_group = f.require_group(dataset_name)
        subset_group = ds_group.require_group(subset)
        img_group = subset_group.require_group(image_name)

        if method_name in img_group:
            del img_group[method_name]

        method_group = img_group.create_group(method_name)
        method_group.create_dataset("rec", data=data_dict["rec"], compression="gzip")
        method_group.attrs["mse"] = data_dict["mse"]
        method_group.attrs["psnr"] = data_dict.get("psnr", np.nan)
        method_group.attrs["ssim"] = data_dict.get("ssim", np.nan)
        method_group.attrs["params"] = json.dumps(data_dict.get("params", {}))
        method_group.attrs["time"] = data_dict["time"]

def save_metrics_summary(output_file, dataset_name, subset, method_name, metrics):
    with h5py.File(output_file, "a") as f:
        ds_group = f.require_group(dataset_name)
        subset_group = ds_group.require_group(subset)
        stats_group = subset_group.require_group(f"{method_name}_stats")

        for k, v in metrics.items():
            stats_group.attrs[k] = float(v)


def test_sec(dataset_name, datasets_dir, output_file):
    method_name = "perstree_rec_sec"
    dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
    for subset in dataset.get_subsets():
        dataset.set_subset(subset)

        all_mse, all_psnr, all_ssim = [], [], []
        all_time, all_training_time = [], []

        for image_name, img, gth in tqdm(dataset, desc=f"{dataset_name}/{subset}"):
            t0 = time.time()
            rec, mse, params = pt_denoise_sec(
                img, gth, max_iter=100, lifetime_t=None, cut=False
            )
            elapsed = time.time() - t0
            psnr_val = float(psnr(gth, rec, data_range=gth.max() - gth.min()))
            ssim_val = float(ssim(gth, rec, data_range=gth.max() - gth.min()))

            all_mse.append(mse)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_time.append(params['time'])
            all_training_time.append(elapsed)

            save_to_h5(
                output_file,
                dataset_name,
                subset,
                image_name,
                method_name,
                {
                    "rec": rec,
                    "mse": mse,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "params": params,
                    "training_time": elapsed,
                    "time": params['time']},
            )



        # Aggregate statistics.
        metrics = {
            "mean_mse": np.mean(all_mse),
            "var_mse": np.var(all_mse),
            "mean_psnr": np.mean(all_psnr),
            "var_psnr": np.var(all_psnr),
            "mean_ssim": np.mean(all_ssim),
            "var_ssim": np.var(all_ssim),
            "mean_time": np.mean(all_time),
            "var_time": np.var(all_time),
            "mean_training_time": np.mean(all_training_time),
            "var_training_time": np.var(all_training_time),
        }

        save_metrics_summary(output_file, dataset_name, subset, method_name, metrics)

        print(f"\nAggregate statistics for subset '{subset}' ({method_name}):")
        print(f"   MSE  mean={metrics['mean_mse']:.6f}, var={metrics['var_mse']:.6f}")
        print(f"   PSNR mean={metrics['mean_psnr']:.3f}, var={metrics['var_psnr']:.3f}")
        print(f"   SSIM mean={metrics['mean_ssim']:.3f}, var={metrics['var_ssim']:.3f}\n")
        print(f"   TIME mean={metrics['mean_time']:.4f}s, var={metrics['var_time']:.4f}")
        print(f"   TRAIN mean={metrics['mean_training_time']:.4f}s, var={metrics['var_training_time']:.4f}\n")



if __name__ == "__main__":
    test_sec(
        datasets_dir="datasets",
        dataset_name="FORECAST",
        output_file="results/results_all.h5"
    )
