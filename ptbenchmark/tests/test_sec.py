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


def save_perstree_sec_to_h5(output_file, dataset_name, subset, image_name, data_dict):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "a") as f:
        ds_group = f.require_group(dataset_name)
        subset_group = ds_group.require_group(subset)
        img_group = subset_group.require_group(image_name)

        group_name = "perstree_rec_sec"

        # Se esiste già, sovrascrive
        if group_name in img_group:
            del img_group[group_name]

        grp = img_group.create_group(group_name)
        grp.create_dataset("data", data=data_dict["rec"], compression="gzip")
        grp.attrs["mse"] = data_dict["mse"]
        grp.attrs["params"] = json.dumps(data_dict["params"])
        grp.attrs["training_time"] = data_dict["time"]


def test_sec(datasets_dir, dataset_name, output_file):

    dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
    all_mse, all_psnr, all_ssim = [], [], []

    for subset in dataset.get_subsets():
        dataset.set_subset(subset)
        print(f"Processing dataset: {dataset_name} | subset: {subset} | method: perstree_rec_sec")

        for image_name, img, gth in tqdm(dataset, desc=f"{dataset_name}/{subset}"):
            t0 = time.time()
            perstree_rec_sec, perstree_mse_sec, perstree_params_sec = pt_denoise_sec(
                img, gth, max_iter=100, lifetime_t=None, cut=False
            )
            elapsed = time.time() - t0

            psnr_val = float(psnr(gth, perstree_rec_sec, data_range=gth.max() - gth.min()))
            ssim_val = float(ssim(gth, perstree_rec_sec, data_range=gth.max() - gth.min()))

            all_mse.append(perstree_mse_sec)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)

            save_perstree_sec_to_h5(
                output_file,
                dataset_name,
                subset,
                image_name,
                {
                    "rec": perstree_rec_sec,
                    "mse": perstree_mse_sec,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "params": perstree_params_sec,
                    "time": elapsed,
                },
            )

        metrics = {
            "mean_mse": np.mean(all_mse),
            "var_mse": np.var(all_mse),
            "mean_psnr": np.mean(all_psnr),
            "var_psnr": np.var(all_psnr),
            "mean_ssim": np.mean(all_ssim),
            "var_ssim": np.var(all_ssim),
        }

        with h5py.File(output_file, "a") as f:
            ds_group = f.require_group(dataset_name)
            subset_group = ds_group.require_group(subset)
            stats_group = subset_group.require_group("perstree_rec_sec_stats")

            for k, v in metrics.items():
                stats_group.attrs[k] = float(v)

        print(f"\nStatistiche globali per subset '{subset}' (perstree_rec_sec):")
        print(f"   MSE  mean={metrics['mean_mse']:.6f}, var={metrics['var_mse']:.6f}")
        print(f"   PSNR mean={metrics['mean_psnr']:.3f}, var={metrics['var_psnr']:.3f}")
        print(f"   SSIM mean={metrics['mean_ssim']:.3f}, var={metrics['var_ssim']:.3f}\n")


if __name__ == "__main__":
    test_sec(
        datasets_dir="datasets",
        dataset_name="FORECAST",
        output_file="results/results_all.h5"
    )
