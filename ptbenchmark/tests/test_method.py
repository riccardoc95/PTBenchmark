import os
import time
import json
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from ptbenchmark.src import (
    Dataset, CropDataset,
    DnCNN, UNet, train_one_epoch, validate,
    pt_denoise, pm_denoise, median_denoise, gaussian_denoise,
    wavelet_denoise, nl_means_denoise, bm3d_denoise
)

supervised_methods = {
    "unet": UNet(n_channels=1, n_classes=1),
    "dncnn":DnCNN(channels=1),
}

unsupervised_methods = {
            "perstree": partial(pt_denoise, cut=False),
            "perstree_cut": partial(pt_denoise, cut=True),
            "peronamalik": pm_denoise,
            "median": median_denoise,
            "gaussian": gaussian_denoise,
            "wavelet": wavelet_denoise,
            "nlmeans": nl_means_denoise,
            "bm3d": bm3d_denoise,
        }


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


def test_method(method_name, dataset_name, datasets_dir, output_file, device="cpu", n_epochs=20, batch_size=8):
    dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
    device = torch.device(device)

    if method_name.lower() in ["unet", "dncnn"]:
        for subset in dataset.get_subsets():
            dataset.set_subset(subset)

            min_w, min_h = float("inf"), float("inf")
            for _, img, _ in dataset:
                w, h = img.shape
                min_w, min_h = min(min_w, w), min(min_h, h)
            target_size = (min_w, min_h)

            n_total = len(dataset)
            n_train = int(n_total * 0.8)
            n_valid = n_total - n_train
            train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid])

            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = CropDataset(train_dataset, target_size, transform)
            valid_dataset = CropDataset(valid_dataset, target_size, transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            # Modello
            model = supervised_methods[method_name.lower()]
            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # Training
            start_train = time.time()
            for epoch in range(n_epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_psnr, val_ssim = validate(model, valid_loader, criterion, device)
                print(f"[{method_name}] Epoch {epoch+1}/{n_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | PSNR {val_psnr:.2f}")
            training_time = time.time() - start_train

            all_mse, all_psnr, all_ssim = [], [], []
            all_time, all_training_time = [], []

            model.eval()
            with torch.no_grad():
                for image_names, inputs, targets in tqdm(valid_loader, desc=f"{dataset_name}/{subset}"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    t0 = time.time()
                    outputs = model(inputs)
                    elapsed = time.time() - t0

                    outputs_np = outputs.squeeze(1).cpu().numpy()
                    targets_np = targets.squeeze(1).cpu().numpy()

                    for i, img_name in enumerate(image_names):
                        rec = outputs_np[i]
                        gth = targets_np[i]
                        mse = float(((rec - gth) ** 2).mean())
                        psnr_val = float(psnr(gth, rec, data_range=gth.max() - gth.min()))
                        ssim_val = float(ssim(gth, rec, data_range=gth.max() - gth.min()))
                        all_mse.append(mse)
                        all_psnr.append(psnr_val)
                        all_ssim.append(ssim_val)
                        all_time.append(elapsed)
                        all_training_time.append(training_time)

                        save_to_h5(
                            output_file,
                            dataset_name,
                            subset,
                            img_name,
                            method_name,
                            {
                                "rec": rec,
                                "mse": mse,
                                "psnr": psnr_val,
                                "ssim": ssim_val,
                                "params": {"epochs": n_epochs},
                                "training_time": training_time,
                                "time": elapsed,
                            },
                        )

            # Metriche globali
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

            print(f"\nStatistiche globali per subset '{subset}' ({method_name}):")
            print(f"   MSE  mean={metrics['mean_mse']:.6f}, var={metrics['var_mse']:.6f}")
            print(f"   PSNR mean={metrics['mean_psnr']:.3f}, var={metrics['var_psnr']:.3f}")
            print(f"   SSIM mean={metrics['mean_ssim']:.3f}, var={metrics['var_ssim']:.3f}\n")
            print(f"   TIME mean={metrics['mean_time']:.4f}s, var={metrics['var_time']:.4f}")
            print(f"   TRAIN mean={metrics['mean_training_time']:.4f}s, var={metrics['var_training_time']:.4f}\n")

    else:
        if method_name.lower() not in unsupervised_methods:
            raise ValueError(f"Metodo non riconosciuto: {method_name}")

        denoise_func = unsupervised_methods[method_name.lower()]

        for subset in dataset.get_subsets():
            dataset.set_subset(subset)

            all_mse, all_psnr, all_ssim = [], [], []
            all_time, all_training_time = [], []

            for image_name, img, gth in tqdm(dataset, desc=f"{dataset_name}/{subset}"):
                t0 = time.time()
                rec, mse, params = denoise_func(img, gth)
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


            # Statistiche globali
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

            print(f"\nStatistiche globali per subset '{subset}' ({method_name}):")
            print(f"   MSE  mean={metrics['mean_mse']:.6f}, var={metrics['var_mse']:.6f}")
            print(f"   PSNR mean={metrics['mean_psnr']:.3f}, var={metrics['var_psnr']:.3f}")
            print(f"   SSIM mean={metrics['mean_ssim']:.3f}, var={metrics['var_ssim']:.3f}\n")
            print(f"   TIME mean={metrics['mean_time']:.4f}s, var={metrics['var_time']:.4f}")
            print(f"   TRAIN mean={metrics['mean_training_time']:.4f}s, var={metrics['var_training_time']:.4f}\n")




if __name__ == "__main__":
    datasets_dir = "datasets"
    output_file = os.path.join("results", "results_all.h5")

    # Esempio:
    test_method("perstree", "FORECAST", datasets_dir, output_file)
    # test_method("UNet", "CBSD68", datasets_dir, output_file, device="cpu", n_epochs=5)
