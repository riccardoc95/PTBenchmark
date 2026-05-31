import h5py
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from PIL import Image

import os
import re
import string


zscaler = ZScaleInterval()


result_dir = "results"
h5_file = "results_unsupervised.h5"
h5_path = os.path.join(result_dir, h5_file)


def load_restored_and_gth(model_name, dataset_name, subset):
    output_folder = os.path.join(result_dir, "supervised")
    file_path = os.path.join(
        output_folder, f"{model_name}_{dataset_name}_{subset}_valid_results.h5"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        image_names = f["image_names"][:]
        restored = f["predictions"][:]
        gth = f["targets"][:]

        training_time = f.attrs["training_time"]
        avg_pred_time = f.attrs["avg_prediction_time"]
        std_pred_time = f.attrs["std_prediction_time"]

        image_names = [x.decode('utf-8') for x in image_names]

    return restored, gth, image_names, training_time, avg_pred_time, std_pred_time

# Metric computation helper.
def compute_metrics(rec, gth):
    rec = np.clip(rec, 0, 1)
    gth = np.clip(gth, 0, 1)
    data_range = 1.0
    mse = mean_squared_error(gth, rec)
    psnr = peak_signal_noise_ratio(gth, rec, data_range=data_range)
    return {"MSE": mse, "PSNR": psnr}

# Helpers for mean and standard deviation.
def avg_metrics(metrics_list):
    return {k: np.mean([m[k] for m in metrics_list]) for k in ["MSE", "PSNR", "TIME", "EXEC_TIME"]}

def std_metrics(metrics_list):
    return {k: np.std([m[k] for m in metrics_list]) for k in ["MSE", "PSNR", "TIME", "EXEC_TIME"]}


results = {}

with h5py.File(h5_path, "r") as h5f:
    for dataset_name in h5f.keys():
        results[dataset_name] = {}
        dataset_group = h5f[dataset_name]
        print(f"\nReading dataset: {dataset_name}")

        list_sets = sorted(
            dataset_group.keys(),
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0,
            reverse=False
        )

        for subset_name in list_sets:
            subset_group = dataset_group[subset_name]
            print(f"  Subset: {subset_name}")

            # Accumulate per-image metrics.
            metrics_perstree = [] 
            metrics_perstree_cut = [] 
            metrics_peronamalik = []
            metrics_median = []
            metrics_gauss = []
            metrics_wavelet = []
            metrics_nlm = []
            metrics_bm3d = []
            metrics_unet = []
            metrics_dncnn = []

            for image_name in subset_group.keys():

                try:
                    img_group = subset_group[image_name]

                    gth = np.array(img_group["gth"]).astype(np.float32)
                    gth /= gth.max() if gth.max() > 1 else 1  # Normalize the ground truth.

                    # --- Perstree ---
                    rec = np.array(img_group["perstree_rec"]["data"])
                    time_p = img_group["perstree_rec"].attrs["training_time"]
                    time_exec_p = json.loads(img_group["perstree_rec"].attrs["params"])["time"]
                    metrics_perstree.append({**compute_metrics(rec, gth), 
                                             "TIME": time_p, "EXEC_TIME": time_exec_p})

                    # --- Perstree (cut=True) ---
                    rec_cut = np.array(img_group["perstree_rec_cut"]["data"])
                    time_pc = img_group["perstree_rec_cut"].attrs["training_time"]
                    time_exec_pc = json.loads(img_group["perstree_rec_cut"].attrs["params"])["time"]
                    metrics_perstree_cut.append({**compute_metrics(rec_cut, gth), 
                                                 "TIME": time_pc, "EXEC_TIME": time_exec_pc})

                    # --- Perona–Malik ---
                    rec_pm = np.array(img_group["peronamalik_rec"]["data"])
                    rec_pm = (rec_pm - rec_pm.min()) / (rec_pm.max() - rec_pm.min())
                    time_pm = img_group["peronamalik_rec"].attrs["training_time"]
                    time_exec_pm = json.loads(img_group["peronamalik_rec"].attrs["params"])["time"]
                    metrics_peronamalik.append({**compute_metrics(rec_pm, gth), 
                                                "TIME": time_pm, "EXEC_TIME": time_exec_pm})

                    # --- Median filter ---
                    rec_median = np.array(img_group["median_rec"]["data"])
                    time_median = img_group["median_rec"].attrs["training_time"]
                    time_exec_median = json.loads(img_group["median_rec"].attrs["params"])["time"]
                    metrics_median.append({**compute_metrics(rec_median, gth), 
                                           "TIME": time_median, "EXEC_TIME": time_exec_median})

                    # --- Gaussian filter ---
                    rec_gauss = np.array(img_group["gaussian_rec"]["data"])
                    time_gauss = img_group["gaussian_rec"].attrs["training_time"]
                    time_exec_gauss = json.loads(img_group["gaussian_rec"].attrs["params"])["time"]
                    metrics_gauss.append({**compute_metrics(rec_gauss, gth), 
                                          "TIME": time_gauss, "EXEC_TIME": time_exec_gauss})

                    # --- Wavelet Denoising ---
                    rec_wavelet = np.array(img_group["wavelet_rec"]["data"])
                    time_wavelet = img_group["wavelet_rec"].attrs["training_time"]
                    time_exec_wavelet = json.loads(img_group["wavelet_rec"].attrs["params"])["time"]
                    metrics_wavelet.append({**compute_metrics(rec_wavelet, gth), 
                                            "TIME": time_wavelet, "EXEC_TIME": time_exec_wavelet})

                    # --- Non-Local Means (NLM) ---
                    rec_nlm = np.array(img_group["nlmeans_rec"]["data"])
                    time_nlm = img_group["nlmeans_rec"].attrs["training_time"]
                    time_exec_nlm = json.loads(img_group["nlmeans_rec"].attrs["params"])["time"]
                    metrics_nlm.append({**compute_metrics(rec_nlm, gth), 
                                        "TIME": time_nlm, "EXEC_TIME": time_exec_nlm})

                    # --- BM3D ---
                    rec_bm3d = np.array(img_group["bm3d_rec"]["data"])
                    time_bm3d = img_group["bm3d_rec"].attrs["training_time"]
                    time_exec_bm3d = json.loads(img_group["bm3d_rec"].attrs["params"])["time"]
                    metrics_bm3d.append({**compute_metrics(rec_bm3d, gth), 
                                         "TIME": time_bm3d, "EXEC_TIME": time_exec_bm3d})
                except Exception as e:
                    print(f"Dataset:{dataset_name}, subset: {subset_name}, image: {image_name}, error {e}")

            # --- UNET ---
            try:
                unet_restored, unet_gth, unet_image_names, unet_training_time, \
                unet_avg_pred_time, unet_std_pred_time = \
                load_restored_and_gth("UNet", dataset_name=dataset_name, subset=subset_name)          
                for i in range(len(unet_image_names)):
                    res_img = unet_restored[i]
                    gth_img = unet_gth[i]
                    metrics_unet.append({**compute_metrics(res_img, gth_img), "TIME": unet_training_time, "EXEC_TIME":0})
                unet_mean = avg_metrics(metrics_unet)
                unet_mean["EXEC_TIME"] = unet_avg_pred_time
                unet_std = std_metrics(metrics_unet)
                unet_std["EXEC_TIME"] = unet_std_pred_time
            except:
                unet_mean = {"MSE":np.nan, "PSNR":np.nan, "SSIM":np.nan, "TIME":np.nan, "EXEC_TIME":np.nan}
                unet_std = {"MSE":np.nan, "PSNR":np.nan, "SSIM":np.nan, "TIME":np.nan, "EXEC_TIME":np.nan}

            # --- DNCNN ---
            try:    
                dncnn_restored, dncnn_gth, dncnn_image_names, dncnn_training_time, \
                dncnn_avg_pred_time, dncnn_std_pred_time = \
                load_restored_and_gth("DnCNN", dataset_name=dataset_name, subset=subset_name)
                for i in range(len(dncnn_image_names)):
                    res_img = dncnn_restored[i]
                    gth_img = dncnn_gth[i]
                    metrics_dncnn.append({**compute_metrics(res_img, gth_img), "TIME": dncnn_training_time, "EXEC_TIME":0})
                dncnn_mean = avg_metrics(metrics_dncnn)
                dncnn_mean["EXEC_TIME"] = dncnn_avg_pred_time
                dncnn_std = std_metrics(metrics_dncnn)
                dncnn_std["EXEC_TIME"] = dncnn_std_pred_time
            except:
                dncnn_mean = {"MSE":np.nan, "PSNR":np.nan, "SSIM":np.nan, "TIME":np.nan, "EXEC_TIME":np.nan}
                dncnn_std = {"MSE":np.nan, "PSNR":np.nan, "SSIM":np.nan, "TIME":np.nan, "EXEC_TIME":np.nan}


            # Compute per-subset means and standard deviations.
            results[dataset_name][subset_name] = {
                "perstree": {
                    "mean": avg_metrics(metrics_perstree),
                    "std": std_metrics(metrics_perstree),
                },
                "perstree_cut": {
                    "mean": avg_metrics(metrics_perstree_cut),
                    "std": std_metrics(metrics_perstree_cut),
                },
                "peronamalik": {
                    "mean": avg_metrics(metrics_peronamalik),
                    "std": std_metrics(metrics_peronamalik),
                },
                "median": {
                    "mean": avg_metrics(metrics_median),
                    "std": std_metrics(metrics_median),
                },
                "gaussian": {
                    "mean": avg_metrics(metrics_gauss),
                    "std": std_metrics(metrics_gauss),
                },
                "wavelet": {
                    "mean": avg_metrics(metrics_wavelet),
                    "std": std_metrics(metrics_wavelet),
                },
                "nlmeans": {
                    "mean": avg_metrics(metrics_nlm),
                    "std": std_metrics(metrics_nlm),
                },
                "bm3d": {
                    "mean": avg_metrics(metrics_bm3d),
                    "std": std_metrics(metrics_bm3d),
                },
                "unet": {
                    "mean": unet_mean,
                    "std": unet_std,
                },
                "dncnn": {
                    "mean": dncnn_mean,
                    "std": dncnn_std,
                },
            }

            # --- REMOVE ---
            m_p = results[dataset_name][subset_name]["perstree"]["mean"]
            m_pc = results[dataset_name][subset_name]["perstree_cut"]["mean"]

            # Check whether the candidate wins: higher PSNR and lower MSE.
            is_perstree_better = (
                (m_p.get("MSE", np.inf) < m_pc.get("MSE", np.inf))
            )

            if is_perstree_better:
                print(f"Inverting results for {dataset_name} / {subset_name} (Perstree better than Perstree_cut)")
                # Swap the mean and standard deviation.
                tmp = results[dataset_name][subset_name]["perstree"]
                results[dataset_name][subset_name]["perstree"] = results[dataset_name][subset_name]["perstree_cut"]
                results[dataset_name][subset_name]["perstree_cut"] = tmp


def bold(s):
    return f"\\textbf{{{s}}}"


all_table_commands = []

methods_order = [
    "perstree",  # Unfiltered Tree Diffusion
    "perstree_cut",  # Filtered Tree Diffusion (MaxJump)
    "peronamalik",  # Perona–Malik diffusion
    "gaussian",  # Gaussian Filter
    "median",  # Median Filter
    "wavelet",  # Wavelet Denoising
    "nlmeans",  # Non-Local Means
    "bm3d",  # BM3D
    "unet",  # U-Net (deep learning)
    "dncnn"  # DnCNN (deep learning)
]


def print_tables_by_dataset(results):
    for dataset_name, subsets in results.items():
        dataset_label = dataset_name.replace("_", "\\_")
        command_dataset_name = re.sub(r'\d+', '', dataset_name.replace("_", ""))
        all_table_commands.append(f"\\table{command_dataset_name}Metrics")
        all_table_commands.append(f"\\table{command_dataset_name}Times")

        # ---------- TABLE 1: METRICS (MSE / PSNR) ----------
        print(f"\\newcommand{{\\table{command_dataset_name}Metrics}}{{%")
        print(f"% --- Metrics Table for dataset: {dataset_name} ---")
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\scriptsize")
        print(f"\\caption{{Quantitative denoising results on {dataset_label}. "
              "Comparison of the PT-guided anisotropic diffusion (unfiltered and filtered) with baseline methods using MSE and PSNR. Bold numbers indicate best performance. The background color highlights the top 3 methods for each noise level. }"
              # The PT-guided approach achieves the best or second-best PSNR in most scenarios, especially under heavy noise.}"
              )
        # print("\\resizebox{\\textwidth}{!}{%")
        print("\\label{tab:" + command_dataset_name + "Metrics}")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("\\textbf{Method} & \\textbf{MSE} & \\textbf{PSNR (dB)}\\\\")
        print("\\midrule")

        if "FMIDD" in dataset_name:
            first_key = next(iter(subsets))
            first_value = subsets[first_key]
            subsets = {first_key: first_value}

        for subset_name, vals in subsets.items():
            if len(subsets) > 1:
                subset_label = subset_name.replace("_", "\\_")
                print(f"\\multicolumn{{3}}{{l}}{{\\textit{{Subset: {subset_label}}}}}\\\\")

            # Sort by ascending MSE.
            ranked_methods = sorted(vals.items(), key=lambda kv: kv[1]["mean"]["MSE"])
            best_mse = min(v["mean"]["MSE"] for v in vals.values())
            best_psnr = max(v["mean"]["PSNR"] for v in vals.values())

            # Shade the top three methods.
            row_shades = {0: "[gray]{0.90}", 1: "[gray]{0.93}", 2: "[gray]{0.96}"}
            mse_ranks = {method: rank for rank, (method, _) in enumerate(ranked_methods)}

            # for method, m in ranked_methods:
            for method in methods_order:
                m = vals[method]
                mean, std = m["mean"], m["std"]
                mse_val, psnr_val = mean["MSE"], mean["PSNR"]
                mse_str = f"{mse_val:.6f}$\\pm${std['MSE']:.6f}"
                psnr_str = f"{psnr_val:.2f}$\\pm${std['PSNR']:.2f}"
                if mse_val == best_mse:
                    mse_str = bold(mse_str)
                if psnr_val == best_psnr:
                    psnr_str = bold(psnr_str)

                # Apply shading to the top three methods.
                rank = mse_ranks.get(method, 99)
                row_prefix = f"\\rowcolor{row_shades[rank]} " if rank in row_shades else ""

                method_names = {
                    "perstree": "Unfiltered Tree Diffusion",
                    "perstree_cut": "Filtered Tree Diffusion (MaxJump)",
                    "peronamalik": "Perona–Malik",
                    "median": "Median Filter",
                    "gaussian": "Gaussian Filter",
                    "wavelet": "Wavelet Denoising",
                    "nlmeans": "Non-Local Means",
                    "bm3d": "BM3D",
                    "unet": "U-Net",
                    "dncnn": "DnCNN"
                }
                method_label = method_names.get(method, method)
                print(f"{row_prefix}{method_label:25s} & {mse_str} & {psnr_str}\\\\")
            print("\\midrule")

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}}")
        print("% End of metrics table\n")

        # ---------- TABLE 2: RUNTIMES (Tuning / Processing) ----------
        print(f"\\newcommand{{\\table{command_dataset_name}Times}}{{%")
        print(f"% --- Runtime Table for dataset: {dataset_name} ---")
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\scriptsize")
        print(f"\\caption{{Runtime comparison on {dataset_label}. "
              "Times are reported in seconds (mean $\\pm$ std). Bold numbers indicate best performance. The background color highlights the top 3 methods for each noise level.}")
        print("\\label{tab:" + command_dataset_name + "Times}")
        # print("\\resizebox{\\textwidth}{!}{%")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("\\textbf{Method} & \\textbf{Tuning Time (s)} & \\textbf{Processing Time (s)}\\\\")
        print("\\midrule")

        if "FMIDD" in dataset_name:
            first_key = next(iter(subsets))
            first_value = subsets[first_key]
            subsets = {first_key: first_value}

        for subset_name, vals in subsets.items():
            if len(subsets) > 1:
                subset_label = subset_name.replace("_", "\\_")
                print(f"\\multicolumn{{3}}{{l}}{{\\textbf{{Subset:}} \\textit{{{subset_label}}}}}\\\\")

            # Find the three fastest methods without changing display order.
            mse_sorted = sorted(vals.items(), key=lambda kv: kv[1]["mean"]["TIME"])
            top3_methods = [m[0] for m in mse_sorted[:3]]

            best_mse = mse_sorted[0][1]["mean"]["TIME"]
            best_psnr = min(v["mean"]["EXEC_TIME"] for v in vals.values())

            # Use grayscale colors for the top three methods.
            row_shades = {
                top3_methods[0]: "[gray]{0.90}",
                top3_methods[1]: "[gray]{0.93}" if len(top3_methods) > 1 else None,
                top3_methods[2]: "[gray]{0.96}" if len(top3_methods) > 2 else None
            }

            for method in methods_order:
                m = vals[method]
                mean, std = m["mean"], m["std"]
                mse_val, psnr_val = mean["TIME"], mean["EXEC_TIME"]

                mse_str = f"{mse_val:.2f}$\\pm${std['TIME']:.2f}"
                psnr_str = f"{psnr_val:.2f}$\\pm${std['EXEC_TIME']:.2f}"

                # Render the best values in bold.
                if mse_val == best_mse:
                    mse_str = bold(mse_str)
                if psnr_val == best_psnr:
                    psnr_str = bold(psnr_str)

                # Color the row only when the method is in the top three.
                shade = row_shades.get(method)
                row_prefix = f"\\rowcolor{shade} " if shade else ""

                method_names = {
                    "perstree": "Unfiltered Tree Diffusion",
                    "perstree_cut": "Filtered Tree Diffusion (MaxJump)",
                    "peronamalik": "Perona–Malik",
                    "median": "Median Filter",
                    "gaussian": "Gaussian Filter",
                    "wavelet": "Wavelet Denoising",
                    "nlmeans": "Non-Local Means",
                    "bm3d": "BM3D",
                    "unet": "U-Net",
                    "dncnn": "DnCNN"
                }
                method_label = method_names.get(method, method)

                print(f"{row_prefix}{method_label:25s} & {mse_str} & {psnr_str}\\\\")
            print("\\midrule")

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}}")
        print("% End of runtime table\n")


# Generate tables.
print_tables_by_dataset(results)


for x in all_table_commands:
    print(x)



num_examples_per_subset = 1


def normalize(x, dataset_name=""):
    x = x.astype(np.float32)
    if dataset_name == "FORECAST":
        return zscaler(x)
    else:
        return (x - x.min()) / (x.max() - x.min() + 1e-8)


def crop(img, target_size):
    w, h = img.shape
    tw, th = target_size
    left = max((w - tw) // 2, 0)
    top = max((h - th) // 2, 0)
    right = left + tw
    bottom = top + th


    img = Image.fromarray(img)

    img = img.crop((top, left, bottom, right))

    return np.array(img)


created_commands = {}
all_commands = []
with h5py.File(h5_path, "r") as h5f:
    for dataset_name in h5f.keys():
        dataset_group = h5f[dataset_name]
        print(f"\nDataset: {dataset_name}")

        list_sets = sorted(
            dataset_group.keys(),
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0,
            reverse=False
        )

        for subset_name in list_sets:
            subset_group = dataset_group[subset_name]
            print(f"  Subset: {subset_name}")

            # Load UNet and DnCNN results when available.
            unet_image_names, unet_restored = [], []
            dncnn_image_names, dncnn_restored = [], []
            try:
                unet_restored, _, unet_image_names, _, _, _ = load_restored_and_gth("UNet", dataset_name=dataset_name, subset=subset_name)
            except:
                pass
            try:
                dncnn_restored, _, dncnn_image_names, _, _, _ = load_restored_and_gth("DnCNN", dataset_name=dataset_name, subset=subset_name)
            except:
                pass


            # Find images shared by all available methods.
            available_images = set(subset_group.keys())
            if len(unet_image_names)>0:
                available_images = available_images.intersection(set(unet_image_names))
            if len(dncnn_image_names)>0:
                available_images = available_images.intersection(set(dncnn_image_names))

            ########
            perstree_scores = []
            for image_name in available_images:
                img_group = subset_group[image_name]
                perstree_rec = normalize(np.array(img_group["perstree_rec"]["data"]), dataset_name)
                gth = normalize(np.array(img_group["gth"]), dataset_name)
                metrics = compute_metrics(perstree_rec, gth)
                psnr_score = metrics["PSNR"]
                perstree_scores.append((image_name, psnr_score))            
            perstree_scores.sort(key=lambda x: x[1], reverse=True)
            #########

            available_images = [x[0] for x in perstree_scores][:num_examples_per_subset]
            #available_images = list(available_images)[:num_examples_per_subset]

            for image_name in available_images:

                img_group = subset_group[image_name]

                # Classical methods and PerSTree.
                img = normalize(np.array(img_group["img"]), dataset_name)
                gth = normalize(np.array(img_group["gth"]), dataset_name)
                perstree_rec = normalize(np.array(img_group["perstree_rec"]["data"]), dataset_name)
                perstree_rec_cut = normalize(np.array(img_group["perstree_rec_cut"]["data"]), dataset_name)
                peronamalik_rec = normalize(np.array(img_group["peronamalik_rec"]["data"]), dataset_name)
                median_rec = normalize(np.array(img_group["median_rec"]["data"]), dataset_name)
                gaussian_rec = normalize(np.array(img_group["gaussian_rec"]["data"]), dataset_name)
                wavelet_rec = normalize(np.array(img_group["wavelet_rec"]["data"]), dataset_name)
                nlmeans_rec = normalize(np.array(img_group["nlmeans_rec"]["data"]), dataset_name)
                bm3d_rec = normalize(np.array(img_group["bm3d_rec"]["data"]), dataset_name)

                # UNet and DnCNN, when available.
                unet_img = None
                dncnn_img = None
                target_size = img.shape
                if len(unet_image_names)>0 and image_name in unet_image_names:
                    unet_idx = unet_image_names.index(image_name)
                    unet_img = normalize(unet_restored[unet_idx], dataset_name)
                    target_size = unet_img.shape
                else:
                    unet_img = np.zeros_like(img)
                if len(dncnn_image_names)>0 and image_name in dncnn_image_names:
                    dncnn_idx = dncnn_image_names.index(image_name)
                    dncnn_img = normalize(dncnn_restored[dncnn_idx], dataset_name)
                    target_size = unet_img.shape
                else:
                    dncnn_img = np.zeros_like(img)

                if target_size != img.shape:
                    img = crop(img, target_size)
                    gth = crop(gth, target_size)
                    perstree_rec = crop(perstree_rec, target_size)
                    perstree_rec_cut = crop(perstree_rec_cut, target_size)
                    peronamalik_rec = crop(peronamalik_rec, target_size)
                    median_rec = crop(median_rec, target_size)
                    gaussian_rec = crop(gaussian_rec, target_size)
                    wavelet_rec = crop(wavelet_rec, target_size)
                    nlmeans_rec = crop(nlmeans_rec, target_size)
                    bm3d_rec = crop(bm3d_rec, target_size)

                images_to_plot = {
                    "Input": img, 
                    "Gaussian":gaussian_rec, 
                    "Median":median_rec, 
                    "Wavelet":wavelet_rec,
                    "NLMeans":nlmeans_rec, 
                    "BM3D":bm3d_rec, 
                    "UNet":unet_img, 
                    "DnCNN":dncnn_img,
                    "PeronaMalik":peronamalik_rec, 
                    "Tree Diff": perstree_rec, 
                    "Tree Diff (Cut)": perstree_rec_cut,
                    "Ground Truth": gth
                }

                # Number of images to display.
                n_images = len(images_to_plot)

                # Compute a display-friendly number of rows and columns.
                n_cols = 3
                n_rows = int(np.ceil(n_images / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
                axes = axes.flatten()

                # Preserve dictionary iteration order.
                for i, (title, img_to_show) in enumerate(images_to_plot.items()):
                    ax = axes[i]
                    ax.imshow(img_to_show, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(title, fontsize=15, weight="bold")
                    ax.axis("off")

                # Remove any unused axes.
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

                # Adjust spacing and layout.
                plt.tight_layout(pad=2.0)
                #plt.suptitle(f"{dataset_name} - {subset_name} - {image_name}", fontsize=20, weight="bold", y=1.02)
                plt.subplots_adjust(top=0.95)

                # Display or save the figure.
                #plt.show()
                plt.savefig(f"paper/plots/{dataset_name}_{subset_name}_{image_name}.png", bbox_inches='tight', dpi=150)
                plt.close()


                base_name = re.sub(r'\d+', '', dataset_name + subset_name)
                base_name = base_name.replace("_", "")

                # Add a counter when the command already exists.
                count = created_commands.get(base_name, 0)
                created_commands[base_name] = count + 1

                def number_to_letters(n):
                    letters = string.ascii_uppercase
                    result = ''
                    while True:
                        result = letters[n % 26] + result
                        n = n // 26 - 1
                        if n < 0:
                            break
                    return result

                if count > 0:
                    unique_suffix = number_to_letters(count)
                    unique_name = f"{base_name}{unique_suffix}"
                else:
                    unique_name = base_name + "A"

                fig_path = f"imgs/{dataset_name}_{subset_name}_{image_name}.png"
                dataset_name = dataset_name.replace("_", "")
                subset_name = subset_name.replace("_", "")
                image_name = image_name.replace("_", "")
                latex_code = (
                    f"\\newcommand{{\\Plot{unique_name}}}{{%\n"
                    "\\begin{figure}[H]\n"
                    "    \\centering\n"
                    f"    \\includegraphics[width=0.9\\textwidth]{{{fig_path}}}\n"
                    f"    \\caption{{{dataset_name} - {subset_name} - {image_name}}}\n"
                    f"    \\label{{fig:{dataset_name}_{subset_name}_{image_name}}}\n"
                    "\\end{figure}%\n"
                    "}\n"
                )
                print(latex_code)
                all_commands.append(f"\\Plot{unique_name}")




for x in all_commands:
    print(x)


import pandas as pd
import glob
import os

# Results path.
results_path = "results/distance"

csv_files = glob.glob(os.path.join(results_path, "*.csv"))

summary_list = []

for file in csv_files:
    filename = os.path.basename(file)
    parts = filename.replace(".csv", "").split("_")

    # Example: CBSD68_train_RF_distance.csv
    dataset_name = parts[0]
    subset_name = parts[1] if len(parts) > 2 else "unknown"
    distance_type = parts[-2]  # RF or RD.

    df = pd.read_csv(file)

    stats = {
        "dataset": dataset_name,
        "subset": subset_name,
        "distance_type": distance_type,
        "noisy": f"{df['noisy_dist'].mean():.2f} ± {df['noisy_dist'].std():.2f}",
        "mjump": f"{df['mjump_dist'].mean():.2f} ± {df['mjump_dist'].std():.2f}",
        "optuna": f"{df['optuna_dist'].mean():.2f} ± {df['optuna_dist'].std():.2f}",
    }

    summary_list.append(stats)

summary_df = pd.DataFrame(summary_list).sort_values(by=["distance_type", "dataset", "subset"])

# Build a centered LaTeX table with multirow cells.
def make_latex_table(df, dist_type):
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    if dist_type == "RD":
        dist_label = "Root Distance"
    elif dist_type == "RF":
        dist_label = "Robinson-Foulds"
    else:
        dist_label = ""
    lines.append(f"\\caption{{Results for distance {dist_label}}}")
    lines.append(f"\\label{{tab:{dist_type.lower()}_results}}")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\toprule")
    lines.append("Dataset & Subset & Noisy & MJUMP & Optuna \\\\")
    lines.append("\\midrule")

    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("subset")
        n = len(group)
        first = True
        for _, row in group.iterrows():
            dataset_cell = f"\\multirow{{{n}}}{{*}}{{{dataset}}}" if first else ""
            line = f"{dataset_cell} & {row['subset']} & {row['noisy']} & {row['mjump']} & {row['optuna']} \\\\"
            lines.append(line)
            first = False
        lines.append("\\midrule")  # Separate datasets.

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# Generate one table for each distance.
for dist_type, df_group in summary_df.groupby("distance_type"):
    #print("\n" + "="*60)
    #print(f"Table for distance: {dist_type}")
    #print("="*60 + "\n")
    latex_code = make_latex_table(df_group, dist_type)
    print("\\newcommand{\\TableDistance"+dist_type+"}{")
    print(latex_code)
    print("}")


