import os
import re
import h5py
import json
import string
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from astropy.visualization import ZScaleInterval

# === Percorsi ===
H5_PATH = "sbatch/merged.h5"
OUTPUT_JSON = "sbatch/results_summary.json"
OUTPUT_TEX_TABLES = "paper/tables.tex"
OUTPUT_TEX_PLOTS = "paper/figures_plots.tex"
OUTPUT_TEX_DISTANCES = "paper/tables_distances.tex"
OUTPUT_TXT_ALL = "paper/table_and_plots.txt"
OUTPUT_FIG_DIR = "paper/plots"
os.makedirs("paper", exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# === Parametri globali ===
methods = [
    "perstree", "perstree_cut", "perstree_rec_sec", "peronamalik", "median",
    "gaussian", "wavelet", "nlmeans", "bm3d", "unet", "dncnn"
]
zscaler = ZScaleInterval()


methods_order = [
                "perstree",         # Unfiltered Tree Diffusion
                "perstree_cut",     # Filtered Tree Diffusion (MaxJump)
                "peronamalik",      # Perona–Malik diffusion
                "gaussian",         # Gaussian Filter
                "median",           # Median Filter
                "wavelet",          # Wavelet Denoising
                "nlmeans",          # Non-Local Means
                "bm3d",             # BM3D
                "unet",             # U-Net (deep learning)
                "dncnn"             # DnCNN (deep learning)
            ]

def bold(s): return f"\\textbf{{{s}}}"

def normalize(img, dataset_name=""):
    img = img.astype(np.float32)
    if dataset_name == "FORECAST":
        return zscaler(img)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def number_to_letters(n):
    letters = string.ascii_uppercase
    result = ''
    while True:
        result = letters[n % 26] + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def build_results_with_distances(h5_path, output_json):
    results = {}

    with h5py.File(h5_path, "r") as f:
        for dataset_name in f.keys():
            results[dataset_name] = {}
            dataset_group = f[dataset_name]

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

                # --- DISTANCES (RD / RF) ---
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

    # --- Salva su JSON ---
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f_json:
        json.dump(results, f_json, indent=4)

    print(f"Results (inclusi RD/RF) salvati in {output_json}")
    return results


def generate_tables_tex(results, output_path):
    """
    Scrive in output_path i comandi LaTeX per:
      - Tabella Metrics (MSE/PSNR) con shading top-3 su MSE (↓ meglio)
      - Tabella Times (Training/Exec) con shading top-3 su TRAINING_TIME (↓ meglio)
    Ritorna la lista dei comandi creati.
    """
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
        "dncnn": "DnCNN",
    }

    shades_by_rank = {0: "[gray]{0.90}", 1: "[gray]{0.93}", 2: "[gray]{0.96}"}
    all_commands = []

    with open(output_path, "w") as f_tex:
        for dataset_name, subsets in results.items():
            dataset_label = dataset_name.replace("_", "\\_")
            command_name = re.sub(r'\d+', '', dataset_name.replace("_", ""))

            # -------------------- METRICS TABLE --------------------
            lines = []
            lines.append(f"\\newcommand{{\\table{command_name}Metrics}}{{%")
            lines.append("\\begin{table}[H]")
            lines.append("\\centering")
            lines.append("\\scriptsize")
            lines.append(f"\\caption{{Quantitative denoising results on {dataset_label}. Bold marks best per subset.}}")
            lines.append(f"\\label{{tab:{command_name}Metrics}}")
            lines.append("\\begin{tabular}{lcc}")
            lines.append("\\toprule")
            lines.append("\\textbf{Method} & \\textbf{MSE} & \\textbf{PSNR (dB)}\\\\")
            lines.append("\\midrule")

            for subset_name, vals in subsets.items():
                # considera solo i metodi presenti in questo subset
                present_methods = [m for m in methods_order if m in vals]
                if not present_methods:
                    continue

                subset_label = subset_name.replace("_", "\\_")
                lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{Subset: {subset_label}}}}}\\\\")
                # calcolo best per bold
                best_mse = min(vals[m]["mean"]["MSE"] for m in present_methods)
                best_psnr = max(vals[m]["mean"]["PSNR"] for m in present_methods)
                # ranking per shading (MSE crescente)
                ranked = sorted(present_methods, key=lambda m: vals[m]["mean"]["MSE"])
                rank_map = {m: r for r, m in enumerate(ranked)}

                for m in methods_order:
                    if m not in vals:
                        continue
                    mean, std = vals[m]["mean"], vals[m]["std"]
                    mse_str = f"{mean['MSE']:.6f}$\\pm${std['MSE']:.6f}"
                    psnr_str = f"{mean['PSNR']:.2f}$\\pm${std['PSNR']:.2f}"
                    if mean["MSE"] == best_mse:
                        mse_str = bold(mse_str)
                    if mean["PSNR"] == best_psnr:
                        psnr_str = bold(psnr_str)

                    shade = shades_by_rank.get(rank_map.get(m, 99))
                    rowprefix = f"\\rowcolor{shade} " if shade else ""
                    label = method_names.get(m, m)
                    lines.append(f"{rowprefix}{label:25s} & {mse_str} & {psnr_str}\\\\")

                lines.append("\\midrule")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}}")
            lines.append("% End of metrics table\n")
            f_tex.write("\n".join(lines) + "\n")
            all_commands.append(f"\\table{command_name}Metrics")

            # -------------------- TIMES TABLE --------------------
            lines = []
            lines.append(f"\\newcommand{{\\table{command_name}Times}}{{%")
            lines.append("\\begin{table}[H]")
            lines.append("\\centering")
            lines.append("\\scriptsize")
            lines.append(f"\\caption{{Runtime comparison on {dataset_label}. Mean $\\pm$ std in seconds.}}")
            lines.append(f"\\label{{tab:{command_name}Times}}")
            lines.append("\\begin{tabular}{lcc}")
            lines.append("\\toprule")
            lines.append("\\textbf{Method} & \\textbf{Training (s)} & \\textbf{Exec (s)}\\\\")
            lines.append("\\midrule")

            for subset_name, vals in subsets.items():
                present_methods = [m for m in methods_order if m in vals]
                if not present_methods:
                    continue

                subset_label = subset_name.replace("_", "\\_")
                lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{Subset: {subset_label}}}}}\\\\")

                # ranking per shading: TRAINING_TIME crescente (veloce = meglio)
                # (se manca TRAINING_TIME, usa +inf per mandarlo in fondo)
                ranked = sorted(
                    present_methods,
                    key=lambda m: (vals[m]["mean"].get("TRAINING_TIME", float("inf")))
                )
                top3 = ranked[:3]
                shade_map = {
                    top3[0]: "[gray]{0.90}" if len(top3) > 0 else None,
                    top3[1]: "[gray]{0.93}" if len(top3) > 1 else None,
                    top3[2]: "[gray]{0.96}" if len(top3) > 2 else None,
                }

                # opzionale: bold sui migliori (minimi)
                best_train = min(vals[m]["mean"].get("TRAINING_TIME", float("inf")) for m in present_methods)
                best_exec = min(vals[m]["mean"].get("TIME", float("inf")) for m in present_methods)

                for m in methods_order:
                    if m not in vals:
                        continue
                    mean, std = vals[m]["mean"], vals[m]["std"]
                    tr = mean.get("TRAINING_TIME", np.nan)
                    tr_std = std.get("TRAINING_TIME", np.nan)
                    ex = mean.get("TIME", np.nan)
                    ex_std = std.get("TIME", np.nan)

                    tr_str = f"{tr:.2f}$\\pm${tr_std:.2f}" if np.isfinite(tr) else "--"
                    ex_str = f"{ex:.2f}$\\pm${ex_std:.2f}" if np.isfinite(ex) else "--"

                    if tr == best_train:
                        tr_str = bold(tr_str)
                    if ex == best_exec:
                        ex_str = bold(ex_str)

                    shade = shade_map.get(m)
                    rowprefix = f"\\rowcolor{shade} " if shade else ""
                    label = method_names.get(m, m)
                    lines.append(f"{rowprefix}{label:25s} & {tr_str} & {ex_str}\\\\")

                lines.append("\\midrule")

            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}}")
            lines.append("% End of runtime table\n")
            f_tex.write("\n".join(lines) + "\n")
            all_commands.append(f"\\table{command_name}Times")

    return all_commands


def generate_distance_tables_tex(results, output_path):
    with open(output_path, "w") as f_tex:
        all_commands = []

        for dataset_name, subsets in results.items():
            command_name = re.sub(r'\d+', '', dataset_name.replace("_", ""))
            dataset_label = dataset_name.replace("_", "\\_")

            for dist in ["RD", "RF"]:
                found = any(f"distance_{dist}" in vals for vals in subsets.values())
                if not found:
                    continue

                tex_code = []
                tex_code.append(f"\\newcommand{{\\table{command_name}{dist}Distances}}{{%")
                tex_code.append("\\begin{table}[H]\\centering\\scriptsize")
                tex_code.append(f"\\caption{{Persistence-based distances ({dist}) for {dataset_label}.}}")
                tex_code.append("\\begin{tabular}{lccc}\\toprule")
                tex_code.append("\\textbf{Subset} & \\textbf{Noisy} & \\textbf{MJump} & \\textbf{Optuna}\\\\\\midrule")

                for subset_name, vals in subsets.items():
                    key = f"distance_{dist}"
                    if key not in vals:
                        continue
                    mean, std = vals[key]["mean"], vals[key]["std"]
                    noisy = f"{mean['NOISY']:.4f}$\\pm${std['NOISY']:.4f}"
                    mjump = f"{mean['MJUMP']:.4f}$\\pm${std['MJUMP']:.4f}"
                    optuna = f"{mean['OPTUNA']:.4f}$\\pm${std['OPTUNA']:.4f}"
                    tex_code.append(f"{subset_name:20s} & {noisy} & {mjump} & {optuna}\\\\")
                    tex_code.append("\\midrule")

                tex_code.append("\\bottomrule\\end{tabular}\\end{table}}")
                f_tex.write("\n".join(tex_code) + "\n\n")
                all_commands.append(f"\\table{command_name}{dist}Distances")

        return all_commands


def generate_plots_tex(h5_path, output_tex, output_fig_dir, num_examples_per_subset=1, metric_target="psnr"):
    created_commands = {}
    all_commands = []
    zscaler = ZScaleInterval()

    def normalize(img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    with h5py.File(h5_path, "r") as f, open(output_tex, "w") as f_tex:
        for dataset_name in f.keys():
            for subset_name in f[dataset_name].keys():
                subset_group = f[dataset_name][subset_name]
                if not isinstance(subset_group, h5py.Group):
                    continue

                perstree_scores = []
                for image_name, img_group in subset_group.items():
                    if not isinstance(img_group, h5py.Group) or "perstree" not in img_group:
                        continue
                    attrs = img_group["perstree"].attrs
                    mse = attrs.get("mse", np.nan)
                    psnr = attrs.get("psnr", np.nan)
                    perstree_scores.append((image_name, mse, psnr))
                if not perstree_scores:
                    continue

                key_idx = 2 if metric_target == "psnr" else 1
                perstree_scores.sort(key=lambda x: x[key_idx], reverse=(metric_target == "psnr"))
                best_images = [x[0] for x in perstree_scores[:num_examples_per_subset]]

                for image_name in best_images:
                    img_group = subset_group[image_name]
                    #gth = normalize(np.array(img_group["gth"]))
                    rec = normalize(np.array(img_group["perstree"]["rec"]))
                    gth = np.zeros_like(rec)

                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    for ax, (title, img) in zip(axes, [("Ground Truth", gth), ("PerSTree", rec)]):
                        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                        ax.set_title(title)
                        ax.axis("off")
                    plt.tight_layout()

                    fig_filename = f"{dataset_name}_{subset_name}_{image_name}.png"
                    fig_path = os.path.join(output_fig_dir, fig_filename)
                    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
                    plt.close()

                    base_name = re.sub(r'\d+', '', dataset_name + subset_name)
                    base_name = base_name.replace("_", "")
                    count = created_commands.get(base_name, 0)
                    created_commands[base_name] = count + 1
                    suffix = number_to_letters(count)
                    unique_name = f"{base_name}{suffix}"

                    latex_code = (
                        f"\\newcommand{{\\Plot{unique_name}}}{{%\n"
                        "\\begin{figure}[H]\n"
                        "    \\centering\n"
                        f"    \\includegraphics[width=0.8\\textwidth]{{imgs/{fig_filename}}}\n"
                        f"    \\caption{{{dataset_name} - {subset_name} - {image_name}}}\n"
                        f"    \\label{{fig:{dataset_name}_{subset_name}_{image_name}}}\n"
                        "\\end{figure}%\n"
                        "}\n"
                    )
                    f_tex.write(latex_code)
                    all_commands.append(f"\\Plot{unique_name}")

    return all_commands


# ----------------------------------------------------------
# 5️⃣ MAIN PIPELINE
# ----------------------------------------------------------
def main():
    print("📘 Lettura HDF5 e costruzione results...")
    results = build_results_with_distances(H5_PATH, OUTPUT_JSON)
    print("📗 Generazione tabelle metriche...")
    table_cmds = generate_tables_tex(results, OUTPUT_TEX_TABLES)

    print("📙 Generazione tabelle distanze...")
    distance_cmds = generate_distance_tables_tex(results, OUTPUT_TEX_DISTANCES)

    print("📕 Generazione plot immagini migliori...")
    plot_cmds = generate_plots_tex(H5_PATH, OUTPUT_TEX_PLOTS, OUTPUT_FIG_DIR)

    # Unione di tutti i comandi
    all_cmds = table_cmds + distance_cmds + plot_cmds
    with open(OUTPUT_TXT_ALL, "w") as f_all:
        for c in all_cmds:
            f_all.write(c + "\n")

    print("\n✅ Tutti i file generati:")
    print(f"   Tables:    {OUTPUT_TEX_TABLES}")
    print(f"   Distances: {OUTPUT_TEX_DISTANCES}")
    print(f"   Plots:     {OUTPUT_TEX_PLOTS}")
    print(f"   Commands:  {OUTPUT_TXT_ALL}")


if __name__ == "__main__":
    main()
