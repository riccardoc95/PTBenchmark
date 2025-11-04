import os
import re
import json
import string
import numpy as np

# === Percorsi ===
RESULTS_JSON = "sbatch/results_summary.json"
OUTPUT_TEX_TABLES = "paper/tables.tex"
OUTPUT_TEX_DISTANCES = "paper/tables_distances.tex"
OUTPUT_TEX_PLOTS = "paper/figures_plots.tex"
OUTPUT_TXT_ALL = "paper/table_and_plots.txt"
os.makedirs("paper", exist_ok=True)

methods = [
    "perstree", "perstree_cut", "perstree_rec_sec", "peronamalik", "median",
    "gaussian", "wavelet", "nlmeans", "bm3d", "unet", "dncnn"
]

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

def number_to_letters(n):
    letters = string.ascii_uppercase
    result = ''
    while True:
        result = letters[n % 26] + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


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


def generate_plots_tex(results, output_tex):
    all_cmds = []
    created = {}
    with open(output_tex, "w") as f_tex:
        for dataset_name, subsets in results.items():
            for subset_name, vals in subsets.items():
                best_info = vals.get("best_image", {})
                if not best_info or not best_info.get("path"):
                    continue
                base = dataset_name.replace("_", "")
                count = created.get(base, 0)
                created[base] = count + 1
                suffix = number_to_letters(count)
                unique = f"{base}{suffix}"
                img_path = best_info["path"]
                caption = f"{dataset_name} - {subset_name} - {best_info['name']}"
                latex = (
                    f"\\newcommand{{\\Plot{unique}}}{{%\n"
                    "\\begin{figure}[H]\\centering\n"
                    f"\\includegraphics[width=0.8\\textwidth]{{{img_path}}}\n"
                    f"\\caption{{{caption}}}\n"
                    f"\\label{{fig:{dataset_name}_{subset_name}_{best_info['name']}}}\n"
                    "\\end{figure}%\n}\n"
                )
                f_tex.write(latex)
                all_cmds.append(f"\\Plot{unique}")
    return all_cmds



def main():
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)
    print("📗 Generazione tabelle metriche...")
    t1 = generate_tables_tex(results, OUTPUT_TEX_TABLES)
    print("📙 Generazione tabelle distanze...")
    t2 = generate_distance_tables_tex(results, OUTPUT_TEX_DISTANCES)
    print("📕 Generazione plot immagini migliori...")
    t3 = generate_plots_tex(results, OUTPUT_TEX_PLOTS)

    with open(OUTPUT_TXT_ALL, "w") as f:
        for c in (t1 + t2 + t3):
            f.write(c + "\n")

    print(f"\n✅ File generati:\n- Tables: {OUTPUT_TEX_TABLES}\n- Distances: {OUTPUT_TEX_DISTANCES}\n"
          f"- Plots: {OUTPUT_TEX_PLOTS}\n- Commands: {OUTPUT_TXT_ALL}")