import h5py
import json
import pandas as pd

def compare_results(h5_path):
    results = []

    with h5py.File(h5_path, "r") as f:
        # Loop su dataset (es. CBSD68, SEN2VENUS, ...)
        for dataset_name in f.keys():
            dataset_group = f[dataset_name]

            # Loop sui subset
            for subset_name in dataset_group.keys():
                subset_group = dataset_group[subset_name]

                # Loop sulle immagini
                for image_name in subset_group.keys():
                    img_group = subset_group[image_name]

                    # --- Versione standard ---
                    grp_std = img_group["perstree_rec"]
                    mse_std = grp_std.attrs["mse"]
                    params_std = json.loads(grp_std.attrs["params"])
                    niter_std = params_std.get("niter", None)

                    # --- Versione con SEC ---
                    grp_sec = img_group["perstree_rec_sec"]
                    mse_sec = grp_sec.attrs["mse"]
                    params_sec = json.loads(grp_sec.attrs["params"])
                    niter_sec = params_sec.get("niter", None)

                    # --- Aggiungi risultati alla lista ---
                    results.append({
                        "dataset": dataset_name,
                        "subset": subset_name,
                        "image": image_name,
                        "mse_std": mse_std,
                        "niter_std": niter_std,
                        "mse_sec": mse_sec,
                        "niter_sec": niter_sec,
                        "Δmse": mse_sec - mse_std,
                        "Δniter": niter_sec - niter_std
                    })

    # Converte in DataFrame per analisi facile
    df = pd.DataFrame(results)
    return df


# --- ESEMPIO DI UTILIZZO ---
h5_path = "results/results_spatial_entropy_change.h5"
df = compare_results(h5_path)

# Mostra riepilogo generale
print("\n📊 Summary by dataset:")
print(df.groupby("dataset")[["mse_std", "mse_sec", "Δmse", "niter_std", "niter_sec", "Δniter"]].mean().round(4))

# Mostra qualche esempio
print("\n🔍 Sample rows:")
print(df.head())

# Eventuale esportazione
df.to_csv("results_comparison.csv", index=False)
