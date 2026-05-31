import os
import h5py
import json
import optuna
import numpy as np
from tqdm import tqdm

from ptbenchmark.src import PersTree, rf_distance, rd_distance, Dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

LIST_OF_DISTANCES = {"RD": rd_distance, "RF": rf_distance}


def save_distance_to_h5(output_file, dataset_name, subset, image_name, distance_name, data_dict):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "a") as f:
        ds_group = f.require_group(dataset_name)
        subset_group = ds_group.require_group(subset)
        img_group = subset_group.require_group(image_name)

        group_name = f"pt_distance_{distance_name.upper()}"
        if group_name in img_group:
            del img_group[group_name]

        grp = img_group.create_group(group_name)
        for key, val in data_dict.items():
            if key == "params":
                grp.attrs[key] = json.dumps(val)
            else:
                grp.attrs[key] = val


def process_single_image(img, gth, lfunction):
    img_tree = PersTree(img, cut=False)
    img_tree_cut = PersTree(img, lifetime_t=None, cut=True, cut_mode="nearest")
    gth_tree = PersTree(gth, cut=False)

    def objective(trial):
        lifetime_t = trial.suggest_float("lifetime_t", 0, 1)
        img_tree_cut_opt = PersTree(img, lifetime_t=lifetime_t, cut=True, cut_mode="nearest")
        return lfunction(img_tree_cut_opt, gth_tree)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    return {
        "noisy_dist": float(lfunction(img_tree, gth_tree)),
        "mjump_dist": float(lfunction(img_tree_cut, gth_tree)),
        "optuna_dist": float(study.best_value),
        "params": {"best_lifetime_t": study.best_params["lifetime_t"]},
    }


def test_pt_distance(datasets_dir, dataset_name, output_file, distance="RD"):
    if distance not in LIST_OF_DISTANCES:
        raise ValueError(f"Unknown distance: {distance}. Valid choices: {list(LIST_OF_DISTANCES.keys())}")

    lfunction = LIST_OF_DISTANCES[distance]

    dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)

    all_noisy, all_mjump, all_optuna = [], [], []

    for subset in dataset.get_subsets():
        dataset.set_subset(subset)
        print(f"Processing dataset: {dataset_name} | subset: {subset} | distance: {distance}")

        for image_name, img, gth in tqdm(dataset, desc=f"{dataset_name}/{subset}/{distance}"):
            data_dict = process_single_image(img, gth, lfunction)
            save_distance_to_h5(output_file, dataset_name, subset, image_name, distance, data_dict)

            all_noisy.append(data_dict["noisy_dist"])
            all_mjump.append(data_dict["mjump_dist"])
            all_optuna.append(data_dict["optuna_dist"])

        stats = {
            "noisy_mean": np.mean(all_noisy),
            "noisy_var": np.var(all_noisy),
            "mjump_mean": np.mean(all_mjump),
            "mjump_var": np.var(all_mjump),
            "optuna_mean": np.mean(all_optuna),
            "optuna_var": np.var(all_optuna),
        }

        with h5py.File(output_file, "a") as f:
            ds_group = f.require_group(dataset_name)
            subset_group = ds_group.require_group(subset)
            stats_group = subset_group.require_group(f"pt_distance_{distance.upper()}_stats")
            for k, v in stats.items():
                stats_group.attrs[k] = float(v)

        print("\nStatistics for subset:", subset)
        print(f"  - noisy_dist : mean={stats['noisy_mean']:.4f}, var={stats['noisy_var']:.4f}")
        print(f"  - mjump_dist : mean={stats['mjump_mean']:.4f}, var={stats['mjump_var']:.4f}")
        print(f"  - optuna_dist: mean={stats['optuna_mean']:.4f}, var={stats['optuna_var']:.4f}\n")


if __name__ == "__main__":
    test_pt_distance(
        datasets_dir="datasets",
        dataset_name="FORECAST",
        output_file="results/results_all.h5",
        distance="RD"
    )
