import pandas as pd
import optuna

from tqdm import tqdm

from ptbenchmark.src import PersTree
from ptbenchmark.src import rf_distance, rd_distance
from ptbenchmark.src import Dataset


optuna.logging.set_verbosity(optuna.logging.WARNING)
LIST_OF_DISTANCES = {"RD": rd_distance, "RF": rf_distance}



def process_single_image(image_name, img, gth, lfunction):
    # costruzione alberi
    img_tree = PersTree(img, cut=False)
    img_tree_cut = PersTree(img, lifetime_t=None, cut=True, cut_mode="nearest")
    gth_tree = PersTree(gth, cut=False)

    # funzione obiettivo per Optuna
    def objective(trial):
        lifetime_t = trial.suggest_float("lifetime_t", 0, 1)
        img_tree_cut_opt = PersTree(img, lifetime_t=lifetime_t, cut=True, cut_mode="nearest")
        return lfunction(img_tree_cut_opt, gth_tree)

    # studio ottimizzazione
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    return {
        "image_name": image_name,
        "noisy_dist": lfunction(img_tree, gth_tree),
        "mjump_dist": lfunction(img_tree_cut, gth_tree),
        "optuna_dist": study.best_value
    }


def test_pt_distance(datasets_dir, distance = "RD", output_folder = "results/distance"):
    def lfunction(img_tree, gth_tree):
        return LIST_OF_DISTANCES[distance](img_tree, gth_tree)


    for dataset_name in ["CBSD68", "MRI", "FORECAST", "SEN2VENUS", "SIDD", "FMIDD_OH16230", "FMIDD_PVD"]:
        dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
        for subset in dataset.get_subsets():
            dataset.set_subset(subset)
            distances = []
            for image_name, img, gth in tqdm(dataset):
                distances.append(process_single_image(image_name, img, gth, lfunction))

            pd.DataFrame(distances).to_csv(f"{output_folder}/{dataset_name}_{subset}_{distance}_distance.csv")
            
if __name__ == "__main__":
    test_pt_distance(datasets_dir="datasets", distance = "RD", output_folder = "results/distance")