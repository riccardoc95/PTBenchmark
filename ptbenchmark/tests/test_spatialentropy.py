import h5py
import json
import time
import os
from tqdm import tqdm

from ptbenchmark.src import Dataset
from ptbenchmark.src import pt_denoise, pt_denoise_sec


def save_image_result(dataset_name, subset, image_name,
                      img, gth,
                      perstree_data, perstree_sec_data,
                      output_file, max_retries=10):
    """
    Salva i risultati di una singola immagine in un file HDF5.
    Se il file è bloccato da un altro processo, aspetta 2 secondi e riprova.
    """
    retries = 0
    while retries < max_retries:
        try:
            # Apertura in modalità append
            with h5py.File(output_file, "a") as h5f:
                dataset_group = h5f.require_group(dataset_name)
                subset_group = dataset_group.require_group(subset)
                img_group = subset_group.require_group(image_name)

                # --- Dati principali ---
                if "img" not in img_group:
                    img_group.create_dataset("img", data=img, compression="gzip")
                if "gth" not in img_group:
                    img_group.create_dataset("gth", data=gth, compression="gzip")

                # --- Perstree senza cut ---
                grp_p = img_group.require_group("perstree_rec")
                grp_p.create_dataset("data", data=perstree_data["rec"], compression="gzip")
                grp_p.attrs["mse"] = perstree_data["mse"]
                grp_p.attrs["params"] = json.dumps(perstree_data["params"])
                grp_p.attrs["training_time"] = perstree_data["time"]

                # --- Perstree con cut ---
                grp_pc = img_group.require_group("perstree_rec_sec")
                grp_pc.create_dataset("data", data=perstree_sec_data["rec"], compression="gzip")
                grp_pc.attrs["mse"] = perstree_sec_data["mse"]
                grp_pc.attrs["params"] = json.dumps(perstree_sec_data["params"])
                grp_pc.attrs["training_time"] = perstree_sec_data["time"]

            break

        except (OSError, BlockingIOError) as e:
            print(f"⚠️  File HDF5 bloccato. Retry {retries+1}/{max_retries} in 5 secondi...")
            time.sleep(5)
            retries += 1

    else:
        # Se dopo max_retries non si riesce a scrivere, solleva l'errore
        raise RuntimeError(f"Impossibile scrivere su {output_file} dopo {max_retries} tentativi.")


def test_spatialentropychange(datasets_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for dataset_name in ["CBSD68", "SEN2VENUS", "FMIDD_OH16230", "FMIDD_PVD", "SIDD", "FORECAST", "MRI"]:
        dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)

        for subset in dataset.get_subsets():
            dataset.set_subset(subset)

            for image_name, img, gth in tqdm(dataset, desc=f"{dataset_name}/{subset}"):
                t0 = time.time()
                perstree_rec, perstree_mse, perstree_params = pt_denoise(
                    img, gth, max_iter=100, lifetime_t=None, cut=False
                )
                perstree_time = time.time() - t0

                t1 = time.time()
                perstree_rec_sec, perstree_mse_sec, perstree_params_sec = pt_denoise_sec(
                    img, gth, max_iter=100, lifetime_t=None, cut=False
                )
                perstree_time_sec = time.time() - t1

                save_image_result(
                    dataset_name, subset, image_name,
                    img, gth,
                    perstree_data={"rec": perstree_rec,
                                   "mse": perstree_mse,
                                   "params": perstree_params,
                                   "time": perstree_time},
                    perstree_sec_data={"rec": perstree_rec_sec,
                                       "mse": perstree_mse_sec,
                                       "params": perstree_params_sec,
                                       "time": perstree_time_sec},
                    output_file=output_file
                )


if __name__ == "__main__":
    output_file = os.path.join("results", "results_spatial_entropy_change.h5")
    test_spatialentropychange(datasets_dir="datasets", output_file=output_file)
