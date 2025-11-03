import h5py
import json
import time
import os
from tqdm import tqdm

from ptbenchmark.src import Dataset
from ptbenchmark.src import (pt_denoise, pm_denoise, median_denoise, gaussian_denoise,
                             wavelet_denoise, nl_means_denoise, bm3d_denoise)



def save_image_result(dataset_name, subset, image_name,
                      img, gth,
                      perstree_data, perstree_cut_data, peronamalik_data,
                      median_data, gaussian_data, wavelet_data, nlmeans_data, bm3d_data,
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
                grp_pc = img_group.require_group("perstree_rec_cut")
                grp_pc.create_dataset("data", data=perstree_cut_data["rec"], compression="gzip")
                grp_pc.attrs["mse"] = perstree_cut_data["mse"]
                grp_pc.attrs["params"] = json.dumps(perstree_cut_data["params"])
                grp_pc.attrs["training_time"] = perstree_cut_data["time"]

                # --- Perona-Malik ---
                grp_pm = img_group.require_group("peronamalik_rec")
                grp_pm.create_dataset("data", data=peronamalik_data["rec"], compression="gzip")
                grp_pm.attrs["mse"] = peronamalik_data["mse"]
                grp_pm.attrs["params"] = json.dumps(peronamalik_data["params"])
                grp_pm.attrs["training_time"] = peronamalik_data["time"]

                # --- Median ---
                grp_med = img_group.require_group("median_rec")
                grp_med.create_dataset("data", data=median_data["rec"], compression="gzip")
                grp_med.attrs["mse"] = median_data["mse"]
                grp_med.attrs["params"] = json.dumps(median_data["params"])
                grp_med.attrs["training_time"] = median_data["time"]

                # --- Gaussian ---
                grp_gauss = img_group.require_group("gaussian_rec")
                grp_gauss.create_dataset("data", data=gaussian_data["rec"], compression="gzip")
                grp_gauss.attrs["mse"] = gaussian_data["mse"]
                grp_gauss.attrs["params"] = json.dumps(gaussian_data["params"])
                grp_gauss.attrs["training_time"] = gaussian_data["time"]

                # --- Wavelet ---
                grp_wave = img_group.require_group("wavelet_rec")
                grp_wave.create_dataset("data", data=wavelet_data["rec"], compression="gzip")
                grp_wave.attrs["mse"] = wavelet_data["mse"]
                grp_wave.attrs["params"] = json.dumps(wavelet_data["params"])
                grp_wave.attrs["training_time"] = wavelet_data["time"]

                # --- NL-MEANS ---
                grp_nl = img_group.require_group("nlmeans_rec")
                grp_nl.create_dataset("data", data=nlmeans_data["rec"], compression="gzip")
                grp_nl.attrs["mse"] = nlmeans_data["mse"]
                grp_nl.attrs["params"] = json.dumps(nlmeans_data["params"])
                grp_nl.attrs["training_time"] = nlmeans_data["time"]

                # --- BM3D ---
                grp_bm = img_group.require_group("bm3d_rec")
                grp_bm.create_dataset("data", data=bm3d_data["rec"], compression="gzip")
                grp_bm.attrs["mse"] = bm3d_data["mse"]
                grp_bm.attrs["params"] = json.dumps(bm3d_data["params"])
                grp_bm.attrs["training_time"] = bm3d_data["time"]

            # Se tutto va bene, esce dal loop
            break

        except (OSError, BlockingIOError) as e:
            print(f"⚠️  File HDF5 bloccato. Retry {retries+1}/{max_retries} in 5 secondi...")
            time.sleep(5)
            retries += 1

    else:
        # Se dopo max_retries non si riesce a scrivere, solleva l'errore
        raise RuntimeError(f"Impossibile scrivere su {output_file} dopo {max_retries} tentativi.")




def test_unsupervised(datasets_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for dataset_name in ["CBSD68", "SEN2VENUS", "FMIDD_OH16230", "FMIDD_PVD", "SIDD",  "FORECAST", "MRI"]:
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
                perstree_rec_cut, perstree_mse_cut, perstree_params_cut = pt_denoise(
                    img, gth, max_iter=100, lifetime_t=None, cut=True
                )
                perstree_time_cut = time.time() - t1
    
                t2 = time.time()
                peronamalik_rec, peronamalik_mse, peronamalik_params = pm_denoise(img, gth)
                peronamalik_time = time.time() - t2
    
                t3 = time.time()
                median_rec, median_mse, median_params = median_denoise(img, gth)
                median_time = time.time() - t3

                t4 = time.time()
                gaussian_rec, gaussian_mse, gaussian_params = gaussian_denoise(img, gth)
                gaussian_time = time.time() - t4

                t5 = time.time()
                wavelet_rec, wavelet_mse, wavelet_params = wavelet_denoise(img, gth)
                wavelet_time = time.time() - t5

                t6 = time.time()
                nlmeans_rec, nlmeans_mse, nlmeans_params = nl_means_denoise(img, gth)
                nlmeans_time = time.time() - t6

                t7 = time.time()
                bm3d_rec, bm3d_mse, bm3d_params = bm3d_denoise(img, gth)
                bm3d_time = time.time() - t7

                save_image_result(
                    dataset_name, subset, image_name,
                    img, gth,
                    perstree_data={"rec": perstree_rec, 
                                   "mse": perstree_mse, 
                                   "params": perstree_params, 
                                   "time": perstree_time},
                    perstree_cut_data={"rec": perstree_rec_cut, 
                                       "mse": perstree_mse_cut, 
                                       "params": perstree_params_cut, 
                                       "time": perstree_time_cut},
                    peronamalik_data={"rec": peronamalik_rec, 
                                      "mse": peronamalik_mse, 
                                      "params": peronamalik_params, 
                                      "time": peronamalik_time},
                    median_data={"rec": median_rec, 
                                 "mse": median_mse, 
                                 "params": median_params, 
                                 "time": median_time},
                    gaussian_data={"rec": gaussian_rec, 
                                   "mse": gaussian_mse, 
                                   "params": gaussian_params, 
                                   "time": gaussian_time},
                    wavelet_data={"rec": wavelet_rec, 
                                  "mse": wavelet_mse, 
                                  "params": wavelet_params, 
                                  "time": wavelet_time},
                    nlmeans_data={"rec": nlmeans_rec, 
                                  "mse": nlmeans_mse, 
                                  "params": nlmeans_params, 
                                  "time": nlmeans_time},
                    bm3d_data={"rec": bm3d_rec, 
                               "mse": bm3d_mse, 
                               "params": bm3d_params, 
                               "time": bm3d_time},
                    output_file=output_file
                )

if __name__ == "__main__":
    output_file = os.path.join("results", "results_unsupervised.h5")
    test_unsupervised(datasets_dir="datasets", output_file=output_file)
