import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
from bm3d import bm3d

import time


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def median_denoise(img, gth, size_list=[3,5,7,9]):
    best_mse = 1e10
    best_size = None
    best_img = None
    best_time = None
    for s in size_list:
        t0 = time.time()
        den = median_filter(img, size=s)
        exec_time = time.time() - t0
        cur_mse = mse(den, gth)
        if cur_mse < best_mse:
            best_mse = cur_mse
            best_size = s
            best_img = den.copy()
            best_time = exec_time
    return best_img, best_mse, {"size": best_size, "time": best_time}

def gaussian_denoise(img, gth, sigma_list=[0.5,1,1.5,2,2.5,3]):
    best_mse = 1e10
    best_sigma = None
    best_img = None
    best_time = None
    for s in sigma_list:
        t0 = time.time()
        den = gaussian_filter(img, sigma=s)
        exec_time = time.time() - t0
        cur_mse = mse(den, gth)
        if cur_mse < best_mse:
            best_mse = cur_mse
            best_sigma = s
            best_img = den.copy()
            best_time = exec_time
    return best_img, best_mse, {"sigma": best_sigma, "time": best_time}

def wavelet_denoise(img, gth, method_list=['BayesShrink','VisuShrink'], mode_list=['soft','hard']):
    best_mse = 1e10
    best_params = None
    best_img = None
    best_time = None
    for method in method_list:
        for mode in mode_list:
            t0 = time.time()
            den = denoise_wavelet(img, method=method, mode=mode, rescale_sigma=True)
            exec_time = time.time() - t0
            cur_mse = mse(den, gth)
            if cur_mse < best_mse:
                best_mse = cur_mse
                best_time = exec_time
                best_params = {"method": method, "mode": mode, "time": best_time}
                best_img = den.copy()
                
    return best_img, best_mse, best_params

def nl_means_denoise(img, gth, patch_size_list=[3,5,7], patch_distance_list=[3,5,7], h_factor=[0.8,1.0,1.2]):
    sigma_est = np.mean(estimate_sigma(img))
    best_mse = 1e10
    best_params = None
    best_img = None
    best_time = None
    for ps in patch_size_list:
        for pd in patch_distance_list:
            for hf in h_factor:
                t0 = time.time()
                den = denoise_nl_means(img, h=hf*sigma_est, patch_size=ps, patch_distance=pd, fast_mode=True)
                exec_time = time.time() - t0
                cur_mse = mse(den, gth)
                if cur_mse < best_mse:
                    best_mse = cur_mse
                    best_time = exec_time
                    best_params = {"patch_size": ps, "patch_distance": pd, "h_factor": hf, "time": best_time}
                    best_img = den.copy()
                    
    return best_img, best_mse, best_params

def bm3d_denoise(img, gth, sigma_list=[5, 10, 15, 20]):
    best_mse = float('inf')
    best_sigma = None
    best_img = None
    best_time = None

    for s in sigma_list:
        t0 = time.time()
        denoised = bm3d(img, sigma_psd=s/255.0)
        exec_time = time.time() - t0
        cur_mse = np.mean((denoised - gth) ** 2)
        if cur_mse < best_mse:
            best_mse = cur_mse
            best_sigma = s
            best_img = denoised.copy()
            best_time = exec_time

    return best_img, best_mse, {"sigma": best_sigma, "time": best_time}

def anisodiff(u, u0, g, k, dt=0.2, T_max=500, tol=1e-10):
    """
    Esegue la diffusione anisotropa di Perona-Malik.
    Parametri:
      u0 : immagine normalizzata [0,1]
      g  : tipo di funzione conduttività (1–5)
      k  : parametro di soglia
      dt : passo temporale
      T_max : massimo numero di iterazioni
    """
    ni, nj = u.shape
    mse, mse_old = 1e3, 2e3

    for T in range(T_max):
        mse_old = mse

        # differenze direzionali
        DuN = np.zeros_like(u)
        DuS = np.zeros_like(u)
        DuE = np.zeros_like(u)
        DuW = np.zeros_like(u)

        DuN[1:, :] = u[:-1, :] - u[1:, :]
        DuS[:-1, :] = u[1:, :] - u[:-1, :]
        DuW[:, 1:] = u[:, :-1] - u[:, 1:]
        DuE[:, :-1] = u[:, 1:] - u[:, :-1]

        # Conduttività
        if g == 1:
            cN = 1.0 / (1.0 + (DuN / k) ** 2)
            cS = 1.0 / (1.0 + (DuS / k) ** 2)
            cW = 1.0 / (1.0 + (DuW / k) ** 2)
            cE = 1.0 / (1.0 + (DuE / k) ** 2)
        elif g == 2:
            cN = np.exp(-(DuN / k) ** 2)
            cS = np.exp(-(DuS / k) ** 2)
            cW = np.exp(-(DuW / k) ** 2)
            cE = np.exp(-(DuE / k) ** 2)
        elif g == 3:
            maskN = np.abs(DuN) <= k * np.sqrt(2)
            maskS = np.abs(DuS) <= k * np.sqrt(2)
            maskW = np.abs(DuW) <= k * np.sqrt(2)
            maskE = np.abs(DuE) <= k * np.sqrt(2)
            cN = np.where(maskN, 0.5 * (1 - (DuN / (k * np.sqrt(2))) ** 2) ** 2, 0)
            cS = np.where(maskS, 0.5 * (1 - (DuS / (k * np.sqrt(2))) ** 2) ** 2, 0)
            cW = np.where(maskW, 0.5 * (1 - (DuW / (k * np.sqrt(2))) ** 2) ** 2, 0)
            cE = np.where(maskE, 0.5 * (1 - (DuE / (k * np.sqrt(2))) ** 2) ** 2, 0)
        elif g == 4:
            f = lambda d: 1.0 / (1.0 + np.abs(d / k) ** (2 - 2 / (1 + np.abs(d / k) ** 2)))
            cN, cS, cW, cE = f(DuN), f(DuS), f(DuW), f(DuE)
        elif g == 5:
            f = lambda d: np.where(np.abs(d) != 0, 1 - np.exp(-3.31488 * (k / d) ** 8), 1)
            cN, cS, cW, cE = f(DuN), f(DuS), f(DuW), f(DuE)
        else:
            raise ValueError("Filtro g deve essere un intero da 1 a 5")

        # Aggiornamento
        u_new = u + dt * (cN * DuN + cS * DuS + cW * DuW + cE * DuE)

        u = u_new

        # MSE rispetto all'originale
        mse = np.mean((u - u0) ** 2)
        if mse_old <= mse or abs(mse_old - mse) / mse_old < tol:
            break

    return u, mse, T

def perona_malik(img, gth,
                 k_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5, 10, 20, 25]):

    best_mse = 1e10
    best_time = None
    best_k = None
    best_u = None
    g=1
    for k in k_list:
        t0 = time.time()
        u, mse, T = anisodiff(img, gth, g, k)
        exec_time = time.time() - t0
        if mse < best_mse:
            best_mse = mse
            best_k = k
            best_u = u.copy()
            best_time = exec_time
    return best_u, best_mse, {"niter":T, "k": best_k, "time": best_time}
