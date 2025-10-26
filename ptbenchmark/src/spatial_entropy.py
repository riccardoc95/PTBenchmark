import numpy as np

from .perstrees import PersTree
from .perstrees_anisodiff import lifetime_denoise_tree
import time


def spatial_entropy(image, bins=256, normalize=True):
    """
    Compute the spatial entropy of an image based on intensity histogram.

    Parameters
    ----------
    image : ndarray
        Input image (2D or 3D). Can be float or uint8.
    bins : int, optional
        Number of histogram bins (default: 256).
    normalize : bool, optional
        If True, normalize image to [0, 1] before computing histogram.

    Returns
    -------
    entropy : float
        Shannon entropy value (in bits).
    """
    if normalize:
        img = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-12)
    else:
        img = image

    hist, _ = np.histogram(img.ravel(), bins=bins, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def spatial_entropy_change(image_prev, image_curr, bins=256):
    """
    Compute the spatial entropy change between two consecutive images.

    Parameters
    ----------
    image_prev : ndarray
        Image at iteration t-1.
    image_curr : ndarray
        Image at iteration t.
    bins : int, optional
        Number of histogram bins for entropy calculation.

    Returns
    -------
    delta_H : float
        Difference in spatial entropy (H_t - H_{t-1}).
    """
    H_prev = spatial_entropy(image_prev, bins)
    H_curr = spatial_entropy(image_curr, bins)
    return H_curr - H_prev


def sec_perstree_anisodiff(img, gth, max_iter=100, stop_threshold=1e-4, lifetime_t=None, cut=True, cut_mode="nearest"):
    tree = PersTree(img, lifetime_t=lifetime_t, cut=cut, cut_mode=cut_mode)
    values = tree.features[:, 1].copy()
    lifetimes = tree.features[:, -1].copy()
    lifetime_t = tree.lifetime_t

    parents = tree.parent
    child_index = tree.child_index
    children_all = tree.children_all
    rows, cols = tree.rows, tree.cols

    entropy_changes = []
    prev_img = img.copy()

    for t in range(max_iter):
        t0 = time.time()
        rec = lifetime_denoise_tree(values, lifetimes, parents, child_index, children_all, rows, cols)
        exec_time = time.time() - t0
        delta_H = spatial_entropy_change(prev_img, rec)
        entropy_changes.append(delta_H)

        print(delta_H, abs(delta_H))

        if t > 2 and np.sign(entropy_changes[-1]) != np.sign(entropy_changes[-2]):
            best_u = rec.copy()
            best_time = exec_time
            best_mse = np.mean((gth - best_u)**2)
            return best_u, best_mse, {"niter": t, "lifetime_t": lifetime_t, "time": best_time}
        if abs(delta_H) < stop_threshold:
            best_u = rec.copy()
            best_time = exec_time
            best_mse = np.mean((gth - best_u) ** 2)
            return best_u, best_mse, {"niter": t, "lifetime_t": lifetime_t, "time": best_time}

        prev_img = rec.copy()
        values = rec.flatten()


