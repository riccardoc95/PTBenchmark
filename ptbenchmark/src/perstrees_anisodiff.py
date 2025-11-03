from .perstrees import PersTree

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

import time


def lifetime_denoise_tree(values, lifetimes, parents, child_index, children_all, rows, cols,
                          num_iter=1):

    v = values.copy().astype(float)
    n = len(v)

    row_coords = np.arange(n) // cols
    col_coords = np.arange(n) % cols

    L = (lifetimes - lifetimes.min()) / (lifetimes.max() - lifetimes.min() + 1e-12)
    base_relax = 1e-05 * (1.0 - L)

    for iteration in range(num_iter):
        v_2d = v.reshape(rows, cols)

        gx = np.zeros_like(v_2d)
        gy = np.zeros_like(v_2d)
        gx[:, 1:] = np.diff(v_2d, axis=1)
        gy[1:, :] = np.diff(v_2d, axis=0)

        grad_strength = (gx**2 + gy**2).flatten()
        grad_strength = grad_strength / (grad_strength.max() + 1e-12)

        agg = np.zeros_like(v)
        count = np.zeros_like(v)

        for i in range(n):
            if i < len(child_index) - 1:
                start, end = child_index[i], child_index[i + 1]
                if end > start:
                    children = children_all[start:end]
                    pi_row, pi_col = row_coords[i], col_coords[i]

                    for c in children:
                        pc_row, pc_col = row_coords[c], col_coords[c]
                        dx, dy = abs(pc_row - pi_row), abs(pc_col - pi_col)
                        spatial_dist = max(dx, dy, 1)
                        lifetime_sim = np.exp(-2 * L[i])#abs(L[i] - L[c]))
                        aniso_weight = lifetime_sim / spatial_dist

                        agg[i] += aniso_weight * v[c]
                        count[i] += aniso_weight

        mask = count > 1e-6
        edge_factor = 1 - grad_strength
        relax_bu = base_relax * edge_factor
        v[mask] = (1 - relax_bu[mask]) * v[mask] + relax_bu[mask] * (agg[mask] / count[mask])

        for i in range(n):
            p = parents[i]
            if p != -1:
                dx = row_coords[i] - row_coords[p]
                dy = col_coords[i] - col_coords[p]

                direction_factor = 1.0
                if abs(dx) == abs(dy):
                    direction_factor = 1.2
                elif dx == 0 or dy == 0:
                    direction_factor = 1.1

                lifetime_sim = np.exp(-2 * L[i])#abs(L[i] - L[p]))
                relax_td = base_relax[i] * direction_factor * lifetime_sim
                #relax_td = min(relax_td, 0.7)

                v[i] = (1 - relax_td) * v[i] + relax_td * v[p]

        v_2d = v.reshape(rows, cols)
        radius = 1#max(2, 4 - iteration // 3)  
        epsilon = 0.0005 * (1 + grad_strength.reshape(rows, cols)).mean()  
        #v_2d_smooth = guided_filter_simple(v_2d, v_2d, radius, epsilon)

        rows, cols = v_2d.shape    
        mean_image = uniform_filter(v_2d, size=2*radius+1) 
        mean_guide_image = uniform_filter(v_2d * v_2d, size=2*radius+1)
    
        cov_guide_image = mean_guide_image - mean_image * mean_image
        var_guide = uniform_filter(v_2d * v_2d, size=2*radius+1) - mean_image * mean_image
    
        a = cov_guide_image / (var_guide + epsilon)
        b = mean_image - a * mean_image
    
        mean_a = uniform_filter(a, size=2*radius+1)
        mean_b = uniform_filter(b, size=2*radius+1)    
        v_2d_smooth = mean_a * v_2d + mean_b
    
        
        v = v_2d_smooth.flatten()
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)

    return v.reshape(rows, cols)


def perstree_anisodiff(img, gth, max_iter=100, lifetime_t=None, cut=False, cut_mode="nearest"):
    tree = PersTree(img, lifetime_t=lifetime_t, cut=cut, cut_mode=cut_mode)
    labels = tree.features[:,0].astype(int).copy()
    values = tree.features[:,1].copy()
    lifetimes = tree.features[:,-1].copy()
    lifetime_t = tree.lifetime_t
    
    parents = tree.parent
    child_index = tree.child_index
    children_all = tree.children_all
    rows, cols = tree.rows, tree.cols

    best_mse = 1e10
    best_u = None
    best_time = None
    
    for i in range(max_iter):
        t0 = time.time()
        rec = lifetime_denoise_tree(values, lifetimes, parents, child_index, children_all, rows, cols)
        exec_time = time.time() - t0
        mse = np.mean((gth - rec)**2)
        if mse < best_mse:
            best_mse = mse
            best_u = rec.copy()
            values = rec.flatten()
            best_time = exec_time
        else:
            break
    return best_u, best_mse, {"niter":i, "lifetime_t":lifetime_t, "time": best_time}

    