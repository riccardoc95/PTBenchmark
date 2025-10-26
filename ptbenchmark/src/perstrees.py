import numpy as np
import pixhomology as px
from scipy import ndimage

def max_jump_threshold(lifetimes):
    """
    from: https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2024.1260828/full
    :param lifetimes: array of lifetimes
    :return: threshold value
    """
    L = lifetimes
    SL = np.sort(L)
    J = SL[1:] - SL[:-1]
    i_max = np.argmax(J)
    thr = (SL[i_max - 2] + SL[i_max - 1]) / 2
    
    return np.median(L) / 2


class PersTree:
    def __init__(self, image, lifetime_t=None, cut=False, cut_mode="nearest"):
        rows, cols = image.shape
        min_ = image.min()
        max_ = image.max()

        image = (image - min_) / (max_ - min_)

        dgm, dgm_idx, edges, labels = px.computePH(image, return_index=True)

        edges = edges.flatten().astype(np.int32)
        labels = labels.flatten().astype(np.int32)

        dgm_to_labels = dgm_idx[:, 1] * image.shape[1] + dgm_idx[:, 0]
        common = np.intersect1d(labels, dgm_to_labels)
        diff = np.setdiff1d(labels, dgm_to_labels)

        while True:
            mask = np.isin(labels, diff).flatten()
            if not mask.any():
                break

            new_edges = edges[edges][mask]
            labels[mask] = labels[new_edges]
            edges[mask] = new_edges

        index_map = {idx: i for i, idx in enumerate(dgm_to_labels)}
        positions = np.array([index_map[q] for q in labels])
        births = dgm[:, 0][positions]
        deaths = dgm[:, 1][positions]
        lifetimes = births - deaths

        if cut:
            if lifetime_t is None:
                lifetime_t = max_jump_threshold(np.unique(lifetimes))

            if cut_mode == "nearest":
                labels[lifetimes <= lifetime_t] = -1
                edges = edges.reshape(image.shape)
                labels = labels.reshape(image.shape)

                mask = labels != -1

                # Calcolo della distanza e degli indici del punto valido più vicino
                dist, (idx_x, idx_y) = ndimage.distance_transform_edt(~mask, return_indices=True)

                # Assegniamo ad ogni -1 l’etichetta più vicina
                labels[~mask] = labels[idx_x[~mask], idx_y[~mask]]
                edges[~mask] = edges[idx_x[~mask], idx_y[~mask]]
                labels = labels.astype(np.int32).flatten()
                edges = edges.astype(np.int32).flatten()
                births = births[labels]
                deaths = deaths[labels]
                #lifetimes[lifetimes <= lifetime_t] = 0
                lifetimes = lifetimes[labels]
                if (lifetimes.max() - lifetimes.min()) != 0:
                    lifetimes = (lifetimes - lifetimes.min())/(lifetimes.max() - lifetimes.min())
                
            elif cut_mode == "tree":
                lifetimes_orig = np.copy(lifetimes)
                while True:
                    edges_mask = np.copy(edges)
                    edges_mask[lifetimes < lifetime_t] = -1
                    mask = edges_mask == -1
                    if not mask.any():
                        break

                    new_edges = edges[edges][mask]
                    labels[mask] = labels[new_edges]
                    births[mask] = births[new_edges]
                    deaths[mask] = deaths[new_edges]
                    lifetimes[mask] = lifetimes_orig[new_edges]
                    edges[mask] = new_edges
                #lifetimes = lifetimes_orig
                #lifetimes[lifetimes_orig <= lifetime_t] = 0
            else:
                raise ValueError("cut_mode must be 'nearest' or 'tree'")

        features = np.hstack([
            labels.reshape(-1, 1),
            image.reshape(-1, 1),
            births.reshape(-1, 1),
            deaths.reshape(-1, 1),
            lifetimes.reshape(-1, 1),
        ])

        # Edges
        edges = np.hstack([
            edges.reshape(-1, 1),
            np.arange(edges.size).reshape(-1, 1)
        ]).astype(np.int32)

        N = features.shape[0]

        # inizializziamo lista dei figli per ogni nodo
        children_lists = [[] for _ in range(N)]
        parent = np.full(N, -1, dtype=np.int32)

        for p, c in edges:
            if p == c:
                continue
            children_lists[p].append(c)
            parent[c] = p

        # costruiamo CSR arrays
        child_index = np.zeros(N, dtype=np.int32)
        children_all = []

        offset = 0
        for i in range(N):
            child_index[i] = offset
            children_all.extend(children_lists[i])
            offset += len(children_lists[i])

        children_all = np.array(children_all, dtype=np.int32)

        self.child_index = child_index
        self.children_all = children_all
        self.parent = parent
        self.features = features
        self.N = len(parent)
        self.rows = rows
        self.cols = cols
        self.lifetime_t = lifetime_t

    def get_children(self, node_id: int) -> np.ndarray:
        start = self.child_index[node_id]
        end = self.child_index[node_id + 1] if node_id + 1 < self.N else len(self.children_all)
        return self.children_all[start:end]

    def get_parent(self, node_id: int) -> int:
        return self.parent[node_id]

    def get_features(self, node_id: int) -> np.ndarray:
        return self.features[node_id]

    def set_features(self, node_id: int, values: np.ndarray):
        """Sostituisce tutte le feature del nodo."""
        self.features[node_id] = values

    def set_feature(self, node_id: int, dim: int, value: float):
        """Modifica una singola feature (colonna) del nodo."""
        self.features[node_id, dim] = value

    def dist_from_root(self):
        parent = np.copy(self.parent)

        n = len(parent)
        distance = np.zeros(n, dtype=int)

        to_update = np.where(parent != -1)[0]

        while len(to_update) > 0:
            distance[to_update] += 1
            to_update = np.array([i for i in to_update if parent[i] != -1])
            parent[to_update.astype(np.int32)] = parent[parent[to_update.astype(np.int32)].astype(np.int32)]
        return np.array(distance)



