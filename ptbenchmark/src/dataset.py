import h5py
import numpy as np
import os


def rgb_to_grayscale(rgb_array):
    # Controlla se i canali sono nella prima dimensione o nell'ultima
    if rgb_array.shape[0] == 3:
        # canali come prima dimensione (3, H, W)
        R, G, B = rgb_array[0], rgb_array[1], rgb_array[2]
    elif rgb_array.shape[-1] == 3:
        # canali come ultima dimensione (H, W, 3)
        R, G, B = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    else:
        raise ValueError("Input rgb_array non ha 3 canali nella prima o ultima dimensione")
    
    grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return grayscale.astype(np.uint16)


class Dataset:
    def __init__(self, datasets_folder, dataset_name, return_image_name=False):
        self.dataset_name = dataset_name
        self.return_image_name = return_image_name
        self.datasets_folder = datasets_folder
        with h5py.File(os.path.join(self.datasets_folder, f"{self.dataset_name}.h5"), "r") as f:
            self.list_sets = list(f.keys())
            self.list_sets.remove("original")
            self.list_sets = sorted(
                self.list_sets,
                key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0,
                reverse=False
            )
            self.list_images = list(f["original"].keys())
        self.set_type = None
    def __len__(self):
        return len(self.list_images)

    def get_subsets(self):
        return(self.list_sets)

    def set_subset(self, set_name):
        if set_name in self.list_sets:
            self.set_type = set_name
        else:
            print("Subset not found!")

    def __getitem__(self, i):
        if self.set_type is None:
            self.set_type = self.list_sets[0]
        with h5py.File(os.path.join(self.datasets_folder, f"{self.dataset_name}.h5"), "r") as f:
            img = np.array(f[self.set_type][self.list_images[i]])
            gth = np.array(f["original"][self.list_images[i]])

        if len(img.shape) == 3:
            img = rgb_to_grayscale(img)
        if len(gth.shape) == 3:
            gth = rgb_to_grayscale(gth)

        if img.max() - img.min() != 1:
            img = (img - img.min()) / (img.max() - img.min())
        if gth.max() - gth.min() != 1:
            gth = (gth - gth.min()) / (gth.max() - gth.min())

        if self.return_image_name: 
            return self.list_images[i], img, gth
        else:
            return img, gth

    def get_labels(self):
        return self.list_images
        
    def get_gth_from_label(self, label):
        with h5py.File(os.path.join(self.datasets_folder, f"{self.dataset_name}.h5"), "r") as f:
            gth = np.array(f["original"][label])
        if len(gth.shape) == 3:
            gth = rgb_to_grayscale(gth)
        if gth.max() - gth.min() != 1:
            gth = (gth - gth.min()) / (gth.max() - gth.min())
        if gth.max() - gth.min() != 1:
            gth = (gth - gth.min()) / (gth.max() - gth.min())

        return gth
        
                