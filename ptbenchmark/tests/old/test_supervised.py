import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from tqdm import tqdm
import random
import h5py
import os
import sys
import time

from ptbenchmark.src import Dataset
from ptbenchmark.src import DnCNN, UNet, CropDataset, train_one_epoch, validate


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def test_supervised(datasets_dir, output_folder, train_ratio=0.8, batch_size=8, n_epochs=20, device="cpu"):
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    sys.stdout = Logger(os.path.join(output_folder, "logfile.txt"))

    for dataset_name in ["CBSD68", "SEN2VENUS", "FMIDD_OH16230", "FMIDD_PVD", "SIDD",  "FORECAST", "MRI"]:
        dataset = Dataset(datasets_dir, dataset_name, return_image_name=True)
        for subset in dataset.get_subsets():
            dataset.set_subset(subset)

            print(f"---------- DATASET: {dataset_name}, SUBSET: {subset} ----------")

            min_w, min_h = float('inf'), float('inf')
            for i in range(len(dataset)):
                _, img, _ = dataset[i]
                w, h = img.shape
                min_w = min(min_w, w)
                min_h = min(min_h, h)

            target_size = (min_w, min_h)

            n_total = len(dataset)
            n_train = int(n_total * train_ratio)
            n_valid = n_total - n_train

            train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid])


            train_indices = train_dataset.indices
            valid_indices = valid_dataset.indices

            train_image_names = [dataset[i][0] for i in train_indices]
            valid_image_names = [dataset[i][0] for i in valid_indices]

            # --- Save as .npy ---
            np.save(os.path.join(output_folder, f"{dataset_name}_{subset}_train_image_names.npy"), np.array(train_image_names))
            np.save(os.path.join(output_folder, f"{dataset_name}_{subset}_valid_image_names.npy"), np.array(valid_image_names))

            # --- Optional alternative: save as .txt ---
            #with open(os.path.join(output_folder, f"{dataset_name}_{subset}_train_image_names.txt"), "w") as f:
            #    f.writelines(f"{name}\n" for name in train_image_names)

            with open(os.path.join(output_folder, f"{dataset_name}_{subset}_valid_image_names.txt"), "w") as f:
                f.writelines(f"{name}\n" for name in valid_image_names)

            print(f"Saved {len(train_image_names)} train and {len(valid_image_names)} valid image names.")

            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            train_dataset = CropDataset(train_dataset, target_size=target_size, transform=transform)
            valid_dataset = CropDataset(valid_dataset, target_size=target_size, transform=transform)


            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            print(f"Total samples: {n_total}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(valid_dataset)}")


            for model_name, model in {"UNet": UNet(n_channels=1, n_classes=1),
                                      "DnCNN":DnCNN(channels=1)}.items():

                print(f"---------- MODEL: {model_name} ----------")

                model = UNet(n_channels=1, n_classes=1)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)



                _=model.to(device)

                start_train_time = time.time()


                for epoch in range(n_epochs):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_psnr, val_ssim = validate(model, valid_loader, criterion, device)

                    print(f"Epoch [{epoch+1}/{n_epochs}] "
                          f"| Train Loss: {train_loss:.6f} "
                          f"| Val Loss: {val_loss:.6f} "
                          f"| PSNR: {val_psnr:.4f} "
                          f"| SSIM: {val_ssim:.4f}")

                end_train_time = time.time()
                training_time = end_train_time - start_train_time
                print(f"Training time: {training_time:.2f} seconds")

                # Save the trained model.
                torch.save(model.state_dict(), os.path.join(output_folder, f"{model_name}_{dataset_name}_{subset}_model.pth"))

                # --- Save predictions for the validation set ---
                model.eval()
                val_predictions = []
                val_targets = []
                val_image_names = []

                total_pred_time = []
                with torch.no_grad():
                    for batch in tqdm(valid_loader, desc="Saving validation outputs"):
                        image_names, inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        start_pred = time.time()
                        outputs = model(inputs)
                        end_pred = time.time()

                        total_pred_time.append(end_pred - start_pred)

                        # Move to CPU and convert to NumPy.
                        outputs_np = outputs.squeeze(1).cpu().numpy()
                        targets_np = targets.squeeze(1).cpu().numpy()

                        val_predictions.append(outputs_np)
                        val_targets.append(targets_np)
                        val_image_names.extend(image_names)

                # Concatenate all batches.
                val_predictions = np.concatenate(val_predictions, axis=0)
                val_targets = np.concatenate(val_targets, axis=0)
                val_image_names = np.array(val_image_names, dtype='S')  # Store as byte strings.

                avg_pred_time = np.mean(total_pred_time)
                std_pred_time = np.std(total_pred_time)
                print(f"Average prediction time per image: {avg_pred_time:.4f} +/- {std_pred_time:.4f} seconds")

                # --- Save as an .h5 file ---
                with h5py.File(os.path.join(output_folder, f"{model_name}_{dataset_name}_{subset}_valid_results.h5"), "w") as f:
                    f.create_dataset("image_names", data=val_image_names)
                    f.create_dataset("predictions", data=val_predictions)
                    f.create_dataset("targets", data=val_targets)
                    f.attrs["training_time"] = training_time
                    f.attrs["avg_prediction_time"] = avg_pred_time
                    f.attrs["std_prediction_time"] = std_pred_time

                print(f"File 'validation_results.h5' saved with {len(val_image_names)} images.")
                print(f"   training_time_sec = {training_time:.2f}")
                print(f"   avg_prediction_time_per_image_sec = {avg_pred_time:.4f} +/- {std_pred_time:.4f}")


if __name__ == "__main__":
    output_folder = "results/supervised"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    test_supervised(datasets_dir="datasets", output_folder=output_folder, train_ratio=0.8, batch_size=8, n_epochs=20, device="cpu")
