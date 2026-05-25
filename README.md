# PTBenchmark

PTBenchmark is a **Python benchmarking framework and CLI tool** for Persistence Trees (PTs) and related topological descriptors derived from persistent homology. It provides utilities to download datasets, run standardized benchmarks (methods, distances, entropy), and merge results for reproducible experiments.

The command-line interface is built using **Typer** and is the main entry point for running experiments.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/<your-org>/PTBenchmark.git
cd PTBenchmark
pip install -e .
```

Then install the required dependencies (including PixHomology):

```bash
ptbenchmark install
```

Initialize the linked supervised-method repositories before using NAFNet,
HIRDiff, or Restormer:

```bash
git submodule update --init --recursive
```

---

## Command Line Interface (CLI)

The CLI exposes several commands to manage dependencies, download datasets, run benchmarks, and merge results.

Invoke the tool with:

```bash
ptbenchmark --help
```

### `install`

Install all required Python dependencies and the PixHomology library.

```bash
ptbenchmark install
```

This command:

* Installs PixHomology (locally if available, otherwise from pip)
* Installs all required third-party libraries (NumPy, SciPy, PyTorch, Optuna, etc.)

---

### `download-dataset`

Download one or more datasets from Google Drive in HDF5 format.

```bash
ptbenchmark download-dataset [OPTIONS]
```

**Options**

* `--dataset`, `-d` *(str, default: ALL)*
  Name of the dataset to download. Use `ALL` to download all available datasets.

Downloaded files are saved in the `datasets/` directory as `.h5` files.

---

### Available Datasets

| Dataset       | Description                         | Download link                                                                                                                        |
| ------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| CBSD68        | Color BSD68 image denoising dataset | [https://drive.google.com/uc?id=1g3fiKmOLNiyS68__724Ycb_sMkOhzPrd](https://drive.google.com/uc?id=1g3fiKmOLNiyS68__724Ycb_sMkOhzPrd) |
| FMIDD_OH16230 | FMIDD dataset (OH16230)             | [https://drive.google.com/uc?id=1kOkdz6qv1N7o2QJmsM-syPy1QyzXsdsa](https://drive.google.com/uc?id=1kOkdz6qv1N7o2QJmsM-syPy1QyzXsdsa) |
| FMIDD_PVD     | FMIDD dataset (PVD)                 | [https://drive.google.com/uc?id=1XjWLEaRlVb-HgaEBRz4wM3gKGM-VvB3d](https://drive.google.com/uc?id=1XjWLEaRlVb-HgaEBRz4wM3gKGM-VvB3d) |
| FORECAST      | Forecasting dataset                 | [https://drive.google.com/uc?id=1pWlt_P6j5hp5YlLvp99yn94YhYuT345j](https://drive.google.com/uc?id=1pWlt_P6j5hp5YlLvp99yn94YhYuT345j) |
| MRI           | MRI image dataset                   | [https://drive.google.com/uc?id=1QSRVk-vX8byBIJDaBuAUrtx1nZq67MIf](https://drive.google.com/uc?id=1QSRVk-vX8byBIJDaBuAUrtx1nZq67MIf) |
| SEN2VENUS     | Sentinel-2 to VENµS dataset         | [https://drive.google.com/uc?id=1xN7F3rPyrOVzIqdXizjAmOaH9qfLoeR8](https://drive.google.com/uc?id=1xN7F3rPyrOVzIqdXizjAmOaH9qfLoeR8) |
| SIDD          | Smartphone Image Denoising Dataset  | [https://drive.google.com/uc?id=1E3rZuH6vnmyXk_1IxyB_8IJWeXuwSu9U](https://drive.google.com/uc?id=1E3rZuH6vnmyXk_1IxyB_8IJWeXuwSu9U) |

---

### `test`

Run benchmarking experiments on a selected dataset.

```bash
ptbenchmark test [OPTIONS]
```

**Required options**

* `--dataset`, `-d` *(str)*
  Dataset name (one of the available datasets).

**Optional options**

* `--mode`, `-m` *(str, default: method)*
  Test mode:

  * `method`: supervised or unsupervised denoising methods
  * `distance`: persistence tree distances (RD or RF)
  * `entropy`: spatial entropy change

* `--method` *(str, default: perstree)*
  Method name when `--mode method` is selected. Can be any supervised or unsupervised method implemented in the framework.
  Supervised methods include `unet`, `dncnn`, `nafnet`, `hirdiff`, and `restormer`.

* `--distance` *(str, default: RD)*
  Distance type for `--mode distance`. Supported values: `RD`, `RF`.

* `--device` *(str, default: cpu)*
  Device for supervised methods: `cpu` or `cuda`.

* `--epochs` *(int, default: 20)*
  Number of training epochs for supervised methods.

* `--datasets-dir` *(str, default: datasets)*
  Directory containing the downloaded datasets.

* `--output` *(str, default: results/results.h5)*
  Output HDF5 file where results are stored.

---

### `merge`

Merge multiple HDF5 result files into a single file.

```bash
ptbenchmark merge --input-dir DIR [--output FILE]
```

**Options**

* `--input-dir`, `-i` *(str, required)*
  Directory containing HDF5 files to merge.

* `--output`, `-o` *(str, default: results/results_all_merged.h5)*
  Output merged HDF5 file.

### `sensitivity`

Run a parameter sensitivity sweep for the PT denoising method or the entropy
stopping variant:

```bash
ptbenchmark sensitivity --dataset CBSD68 --method perstree
ptbenchmark sensitivity --dataset CBSD68 --method entropy
```

The sweep varies the base relaxation scale, guided-filter radius, epsilon
scale, and, for entropy stopping, the stopping threshold. Per-image metrics are
saved to CSV and HDF5.

To compare oracle MSE stopping (`perstree`) against entropy stopping
(`perstree_rec_sec`) after merging benchmark outputs:

```bash
python -m ptbenchmark.results.analyze_oracle_vs_entropy \
  --input results/results_all_merged.h5 \
  --output-dir paper/analysis
```

---

## Intended Use

PTBenchmark is intended for:

* Reproducible benchmarking of persistence-tree–based methods
* Comparison of distance measures and learning pipelines
* Research in Topological Data Analysis (TDA)

It is not optimized for production use.

---

## License and Citation

Please refer to the repository for licensing details. If you use PTBenchmark in academic work, please cite the corresponding publication or software repository.
