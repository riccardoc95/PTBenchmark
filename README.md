
# PHBenchmark — Persistent Homology Benchmark CLI

This repository provides a **Python CLI application** (built with [Typer](https://typer.tiangolo.com/)) for benchmarking persistent homology analysis software on various image datasets.
The app allows you to:

1. **Download datasets** (in `.h5` format) from Google Drive and export them as `.npy` files.
2. **Install local software libraries** from the `libs/` folder.
3. **Run automated tests** using a shell script that produces CSV files with evaluation metrics.

---

## Installation

Clone the repository and install the app in editable mode:

```bash
git clone https://github.com/<your-username>/PHBenchmark.git
cd PHBenchmark
pip install .
```

Now you can run the app using:

```bash
phbenchmark --help
```

---

## Commands Overview

### 1. Download datasets

Download a predefined dataset and extract it into `.npy` images:

```bash
phbenchmark download-dataset <dataset_name>
```

Available datasets and their Google Drive sources:

| Dataset      | Google Drive Link                                                                          |
| ------------ | ------------------------------------------------------------------------------------------ |
| **test**     | [link](https://drive.google.com/file/d/19GnE8Qw375kXTHmG_35GaXabvHVXSfUc/view?usp=sharing) |
| **mnist**    | [link](https://drive.google.com/file/d/1pG1jzW8qNh5PRWT8kC4XT_udWQNSFLFo/view?usp=sharing) |
| **cifar10**  | [link](https://drive.google.com/file/d/17lULKw6WX546SJxcTYl0qsLZLqyPO1Hk/view?usp=sharing) |
| **imagenet** | [link](https://drive.google.com/file/d/1BNl5VivoFJpevkH7b0sYuVnN39d830T9/view?usp=sharing) |
| **div2k**    | [link](https://drive.google.com/file/d/1XkeGHk9ObsDoChMSrunozZHyLeKGiJ5N/view?usp=sharing) |
| **kather**   | [link](https://drive.google.com/file/d/1Enb3EHrumPNcPwEUoe8Eon8wXAynONBD/view?usp=sharing) |

Each dataset is downloaded in `.h5` format (containing a dataset of images) and automatically extracted to `.npy` arrays in `datasets/<name>_npy/`.

Example:

```bash
phbenchmark download-dataset mnist
```

---

### 2. Install local software libraries

Install one or all software packages from the local `libs/` directory:

```bash
phbenchmark install <software_name>
# or
phbenchmark install all
```

Predefined software packages:

| Software     | Path                              |
| ------------ | --------------------------------- |
| **pixh**     | `libs/PixHomology/`               |
| **cripser**  | `libs/CubicalRipser_3dim-0.0.21/` |
| **gudhi**    | `libs/gudhi-devel/`               |
| **ripserpy** | `libs/ripser.py/`                 |


Example:

```bash
phbenchmark install pixh
```

---

### 3. Run software tests and compute statistics

Run tests on a dataset for a given method (software) and maximum dimension.
The script `script.sh` (included in the repository) is executed automatically and produces a CSV file with results.

```bash
phbenchmark test <software_name> <dataset_name> <maxdim>
```

Example:

```bash
phbenchmark test pixh mnist 1
```

This command will:

* Run the shell script that benchmarks the software,
* Save the results to `results/<dataset>_<software>_<maxdim>_results.csv`,
* Compute the **mean** and **standard deviation** for all metrics in the CSV.


---

## Repository Structure

```
PHBenchmark/
│
├── phbenchmark/           # Main Python package
│   ├── app.py             # CLI application (Typer)
│   └── script.sh          # Script used for testing
│
├── datasets/              # Datasets (.h5 and .npy)
├── results/               # Output CSV results
├── libs/                  # Local software libraries
│   ├── PixHomology/
│   ├── CubicalRipser_3dim-0.0.21/
│   ├── gudhi-devel/
│   └── ripser.py/
└── README.md
```

---

## Tested Software

The repository includes and supports benchmarking of several persistent homology and cubical complex libraries:

* [PixHomology](https://github.com/riccardoc95/PixHomology)
* [CubicalRipser_3dim](https://github.com/shizuo-kaji/CubicalRipser_3dim)
* [Gudhi](https://github.com/GUDHI/gudhi-devel)
* [Ripser.py](https://github.com/scikit-tda/ripser.py)

Each tool can be found under the `libs/` folder and is automatically installed in editable mode during setup.

---


## Example Workflow

```bash
# 1. Download a dataset
phbenchmark download-dataset cifar10

# 2. Install all local libraries
phbenchmark install all

# 3. Run a benchmark test
phbenchmark test gudhi cifar10 1
```
