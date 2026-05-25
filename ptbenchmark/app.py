import typer
import os
import subprocess
import h5py
import gdown
from pathlib import Path

from ptbenchmark.tests.test_method import test_method, supervised_methods, unsupervised_methods
from ptbenchmark.tests.test_distance import test_pt_distance
from ptbenchmark.tests.test_sec import test_sec
from ptbenchmark.utils import merge_h5_files

app = typer.Typer()
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Predefined lists ---
DATASETS = {
    "CBSD68": "1g3fiKmOLNiyS68__724Ycb_sMkOhzPrd",
    "FMIDD_OH16230": "1kOkdz6qv1N7o2QJmsM-syPy1QyzXsdsa",
    "FMIDD_PVD": "1XjWLEaRlVb-HgaEBRz4wM3gKGM-VvB3d",
    "FORECAST": "1pWlt_P6j5hp5YlLvp99yn94YhYuT345j",
    "MRI": "1QSRVk-vX8byBIJDaBuAUrtx1nZq67MIf",
    "SEN2VENUS": "1xN7F3rPyrOVzIqdXizjAmOaH9qfLoeR8",
    "SIDD": "1E3rZuH6vnmyXk_1IxyB_8IJWeXuwSu9U",
}

@app.command()
def install():
    """
    Install local Python packages (software tools).
    """
    path = BASE_DIR / "libs" / "PixHomology/"
    try:
        typer.echo(f"Installing PixHomology from {path}...")
        subprocess.run(["pip", "install", path], check=True)
    except subprocess.CalledProcessError:
        typer.echo(f"Installing PixHomology from pip...")
        subprocess.run(["pip", "install", "pixhomology"], check=True)
        typer.echo("PixHomology installed successfully.")

    for library in ["numpy", "scipy", "h5py", "gdown", "torch", "astropy",
                    "torchvision", "scikit-image", "bm3d", "optuna", "einops"]:
        typer.echo(f"Installing {library} from pip...")
        subprocess.run(["pip", "install", library], check=True)
        typer.echo(f"{library} installed successfully.")

@app.command()
def download_dataset(
    dataset: str = typer.Option(
        "ALL",
        "--dataset",
        "-d",
        help=f"Dataset name to download. Available: ALL, {', '.join(DATASETS.keys())}",
    )
):
    """
    Download a dataset (HDF5) from Google Drive and export images as .npy files.
    """
    if dataset not in DATASETS and dataset != "ALL":
        typer.echo(f"Dataset '{dataset}' not found. Available: {list(DATASETS.keys())}")
        raise typer.Exit()

    os.makedirs("datasets", exist_ok=True)

    if dataset == "ALL":
        for dataset in DATASETS.keys():
            typer.echo(f"Downloading dataset '{dataset}'...")
            url = DATASETS[dataset]
            h5_path = f"datasets/{dataset}.h5"

            # Download from Google Drive
            gdown.download(id=url, output=h5_path, quiet=False)
            typer.echo(f"Done!")
    else:
        typer.echo(f"Downloading dataset '{dataset}'...")
        url = DATASETS[dataset]
        h5_path = f"datasets/{dataset}.h5"

        # Download from Google Drive
        gdown.download(id=url, output=h5_path, quiet=False)
        typer.echo(f"Done!")

@app.command()
def test(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help=f"Dataset name. Available: {', '.join(DATASETS.keys())}",
    ),
    mode: str = typer.Option(
        "method",
        "--mode",
        "-m",
        help="Test mode: 'method' (supervised/unsupervised), 'distance' (RD/RF), or 'entropy' (spatial entropy change).",
    ),
    method: str = typer.Option(
        "perstree",
        "--method",
        help=f"Method name for --mode method ({', '.join(list(unsupervised_methods.keys()))}, "
             f"{', '.join(list(supervised_methods.keys()))}).",
    ),
    distance: str = typer.Option(
        "RD",
        "--distance",
        help="Distance type for --mode distance (RD or RF).",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device for supervised methods (cpu or cuda).",
    ),
    n_epochs: int = typer.Option(
        20,
        "--epochs",
        help="Number of training epochs (for supervised).",
    ),
    datasets_dir: str = typer.Option(
        "datasets",
        "--datasets-dir",
        help="Input dataset directory.",
    ),
    output_file: str = typer.Option(
        "results/results.h5",
        "--output",
        help="Output .h5 file for results.",
    )
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    typer.echo(f"\nRunning test for dataset: {dataset}")
    typer.echo(f"Mode: {mode}")
    typer.echo(f"Output file: {output_file}\n")

    if mode == "method":
        typer.echo(f"Running denoising method: {method}")
        test_method(method, dataset, datasets_dir, output_file, device=device, n_epochs=n_epochs)

    elif mode == "distance":
        typer.echo(f"Running persistence distance: {distance}")
        test_pt_distance(datasets_dir, dataset, output_file, distance)

    elif mode == "entropy":
        typer.echo(f"Running spatial entropy change (perstree_rec_sec)")
        test_sec(datasets_dir, dataset, output_file)

    else:
        typer.echo(f"Unknown mode '{mode}'. Must be one of: method, distance, entropy.")
        raise typer.Exit()

    typer.echo("\nTest completed successfully!\n")


@app.command()
def merge(
    input_dir: str = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="Directory containing HDF5 files to merge.",
    ),
    output_file: str = typer.Option(
        "results/results_all_merged.h5",
        "--output",
        "-o",
        help="Output merged HDF5 file.",
    ),
):
    merge_h5_files(input_dir, output_file)

if __name__ == "__main__":
    app()
