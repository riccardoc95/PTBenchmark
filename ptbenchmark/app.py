import typer
import h5py
import numpy as np
import os
import subprocess
import pandas as pd
from pathlib import Path
import gdown

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


# ==========================================================
# 1 INSTALL SOFTWARE LIBRARIES
# ==========================================================
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
                    "torchvision", "scikit-image", "bm3d", "optuna"]:
        typer.echo(f"Installing {library} from pip...")
        subprocess.run(["pip", "install", library], check=True)
        typer.echo(f"{library} installed successfully.")

# ==========================================================
# 2 DOWNLOAD DATASETS
# ==========================================================
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



# ==========================================================
# 3 TESTS AND PRINT STATISTICS
# ==========================================================
@app.command()
def test(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help=f"Dataset name. Available: {', '.join(DATASETS.keys())}",
    ),
):
    """
    Run a test
    """
    pass


if __name__ == "__main__":
    app()
