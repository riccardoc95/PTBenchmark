import h5py
import os
from pathlib import Path
import typer

def copy_group(src, dst):
    """Copia ricorsivamente gruppi e dataset senza cancellare quelli già esistenti."""
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            # Se il gruppo non esiste, crealo
            if key not in dst:
                dst_group = dst.create_group(key)
                # Copia anche gli attributi del gruppo
                for attr, val in item.attrs.items():
                    dst_group.attrs[attr] = val
            else:
                dst_group = dst[key]
            # Ricorsione per i sottogruppi
            copy_group(item, dst_group)

        elif isinstance(item, h5py.Dataset):
            # Se il dataset non esiste, copialo
            if key not in dst:
                src.copy(key, dst)
            else:
                print(f"Dataset già presente: {key}, saltato")

def merge_h5_files(input_dir, output_file):
    input_dir = Path(input_dir)
    files = sorted(list(input_dir.glob("*.h5")))

    if not files:
        typer.echo(f"No .h5 files found in {input_dir}")
        raise typer.Exit()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    typer.echo(f"Merging {len(files)} HDF5 files from {input_dir}")
    typer.echo(f"Output: {output_file}\n")

    with h5py.File(output_file, "a") as fout:
        for fpath in files:
            typer.echo(f"Merging file: {fpath.name}")
            try:
                with h5py.File(fpath, "r") as fin:
                    copy_group(fin, fout)
            except Exception as e:
                print(f"Skipping corrupted file {fpath}: {e}")
                continue

    typer.echo(f"\nMerge completed! Output saved to: {output_file}")