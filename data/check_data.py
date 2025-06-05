#!/usr/bin/env python3
"""Quick dataset check by plotting representative snapshots."""

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import CMAP_SEQ
from data_interface import data_manager


def list_data_files(directory: str) -> list[str]:
    """Return list of files in *directory* using ``grep``."""
    cmd = f"grep -al '' {os.path.join(directory, '*')}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    exts = {".mat", ".h5", ".hdf5", ".cgns"}
    return [f for f in files if os.path.splitext(f)[1].lower() in exts]


def get_loader(file_path: str):
    """Return loader instance capable of handling *file_path* or ``None``."""
    for loader in data_manager.loaders:
        if loader.supports_format(file_path):
            return loader
    return None


def plot_snapshots(data: dict, file_path: str) -> None:
    """Plot four snapshots and save image next to the data file."""
    q = data["q"]
    ns = q.shape[0]
    nx = data.get("Nx", 1)
    ny = data.get("Ny", 1)
    nz = data.get("Nz", 1)
    x = data.get("x", np.arange(nx))
    y = data.get("y", np.arange(ny))

    indices = [0, ns // 4, ns // 2, (3 * ns) // 4]

    is_2d = nx > 1 and ny > 1 and q.shape[1] == nx * ny * nz

    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, 4, i + 1)
        snap = q[idx]
        if is_2d:
            snap = snap.reshape(nx, ny).T
            extent = (
                (
                    x.min(),
                    x.max(),
                    y.min(),
                    y.max(),
                )
                if x.ndim == 1 and y.ndim == 1
                else None
            )
            plt.imshow(snap, origin="lower", extent=extent, cmap=CMAP_SEQ, aspect="auto")
            plt.xlabel("x")
            plt.ylabel("y")
        else:
            plt.plot(snap)
            plt.xlabel("index")
            plt.ylabel("value")
        plt.title(f"Snapshot {idx}")
    plt.tight_layout()

    root = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(os.path.dirname(file_path), f"{root}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    files = list_data_files("data")
    if not files:
        print("No data files found.")
        return

    for file_path in files:
        loader = get_loader(file_path)
        if loader is None:
            print(f"No loader available for {file_path}")
            continue
        data = loader.load(file_path)
        plot_snapshots(data, file_path)


if __name__ == "__main__":
    main()
