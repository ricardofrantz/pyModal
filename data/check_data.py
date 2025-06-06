#!/usr/bin/env python3
"""Quick dataset check by plotting representative snapshots."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import CMAP_SEQ
from data_interface import load_data


def list_data_files(directory: str) -> list[str]:
    """Return list of data files in *directory*."""
    exts = {".mat", ".h5", ".hdf5", ".cgns"}
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
            files.append(path)
    return sorted(files)


def plot_snapshots(data: dict, file_path: str) -> None:
    """Plot four snapshots and save image next to the data file."""
    q = data["q"]
    ns = q.shape[0]
    nx = data.get("Nx", 1)
    ny = data.get("Ny", 1)
    nz = data.get("Nz", 1)
    x = data["x"]
    y = data["y"]

    # Use first, last, and two middle snapshots (unique, sorted)
    indices = sorted(set([0, ns // 3, ns // 2, ns - 1]))

    is_2d = nx > 1 and ny > 1 and q.shape[1] == nx * ny * nz

    nplots = len(indices)
    # Improve aspect ratio for 2D plots
    if is_2d:
        aspect = nx / ny if ny > 0 else 1
        aspect = max(0.5, min(aspect, 2.0))  # Clamp to reasonable bounds
        plt.figure(figsize=(3 * nplots * aspect, 3))
    else:
        plt.figure(figsize=(3 * nplots, 3))
    var_name = data.get('metadata', {}).get('var_name', 'q')
    for i, idx in enumerate(indices):
        plt.subplot(1, nplots, i + 1)
        snap = q[idx]
        if is_2d:
            snap = snap.reshape(nx, ny).T
            use_extent = (
                x.ndim == 1 and y.ndim == 1 and
                np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
            )
            if use_extent:
                extent = (x.min(), x.max(), y.min(), y.max())
                plt.imshow(snap, origin="lower", extent=extent, cmap=CMAP_SEQ, aspect="auto")
                plt.xlabel("x")
                plt.ylabel("y")
            else:
                plt.imshow(snap, origin="lower", cmap=CMAP_SEQ, aspect="auto")
                plt.xlabel("x index")
                plt.ylabel("y index")
        else:
            plt.plot(snap)
            plt.xlabel("index")
            plt.ylabel("value")
        plt.title(f"Ns={idx} [{var_name}]")
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
        try:
            data = load_data(file_path)
        except Exception as exc:
            print(f"Failed to load {file_path}: {exc}")
            continue
        plot_snapshots(data, file_path)


if __name__ == "__main__":
    main()
