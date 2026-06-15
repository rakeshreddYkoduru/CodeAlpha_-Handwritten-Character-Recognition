"""
Download EMNIST dataset for training.

Run this script after cloning the repository to download the required
EMNIST Balanced dataset. The data files are too large for GitHub (>25MB)
so they are excluded from the repository via .gitignore.

Usage:
    python download_data.py
"""

import os
import sys


def download_emnist():
    """Download EMNIST Balanced dataset using torchvision."""
    try:
        from torchvision import datasets
    except ImportError:
        print("ERROR: PyTorch and torchvision are required.")
        print("Install them first: pip install -r requirements.txt")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("  EMNIST Dataset Downloader")
    print("=" * 60)
    print()
    print(f"Download directory: {os.path.abspath(data_dir)}")
    print("Split: balanced (47 classes — digits + uppercase + lowercase)")
    print()

    # Check if data already exists
    raw_dir = os.path.join(data_dir, "EMNIST", "raw")
    expected_file = os.path.join(raw_dir, "emnist-balanced-train-images-idx3-ubyte")
    if os.path.exists(expected_file):
        print("✓ EMNIST Balanced dataset already exists. Skipping download.")
        print(f"  Location: {raw_dir}")
        return

    print("Downloading EMNIST Balanced dataset...")
    print("(This may take a few minutes depending on your connection)")
    print()

    try:
        datasets.EMNIST(root=data_dir, split="balanced", train=True, download=True)
        datasets.EMNIST(root=data_dir, split="balanced", train=False, download=True)
        print()
        print("=" * 60)
        print("  ✓ Download complete!")
        print("=" * 60)
        print()
        print("You can now train the model:")
        print("  python src/train.py")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try again — the server may be temporarily unavailable")
        print("  3. Manually download from: https://www.nist.gov/itl/products-and-services/emnist-dataset")
        sys.exit(1)


if __name__ == "__main__":
    download_emnist()
