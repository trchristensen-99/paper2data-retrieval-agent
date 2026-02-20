"""
Script to download MPRA datasets from Zenodo.

Usage:
    python scripts/download_data.py --dataset k562
    python scripts/download_data.py --dataset yeast
    python scripts/download_data.py --dataset all
"""

import argparse
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from typing import Optional
from tqdm import tqdm


# Zenodo dataset information
DATASETS = {
    'k562': {
        'zenodo_id': '10698014',
        'files': [
            {
                'name': 'DATA-Table_S2__MPRA_dataset.txt',
                'url': 'https://zenodo.org/records/10698014/files/DATA-Table_S2__MPRA_dataset.txt',
                'size': '280.2 MB',
                'expected_sequences': '~370K'
            },
        ],
        'description': 'K562 human MPRA dataset from Gosai et al., Nature 2023 (lentiMPRA)',
        'total_sequences': '~370K'
    },
    'yeast': {
        'zenodo_id': '10633252',
        'files': [
            {
                'name': 'train.txt',
                'url': 'https://zenodo.org/records/10633252/files/train.txt',
                'size': '~400 MB',
                'expected_sequences': '~5.4M'
            },
            {
                'name': 'val.txt',
                'url': 'https://zenodo.org/records/10633252/files/val.txt',
                'size': '~70 MB',
                'expected_sequences': '~0.9M'
            },
            {
                'name': 'test.txt',
                'url': 'https://zenodo.org/records/10633252/files/test.txt',
                'size': '~30 MB',
                'expected_sequences': '~0.4M'
            },
        ],
        'description': 'Yeast promoter MPRA dataset from de Boer et al., Nature Biotech 2024 (Random Promoter DREAM Challenge)',
        'total_sequences': '~6.7M'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: Optional[str] = None) -> None:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"File already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download")
            return
    
    print(f"Downloading: {desc or output_path.name}")
    print(f"URL: {url}")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        print(f"✓ Downloaded to {output_path}")
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """
    Extract a zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        output_dir: Directory to extract to
    """
    print(f"Extracting {archive_path.name}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        print(f"Unknown archive format: {archive_path.suffix}")
        return
    
    print(f"✓ Extracted to {output_dir}")


def download_dataset(dataset_name: str, data_root: Path) -> None:
    """
    Download a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('k562' or 'yeast')
        data_root: Root directory for data storage
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Description: {dataset_info['description']}")
    print(f"Expected sequences: {dataset_info.get('total_sequences', 'Unknown')}")
    print(f"Zenodo: https://zenodo.org/records/{dataset_info['zenodo_id']}")
    print(f"{'='*60}\n")
    
    # Create dataset directory
    dataset_dir = data_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each file
    downloaded_files = []
    for file_info in dataset_info['files']:
        output_path = dataset_dir / file_info['name']
        desc = f"{file_info['name']} ({file_info['size']})"
        if 'expected_sequences' in file_info:
            desc += f" - {file_info['expected_sequences']} sequences"
        
        try:
            download_file(file_info['url'], output_path, desc)
            downloaded_files.append(output_path)
            
            # Extract if archive
            if output_path.suffix in ['.zip', '.tar', '.gz', '.tgz']:
                extract_archive(output_path, dataset_dir)
        
        except Exception as e:
            print(f"Failed to download {file_info['name']}: {e}")
            print("\nNote: You may need to manually download files from Zenodo.")
            print(f"Visit: https://zenodo.org/records/{dataset_info['zenodo_id']}")
            continue
    
    # Validate downloads
    print(f"\n{'='*60}")
    print(f"Download Summary for {dataset_name}:")
    print(f"{'='*60}")
    for file_path in downloaded_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_path.name}: {size_mb:.1f} MB")
        else:
            print(f"✗ {file_path.name}: Not found")
    
    print(f"\n✓ {dataset_name} dataset ready at: {dataset_dir}")
    print(f"\nNext steps:")
    print(f"  1. Verify file sizes match expected values")
    print(f"  2. Run: python -c \"from data.{dataset_name} import {dataset_name.capitalize()}Dataset; d = {dataset_name.capitalize()}Dataset('data/{dataset_name}', 'train'); print(f'Loaded {{len(d)}} sequences')\"")
    print(f"  3. Expected: {dataset_info.get('total_sequences', 'See paper')}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MPRA datasets from Zenodo"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['k562', 'yeast', 'all'],
        help='Which dataset to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Root directory for data storage (default: ./data)'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    
    print("MPRA Dataset Downloader")
    print(f"Data will be saved to: {data_root.absolute()}")
    
    if args.dataset == 'all':
        for dataset_name in ['k562', 'yeast']:
            try:
                download_dataset(dataset_name, data_root)
            except Exception as e:
                print(f"Error downloading {dataset_name}: {e}")
                continue
    else:
        download_dataset(args.dataset, data_root)
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify data files are in the correct format")
    print("2. Run tests: pytest tests/test_data.py")
    print("3. Start training: python experiments/01_baseline_subsets.py")


if __name__ == '__main__':
    main()
