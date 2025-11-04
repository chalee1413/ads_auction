"""
Script to download Kaggle datasets and run the demo

This script:
1. Sets up Kaggle credentials from .env
2. Downloads required datasets
3. Runs EDA on the datasets
4. Runs the main demo
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.config import setup_kaggle_credentials, get_kaggle_credentials
from demo.kaggle_integration import KaggleDatasetIntegration


def download_datasets():
    """Download all required datasets from Kaggle."""
    print("=" * 80)
    print("DOWNLOADING KAGGLE DATASETS")
    print("=" * 80)

    # Set up credentials
    print("\n1. Setting up Kaggle credentials...")
    if not setup_kaggle_credentials():
        creds = get_kaggle_credentials()
        if not creds:
            print("\nERROR: Kaggle credentials not found!")
            print("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
            print("See API_KEYS_SETUP.md for instructions")
            return False
    else:
        print("   Kaggle credentials set up successfully!")

    # Initialize Kaggle integration
    print("\n2. Initializing Kaggle API...")
    kaggle = KaggleDatasetIntegration()

    if kaggle.api is None:
        print("   ERROR: Could not authenticate with Kaggle API")
        return False

    # Download datasets
    datasets = [
        ("saurav9786/real-time-advertisers-auction", "Real-time Advertisers Auction"),
        ("karnikakapoor/video-ads-engagement-dataset", "Video Ads Engagement"),
    ]

    print("\n3. Downloading datasets...")
    success_count = 0

    for dataset_name, dataset_display in datasets:
        print(f"\n   Downloading: {dataset_display} ({dataset_name})")
        try:
            success = kaggle.download_dataset(dataset_name, unzip=True)
            if success:
                print(f"   SUCCESS: {dataset_display} downloaded")
                success_count += 1
            else:
                print(f"   FAILED: Could not download {dataset_display}")
        except Exception as e:
            print(f"   ERROR downloading {dataset_display}: {e}")

    print("\n" + "=" * 80)
    print(f"Download Summary: {success_count}/{len(datasets)} datasets downloaded successfully")
    print("=" * 80)

    return success_count > 0


def main():
    """Main function to download datasets and run demo."""
    print("\n" + "=" * 80)
    print("KAGGLE DATASETS DOWNLOAD AND DEMO RUNNER")
    print("=" * 80)

    # Download datasets
    if download_datasets():
        print("\n" + "=" * 80)
        print("Datasets downloaded successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run EDA: python3 demo/eda_kaggle_datasets.py")
        print("2. Run main demo: python3 -m demo.main")
    else:
        print("\n" + "=" * 80)
        print("Could not download datasets. Please check:")
        print("1. .env file has KAGGLE_USERNAME and KAGGLE_KEY set")
        print("2. Internet connection is available")
        print("3. Kaggle API credentials are valid")
        print("=" * 80)


if __name__ == "__main__":
    main()
