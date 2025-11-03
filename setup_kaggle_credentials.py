"""
Setup Kaggle Credentials and Download Datasets

Interactive script to:
1. Prompt for Kaggle username and API key
2. Save to .env file
3. Download required datasets
4. Run the demo
"""

import os
import sys
from pathlib import Path

# Add demo to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("KAGGLE CREDENTIALS SETUP")
print("=" * 80)
print("\nTo get your Kaggle API credentials:")
print("1. Go to https://www.kaggle.com/")
print("2. Sign in and go to Account settings")
print("3. Scroll to 'API' section")
print("4. Click 'Create New API Token'")
print("5. This downloads kaggle.json file")
print("6. Open kaggle.json and copy the username and key\n")

# Get credentials from user
print("Enter your Kaggle credentials:")
username = input("Kaggle Username: ").strip()
api_key = input("Kaggle API Key: ").strip()

if not username or not api_key:
    print("\nERROR: Username and API key are required!")
    sys.exit(1)

# Create .env file
env_path = Path('.env')
print(f"\nSaving credentials to {env_path}...")

with open(env_path, 'w') as f:
    f.write("# Kaggle API Credentials\n")
    f.write(f"KAGGLE_USERNAME={username}\n")
    f.write(f"KAGGLE_KEY={api_key}\n")

# Set permissions (Unix/Linux)
try:
    os.chmod(env_path, 0o600)
except:
    pass

print(f"Credentials saved successfully!\n")

# Ask if user wants to download datasets now
download = input("Download datasets now? (y/n): ").strip().lower()

if download == 'y':
    print("\n" + "=" * 80)
    print("DOWNLOADING DATASETS")
    print("=" * 80)
    
    try:
        from demo.kaggle_integration import KaggleDatasetIntegration
        
        kaggle = KaggleDatasetIntegration('data/kaggle')
        
        datasets = [
            ('saurav9786/real-time-advertisers-auction', 'Real-time Advertisers Auction'),
            ('karnikakapoor/video-ads-engagement-dataset', 'Video Ads Engagement')
        ]
        
        success_count = 0
        for dataset_name, dataset_display in datasets:
            print(f"\nDownloading: {dataset_display} ({dataset_name})...")
            try:
                success = kaggle.download_dataset(dataset_name, unzip=True)
                if success:
                    print(f"SUCCESS: {dataset_display} downloaded")
                    success_count += 1
                else:
                    print(f"FAILED: Could not download {dataset_display}")
            except Exception as e:
                print(f"ERROR: {e}")
        
        print(f"\n" + "=" * 80)
        print(f"Download Summary: {success_count}/{len(datasets)} datasets downloaded")
        print("=" * 80)
        
        if success_count > 0:
            print("\nYou can now run the demo:")
            print("  demo/.venv/bin/python3 demo/main.py")
    except Exception as e:
        print(f"\nERROR during download: {e}")
        print("You can download datasets manually from Kaggle website")
else:
    print("\nYou can download datasets later using:")
    print("  demo/.venv/bin/python3 demo/run_with_kaggle.py")
    print("\nOr run the demo if datasets are already downloaded:")

