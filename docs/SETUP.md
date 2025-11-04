# Setup and Testing Guide

## Setup Steps

1. **Create Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

2. **Install Dependencies**
```bash
pip install -r demo/requirements.txt
```

3. **Set Up Kaggle API (Optional - for dataset downloads)**
```bash
# Install kaggle package
pip install kaggle

# Set up credentials
# 1. Download kaggle.json from your Kaggle account
# 2. Place it in ~/.kaggle/kaggle.json
# 3. Set permissions: chmod 600 ~/.kaggle/kaggle.json
```

## Testing

### Run Main Demo
```bash
# From project root:
python3 -m demo.main

# Or from demo directory:
cd demo && python3 main.py
```

### Run EDA Script
```bash
# First download datasets from Kaggle:
# - https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction
# - https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset

# Place CSV files in data/kaggle/ directory

# Then run EDA:
python3 demo/eda_kaggle_datasets.py
```

### Test Individual Modules
```bash
# Test incrementality calculations
python3 -c "import sys; sys.path.insert(0, 'demo'); from incrementality import calculate_iroas; print('OK')"

# Test CUPED
python3 -c "import sys; sys.path.insert(0, 'demo'); from cuped import cuped_adjustment; print('OK')"
```

## Dataset Integration

The demo supports integration with two Kaggle datasets:

1. **Real-time Advertisers Auction Dataset** (saurav9786/real-time-advertisers-auction)
   - Real-time advertiser auction data with actual revenue
   - Contains revenue (`total_revenue`), impressions (`total_impressions`), date, and feature columns
   - Use: "integrate_kaggle_with_incrementality(dataset_type='auction', dataset_name='saurav9786/real-time-advertisers-auction')"
   - Note: Dataset contains actual revenue data enabling direct iROAS calculations without synthetic data.

2. **Video Ads Engagement** (karnikakapoor/video-ads-engagement-dataset)
   - Video advertisement engagement metrics
   - Use: "integrate_kaggle_with_incrementality(dataset_type='video_ads', dataset_name='karnikakapoor/video-ads-engagement-dataset')"

## Files Structure

- "demo/main.py": Main demo application
- "demo/eda_kaggle_datasets.py": EDA script for Kaggle datasets
- "demo/kaggle_integration.py": Kaggle dataset integration module
- All algorithm modules in "demo/" directory

