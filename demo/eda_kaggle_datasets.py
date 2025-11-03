"""
EDA: Kaggle Datasets for Incrementality Measurement

This script performs comprehensive Exploratory Data Analysis (EDA) on two Kaggle datasets:

1. Real-time Advertisers Auction Dataset: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction
2. Video Ads Engagement Dataset: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset

Run with: python3 demo/eda_kaggle_datasets.py
Or convert to Jupyter notebook by changing .py to .ipynb
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path('data/kaggle')
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path('data/eda_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("EDA: KAGGLE DATASETS FOR INCREMENTALITY MEASUREMENT")
print("=" * 80)
print("\nLibraries imported successfully\n")


def analyze_dataset_structure(df, dataset_name):
    """Analyze basic structure of a dataset."""
    print(f"=" * 80)
    print(f"{dataset_name.upper()} DATASET OVERVIEW")
    print(f"=" * 80)
    
    print(f"\n1. Basic Information:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n2. Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\n3. Data Types:")
    print(df.dtypes.to_string())
    
    print(f"\n4. First Few Rows:")
    print(df.head(10).to_string())
    
    print(f"\n5. Statistical Summary:")
    print(df.describe().to_string())
    
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }


def analyze_data_quality(df, dataset_name):
    """Analyze data quality issues."""
    print(f"\n" + "=" * 80)
    print(f"DATA QUALITY ANALYSIS - {dataset_name.upper()}")
    print("=" * 80)
    
    print(f"\n1. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("   No missing values found!")
    
    print(f"\n2. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    print(f"\n3. Unique Values per Column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 50:
            print(f"   {col}: {unique_count} unique values")
        else:
            print(f"   {col}: {unique_count} unique values (high cardinality)")
    
    return {
        'missing_values': missing_df.to_dict('records') if len(missing_df) > 0 else [],
        'duplicates': duplicates,
        'duplicate_pct': duplicates/len(df)*100
    }


def identify_metrics(df, dataset_name):
    """Identify key metrics for incrementality measurement."""
    print(f"\n" + "=" * 80)
    print(f"KEY METRICS FOR INCREMENTALITY MEASUREMENT - {dataset_name.upper()}")
    print("=" * 80)
    
    metrics = {
        'Bid-related': [col for col in df.columns if 'bid' in col.lower()],
        'Auction-related': [col for col in df.columns if 'auction' in col.lower() or 'impression' in col.lower()],
        'Click/Conversion': [col for col in df.columns if 'click' in col.lower() or 'conversion' in col.lower() or 'conversion' in col.lower()],
        'Revenue': [col for col in df.columns if 'revenue' in col.lower() or 'value' in col.lower() or 'price' in col.lower()],
        'Time-related': [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower()],
        'User/Session': [col for col in df.columns if 'user' in col.lower() or 'session' in col.lower() or 'id' in col.lower()],
        'Engagement': [col for col in df.columns if 'engagement' in col.lower()],
        'Views': [col for col in df.columns if 'view' in col.lower() or 'watch' in col.lower()],
        'Video/Ad Info': [col for col in df.columns if 'ad' in col.lower() or 'video' in col.lower() or 'campaign' in col.lower()]
    }
    
    identified_metrics = {}
    for category, cols in metrics.items():
        if cols:
            print(f"\n{category}:")
            identified_metrics[category] = []
            for col in cols:
                print(f"   - {col}")
                identified_metrics[category].append(col)
                if df[col].dtype in ['int64', 'float64']:
                    print(f"     Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
                    print(f"     Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
    
    return identified_metrics


def create_visualizations(df, dataset_name, iteration=1):
    """Create visualizations for the dataset."""
    print(f"\n" + "=" * 80)
    print(f"VISUALIZATION - {dataset_name.upper()} (Iteration {iteration})")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Plot distributions
        n_cols = min(6, len(numeric_cols))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{dataset_name.lower().replace(" ", "_")}_distributions_iter{iteration}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved distribution plots to {OUTPUT_DIR}")
    
    # Correlation analysis if multiple numeric columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Matrix - {dataset_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{dataset_name.lower().replace(" ", "_")}_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Find strong correlations
        print("\nStrong Correlations (|r| > 0.7):")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
        
        if strong_corr:
            for col1, col2, corr in strong_corr:
                print(f"   {col1} <-> {col2}: {corr:.3f}")
        else:
            print("   No strong correlations found")


def main():
    """Main EDA function."""
    # Part 1: Real-time Auction Dataset
    print("\n" + "=" * 80)
    print("PART 1: REAL-TIME ADVERTISERS AUCTION DATASET ANALYSIS")
    print("=" * 80)
    
    auction_files = list(DATA_DIR.glob('Dataset.csv')) + list(DATA_DIR.glob('*auction*.csv'))
    
    if auction_files:
        print(f"\nFound Real-time Auction files: {[f.name for f in auction_files]}")
        auction_df = pd.read_csv(auction_files[0])
        print(f"Loaded Real-time Auction dataset: {auction_df.shape[0]} rows, {auction_df.shape[1]} columns")
        
        auction_structure = analyze_dataset_structure(auction_df, "Real-time Auction")
        auction_quality = analyze_data_quality(auction_df, "Real-time Auction")
        auction_metrics = identify_metrics(auction_df, "Real-time Auction")
        create_visualizations(auction_df, "Real-time Auction", iteration=1)
    else:
        print("\nReal-time Auction dataset not found. Please download from:")
        print("https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction")
        auction_df = None
        auction_structure = auction_quality = auction_metrics = None
    
    # Part 2: Video Ads Dataset
    print("\n\n" + "=" * 80)
    print("PART 2: VIDEO ADS ENGAGEMENT DATASET ANALYSIS")
    print("=" * 80)
    
    video_files = list(DATA_DIR.glob('video*.csv')) + list(DATA_DIR.glob('*video*.csv')) + \
                 list(DATA_DIR.glob('engagement*.csv')) + list(DATA_DIR.glob('*engagement*.csv'))
    
    if video_files:
        print(f"\nFound video ads files: {[f.name for f in video_files]}")
        video_df = pd.read_csv(video_files[0])
        print(f"Loaded Video Ads Engagement dataset: {video_df.shape[0]} rows, {video_df.shape[1]} columns")
        
        video_structure = analyze_dataset_structure(video_df, "Video Ads")
        video_quality = analyze_data_quality(video_df, "Video Ads")
        video_metrics = identify_metrics(video_df, "Video Ads")
        create_visualizations(video_df, "Video Ads", iteration=2)
    else:
        print("\nVideo Ads Engagement dataset not found. Please download from:")
        print("https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset")
        video_df = None
        video_structure = video_quality = video_metrics = None
    
    # Part 3: Combined Analysis
    print("\n\n" + "=" * 80)
    print("PART 3: COMBINED ANALYSIS AND INTEGRATION PREPARATION")
    print("=" * 80)
    
    if auction_df is not None and video_df is not None:
        print("\n1. Dataset Comparison:")
        comparison = pd.DataFrame({
            'Metric': ['Rows', 'Columns', 'Memory (MB)'],
            'Real-time Auction Dataset': [
                len(auction_df),
                len(auction_df.columns),
                f"{auction_df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            ],
            'Video Ads Dataset': [
                len(video_df),
                len(video_df.columns),
                f"{video_df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            ]
        })
        print(comparison.to_string(index=False))
        
        print("\n2. Common Columns for Integration:")
        common_cols = set(auction_df.columns) & set(video_df.columns)
        if common_cols:
            print(f"   Found {len(common_cols)} common columns: {list(common_cols)}")
        else:
            print("   No direct common columns found. Integration will use domain mapping.")
        
        print("\n3. Integration Strategy:")
        print("   - Real-time Auction dataset: Auction-level data with revenue for iROAS calculations")
        print("   - Video Ads dataset: Engagement-level data for view and interaction analysis")
        print("   - Integration point: Time-based or user-based joining")
        print("   - Use cases: Combined incrementality measurement across auction revenue and engagement metrics")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("EDA SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Dataset structures and characteristics analyzed")
    print("2. Data quality issues identified and documented")
    print("3. Key metrics for incrementality measurement identified")
    print("4. Relationships and patterns explored")
    print("5. Integration strategy defined")
    print("\nNext Steps:")
    print("1. Clean and preprocess data based on findings")
    print("2. Create feature engineering pipeline")
    print("3. Integrate datasets where possible")
    print("4. Implement incrementality measurement algorithms")
    print("5. Calculate iROAS and other metrics")
    print("6. Validate results and iterate")
    print(f"\nEDA outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

