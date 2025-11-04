"""
Kaggle Dataset Integration

Integration with Kaggle datasets for incrementality measurement.
Primary datasets:
- Real-time Advertisers Auction: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction
- Video Ads Engagement: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset

This module provides functionality to download and process Kaggle datasets
for use with incrementality measurement algorithms.

Usage:
1. Install kaggle package: pip install kaggle
2. Set up Kaggle API credentials: ~/.kaggle/kaggle.json
3. Use download_kaggle_dataset() to fetch the data
4. Process data with process_auction_data()
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from demo.config import get_kaggle_credentials, setup_kaggle_credentials, load_env
except ImportError:
    try:
        from config import get_kaggle_credentials, setup_kaggle_credentials, load_env
    except ImportError:

        def get_kaggle_credentials():
            return {}

        def setup_kaggle_credentials():
            return False

        def load_env():
            pass


try:
    from kaggle.api.kaggle_api_extended import KaggleApi

    KAGGLE_AVAILABLE = True
except (ImportError, IOError, OSError):
    KAGGLE_AVAILABLE = False
    # Don't print warning here - will be handled in class initialization


class KaggleDatasetIntegration:
    """
    Integration with Kaggle datasets for incrementality measurement.

    Supported datasets:
    1. Real-time Advertisers Auction
       URL: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction

    2. Video Ads Engagement
       URL: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset
    """

    def __init__(self, dataset_path: str = "data/kaggle"):
        """
        Initialize Kaggle dataset integration.

        Args:
            dataset_path: Path to store downloaded dataset
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.api = None

        if KAGGLE_AVAILABLE:
            # Set up credentials from .env if available
            load_env()
            creds = get_kaggle_credentials()
            if creds and creds.get("username") and creds.get("key"):
                # Set environment variables for Kaggle API
                os.environ["KAGGLE_USERNAME"] = creds["username"]
                os.environ["KAGGLE_KEY"] = creds["key"]

            # Try to initialize API (don't fail if credentials not available)
            try:
                self.api = KaggleApi()
                self.api.authenticate()
            except (IOError, OSError) as e:
                # Credentials not found - this is OK if datasets already exist locally
                print(f"   WARNING: Could not find Kaggle credentials file: {e}")
                self.api = None
            except Exception as e:
                # Log the actual error to help diagnose the issue
                print(f"   WARNING: Initial authentication attempt failed: {type(e).__name__}: {e}")
                # Try setting up from .env
                load_env()
                creds = get_kaggle_credentials()
                if creds and creds.get("username") and creds.get("key"):
                    os.environ["KAGGLE_USERNAME"] = creds["username"]
                    os.environ["KAGGLE_KEY"] = creds["key"]
                    try:
                        self.api = KaggleApi()
                        self.api.authenticate()
                        print("   INFO: Authentication successful after retry")
                    except Exception as e2:
                        print(
                            f"   ERROR: Authentication failed after retry: {type(e2).__name__}: {e2}"
                        )
                        self.api = None
                else:
                    print("   ERROR: No credentials found in environment variables")
                    self.api = None

    def download_dataset(
        self, dataset_name: str = "saurav9786/real-time-advertisers-auction", unzip: bool = True
    ) -> bool:
        """
        Download dataset from Kaggle.

        Supported datasets:
        - 'saurav9786/real-time-advertisers-auction': Real-time Advertisers Auction - Used in demos (has revenue data)
        - 'karnikakapoor/video-ads-engagement-dataset': Video Ads Engagement - Used in demos

        Args:
            dataset_name: Kaggle dataset name (user/dataset)
            unzip: Whether to unzip downloaded files

        Returns:
            True if successful, False otherwise
        """
        if not KAGGLE_AVAILABLE or self.api is None:
            print("Kaggle API not available. Install with: pip install kaggle")
            return False

        try:
            self.api.dataset_download_files(dataset_name, path=str(self.dataset_path), unzip=unzip)
            print(f"Dataset downloaded successfully to {self.dataset_path}")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False

    def load_dataset(self, dataset_type: str = "auto") -> pd.DataFrame:
        """
        Load dataset from downloaded files.

        Args:
            dataset_type: Type of dataset ('auto', 'rtb', 'video_ads', 'auction')
                'auto' tries to detect based on available files

        Returns:
            DataFrame with dataset data
        """
        # Try to find dataset files
        possible_files = list(self.dataset_path.glob("*.csv"))

        if not possible_files:
            raise FileNotFoundError(
                "No CSV files found. Please download the dataset first using download_dataset()"
            )

        # Auto-detect dataset type
        if dataset_type == "auto":
            # Check for auction files (Dataset.csv - has revenue data)
            auction_files = [
                f
                for f in possible_files
                if "dataset.csv" in f.name.lower() or "auction" in f.name.lower()
            ]
            if auction_files:
                return pd.read_csv(auction_files[0])

            # Check for video ads files
            video_files = [
                f
                for f in possible_files
                if "video" in f.name.lower()
                or "engagement" in f.name.lower()
                or "ad_df.csv" in f.name.lower()
            ]
            if video_files:
                return pd.read_csv(video_files[0])

            # Use first available file
            return pd.read_csv(possible_files[0])

        # Load specific dataset type
        elif dataset_type == "auction":
            auction_files = [
                f
                for f in possible_files
                if "dataset.csv" in f.name.lower() or "auction" in f.name.lower()
            ]
            if auction_files:
                return pd.read_csv(auction_files[0])
            raise FileNotFoundError("Real-time Auction dataset file (Dataset.csv) not found")

        elif dataset_type == "video_ads":
            video_files = [
                f
                for f in possible_files
                if "video" in f.name.lower() or "engagement" in f.name.lower()
            ]
            if video_files:
                return pd.read_csv(video_files[0])
            raise FileNotFoundError("Video Ads dataset file not found")

        else:
            return pd.read_csv(possible_files[0])

    def load_auction_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load auction data from downloaded CSV file.

        Args:
            filename: Name of the CSV file

        Returns:
            DataFrame with auction data
        """
        if filename:
            file_path = self.dataset_path / filename
            if file_path.exists():
                return pd.read_csv(file_path)

        # Use load_dataset for auto-detection
        return self.load_dataset("auto")

    def process_auction_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process auction data for incrementality measurement.

        Expected columns (based on typical auction datasets):
        - auction_id: Unique auction identifier
        - bidder_id: Bidder identifier
        - bid_price: Bid price
        - ad_id: Advertisement identifier
        - timestamp: Auction timestamp
        - advertiser_id: Advertiser identifier
        - placement_id: Placement identifier
        - win: Whether the bid won (0 or 1)
        - click: Whether ad was clicked (0 or 1)
        - conversion: Whether conversion occurred (0 or 1)
        - revenue: Revenue from conversion

        Args:
            df: Raw auction data DataFrame

        Returns:
            Dictionary with processed data:
            - bids: Bid data
            - wins: Winning bids
            - conversions: Conversion data
            - revenue: Revenue data
        """
        processed = {}

        # Extract bid data
        bid_columns = [
            "auction_id",
            "bidder_id",
            "bid_price",
            "ad_id",
            "timestamp",
            "advertiser_id",
            "placement_id",
        ]
        available_bid_cols = [col for col in bid_columns if col in df.columns]

        if available_bid_cols:
            processed["bids"] = df[available_bid_cols].copy()
        else:
            processed["bids"] = df.copy()

        # Extract winning bids
        if "win" in df.columns:
            processed["wins"] = df[df["win"] == 1].copy()
        else:
            processed["wins"] = pd.DataFrame()

        # Extract conversion data
        if "conversion" in df.columns:
            processed["conversions"] = df[df["conversion"] == 1].copy()
        else:
            processed["conversions"] = pd.DataFrame()

        # Extract revenue data
        if "revenue" in df.columns:
            processed["revenue"] = df[["auction_id", "revenue", "timestamp"]].copy()
            processed["revenue"] = processed["revenue"][processed["revenue"]["revenue"] > 0]
        else:
            processed["revenue"] = pd.DataFrame()

        return processed

    def prepare_for_incrementality(
        self,
        df: pd.DataFrame,
        treatment_column: str = "treatment",
        revenue_column: str = "revenue",
        spend_column: str = "bid_price",
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare auction data for incrementality calculations.

        Groups data by treatment/control and calculates metrics.

        Args:
            df: Auction data DataFrame
            treatment_column: Column name for treatment indicator
            revenue_column: Column name for revenue
            spend_column: Column name for ad spend

        Returns:
            Dictionary with test and control group data
        """
        if treatment_column not in df.columns:
            # Create treatment assignment if not exists
            np.random.seed(42)
            df[treatment_column] = np.random.binomial(1, 0.5, len(df))

        # Split into test and control
        test_group = df[df[treatment_column] == 1].copy()
        control_group = df[df[treatment_column] == 0].copy()

        # Calculate metrics
        test_metrics = {}
        control_metrics = {}

        if revenue_column in df.columns:
            test_metrics["revenue"] = test_group[revenue_column].sum()
            control_metrics["revenue"] = control_group[revenue_column].sum()

        if spend_column in df.columns:
            test_metrics["spend"] = test_group[spend_column].sum()
            control_metrics["spend"] = control_group[spend_column].sum()

        # Calculate conversions if available
        if "conversion" in df.columns:
            test_metrics["conversions"] = test_group["conversion"].sum()
            control_metrics["conversions"] = control_group["conversion"].sum()

        return {
            "test_group": test_group,
            "control_group": control_group,
            "test_metrics": test_metrics,
            "control_metrics": control_metrics,
        }


def integrate_kaggle_with_incrementality(
    dataset_path: str = "data/kaggle", dataset_type: str = "auto", dataset_name: str = None
) -> Dict[str, any]:
    """
    Integrate Kaggle dataset with incrementality measurement.

    Supports multiple datasets:
    - 'auto': Auto-detect dataset type
    - 'auction': Real-time Advertisers Auction dataset (Dataset.csv - has revenue data)
    - 'video_ads': Video Ads Engagement dataset

    Args:
        dataset_path: Path to dataset directory
        dataset_type: Type of dataset ('auto', 'rtb', 'video_ads', 'auction')
        dataset_name: Kaggle dataset name for downloading (optional)

    Returns:
        Dictionary with processed data and incrementality metrics
    """
    # Initialize integration
    kaggle = KaggleDatasetIntegration(dataset_path)

    # Download dataset if needed
    if dataset_name:
        print(f"Downloading dataset: {dataset_name}")
        success = kaggle.download_dataset(dataset_name)
        if not success:
            print("Could not download dataset. Please download manually from Kaggle.")

    # Try to load data
    try:
        df = kaggle.load_dataset(dataset_type)
        print(f"Loaded dataset with {len(df)} rows")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please download the dataset using download_dataset()")
        return {}

    # Process data
    processed = kaggle.process_auction_data(df)

    # Prepare for incrementality
    prepared = kaggle.prepare_for_incrementality(df)

    # Summary
    summary = {"total_rows": len(df), "columns": list(df.columns), "dataset_type": dataset_type}

    # Add date range if available
    time_cols = [
        col
        for col in df.columns
        if "time" in col.lower() or "date" in col.lower() or "timestamp" in col.lower()
    ]
    if time_cols:
        time_col = time_cols[0]
        summary["date_range"] = f"{df[time_col].min()} to {df[time_col].max()}"
    else:
        summary["date_range"] = "N/A"

    return {
        "raw_data": df,
        "processed_data": processed,
        "prepared_data": prepared,
        "summary": summary,
    }


# Example usage
if __name__ == "__main__":
    print("Kaggle Dataset Integration")
    print("=" * 80)
    print("\nSupported Datasets:")
    print("1. Real-time Advertisers Auction")
    print("   URL: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction")
    print("2. Video Ads Engagement")
    print("   URL: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset")
    print("\nTo use this module:")
    print("1. Set up .env file with KAGGLE_USERNAME and KAGGLE_KEY")
    print("   See API_KEYS_SETUP.md for detailed instructions")
    print("2. Install kaggle package: pip install kaggle")
    print("3. Use download_dataset() to fetch the data")
    print("4. Load and process data with load_dataset() and process_auction_data()")
    print("\nExample:")
    print("  kaggle = KaggleDatasetIntegration()")
    print("  kaggle.download_dataset('saurav9786/real-time-advertisers-auction')")
    print("  df = kaggle.load_dataset('auction')")
    print("  processed = kaggle.process_auction_data(df)")
    print("\nOr use integration function:")
    print(
        "  result = integrate_kaggle_with_incrementality(dataset_type='auction', dataset_name='saurav9786/real-time-advertisers-auction')"
    )

    # Try to integrate if data is available
    try:
        result = integrate_kaggle_with_incrementality(dataset_type="auto")
        if result and result.get("summary"):
            print("\n" + "=" * 80)
            print("Dataset Summary:")
            print(f"Total rows: {result['summary']['total_rows']}")
            print(f"Dataset type: {result['summary']['dataset_type']}")
            print(f"Columns: {', '.join(result['summary']['columns'][:10])}...")
    except Exception as e:
        print(f"\nNote: Could not load dataset - {e}")
        print("This is expected if the dataset hasn't been downloaded yet.")
        print("\nTo download datasets:")
        print("1. Set up KAGGLE_USERNAME and KAGGLE_KEY in .env file")
        print(
            "2. Run: python3 -c \"from demo.kaggle_integration import KaggleDatasetIntegration; k = KaggleDatasetIntegration(); k.download_dataset('saurav9786/real-time-advertisers-auction')\""
        )
