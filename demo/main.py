"""
Main Demo Application - Incremental Advertising Measurement

Runs demos using real Kaggle datasets:
- Real-time Auction: https://www.kaggle.com/datasets/saurav9786/real-time-advertisers-auction
- Video Ads: https://www.kaggle.com/datasets/karnikakapoor/video-ads-engagement-dataset
"""

import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from demo.incrementality import calculate_iroas, calculate_lift, test_significance
    from demo.cuped import cuped_adjustment, statistical_significance
    from demo.tree_causal import robust_causal_inference
    from demo.uplift_models import TLearner, XLearner, DRLearner, SLearner
    from demo.ghost_bidding import ghost_bidding_simulation
    from demo.kaggle_integration import KaggleDatasetIntegration
except ImportError:
    from incrementality import calculate_iroas, calculate_lift, test_significance
    from cuped import cuped_adjustment, statistical_significance
    from tree_causal import robust_causal_inference
    from uplift_models import TLearner, XLearner, DRLearner, SLearner
    from ghost_bidding import ghost_bidding_simulation
    from kaggle_integration import KaggleDatasetIntegration


def load_datasets() -> Dict[str, pd.DataFrame]:
    """Load Real-time Auction and Video Ads datasets from Kaggle.

    Returns:
        Dictionary with 'auction' and/or 'video_ads' DataFrames
    """
    kaggle = KaggleDatasetIntegration("data/kaggle")
    datasets: Dict[str, pd.DataFrame] = {}

    # Load Real-time Auction dataset (has revenue data)
    filepath = kaggle.dataset_path / "Dataset.csv"
    if filepath.exists():
        datasets["auction"] = pd.read_csv(filepath, nrows=10000)
        logger.info(f"Loaded Real-time Auction: {len(datasets['auction'])} rows")

    # Load Video Ads dataset
    filepath = kaggle.dataset_path / "ad_df.csv"
    if filepath.exists():
        datasets["video_ads"] = pd.read_csv(filepath, nrows=10000)
        logger.info(f"Loaded Video Ads: {len(datasets['video_ads'])} rows")

    return datasets


def prepare_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare dataset for incrementality measurement.

    Args:
        df: Input DataFrame with either video ads or auction data

    Returns:
        Dictionary with 'features', 'treatment', 'outcome', 'spend', 'revenue', 'df'
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is None or empty. Download datasets first.")

    # Video Ads dataset
    if "seconds_played" in df.columns:
        if "creative_duration" not in df.columns:
            raise ValueError("Video Ads dataset missing 'creative_duration' column. Check dataset.")

        outcome = df["seconds_played"].fillna(0).values
        feature_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["seconds_played", "timestamp", "user_average_seconds_played"]
        ][:10]
        if len(feature_cols) == 0:
            raise ValueError("Video Ads dataset missing numeric feature columns.")

        X = df[feature_cols].fillna(0).values
        spend = df["creative_duration"].fillna(0).values
        if spend.sum() == 0:
            raise ValueError("All spend values are zero. Check 'creative_duration' column.")
        revenue = outcome

    # Real-time Auction dataset (has revenue data)
    elif "total_revenue" in df.columns:
        if "total_impressions" not in df.columns:
            raise ValueError("Real-time Auction dataset missing 'total_impressions' column.")

        # Use total_impressions as spend proxy (impression cost)
        # Estimate spend: assume average CPM of $2.00
        spend = (df["total_impressions"].fillna(0).values * 2.0) / 1000.0
        revenue = df["total_revenue"].fillna(0).values
        outcome = (revenue > 0).astype(float)  # Binary conversion indicator

        # Use numeric columns as features
        feature_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c
            not in [
                "total_revenue",
                "total_impressions",
                "viewable_impressions",
                "measurable_impressions",
                "revenue_share_percent",
            ]
        ][:10]
        if len(feature_cols) == 0:
            raise ValueError("Real-time Auction dataset missing numeric feature columns.")

        X = df[feature_cols].fillna(0).values

        if spend.sum() == 0:
            raise ValueError("All spend values are zero. Check 'total_impressions' column.")

    else:
        raise ValueError(
            "Dataset format not recognized. Expected 'seconds_played' (Video Ads) or 'total_revenue' (Real-time Auction) column."
        )

    # Random treatment assignment for RCT
    np.random.seed(42)
    treatment = np.random.binomial(1, 0.5, len(df))

    return {
        "features": X,
        "treatment": treatment,
        "outcome": outcome,
        "spend": spend,
        "revenue": revenue,
        "df": df,
    }


def demo_cuped(rtb_data: Optional[Dict[str, Any]]) -> None:
    """DEMO 1: A/B Test with CUPED

    Args:
        rtb_data: Dictionary with prepared data or None
    """
    if rtb_data is None:
        return

    logger.info("\n" + "=" * 60)
    logger.info("DEMO 1: A/B Test with CUPED")
    logger.info("=" * 60)

    df = rtb_data["df"]
    treatment = rtb_data["treatment"]
    outcome = rtb_data["outcome"]

    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]

    if time_cols:
        time_col = time_cols[0]
        # Convert to datetime if string
        if df[time_col].dtype == "object":
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        median_time = df[time_col].median()
        pre_mask = df[time_col] <= median_time

        test_pre = outcome[(treatment == 1) & pre_mask]
        control_pre = outcome[(treatment == 0) & pre_mask]
        test_post = outcome[(treatment == 1) & ~pre_mask]
        control_post = outcome[(treatment == 0) & ~pre_mask]

        # CUPED requires same length arrays - use simple comparison if sizes differ substantially
        if (
            len(test_pre) > 10
            and len(control_pre) > 10
            and len(test_post) > 10
            and len(control_post) > 10
        ):
            # Try to use min length if sizes differ
            min_len_pre = min(len(test_pre), len(control_pre))
            min_len_post = min(len(test_post), len(control_post))

            if min_len_pre == min_len_post and min_len_pre > 10:
                test_pre = test_pre[:min_len_pre]
                control_pre = control_pre[:min_len_pre]
                test_post = test_post[:min_len_post]
                control_post = control_post[:min_len_post]

                results = cuped_adjustment(test_pre, test_post, control_pre, control_post)
                checks = results["assumption_checks"]

                # Display assumption checks
                logger.info("\nAssumption Checks:")
                logger.info(f"  Sample Size: {'PASS' if checks['sample_size_passed'] else 'FAIL'}")
                if not checks["sample_size_passed"]:
                    for err in checks["sample_size_errors"]:
                        logger.warning(f"    - {err}")
                logger.info(
                    f"  Pre-Post Correlation: {'PASS' if checks['pre_post_correlation_passed'] else 'FAIL'} (r={checks['pre_post_correlation']:.3f})"
                )
                logger.info(
                    f"  Balanced Groups: {'PASS' if checks['balanced_groups_passed'] else 'FAIL'}"
                )
                logger.info(
                    f"  All Assumptions: {'PASS' if results['assumptions_passed'] else 'FAIL'}"
                )

                if not results["assumptions_passed"]:
                    logger.warning("\nWARNING: Some assumptions failed. Results may be invalid.")

                sig = statistical_significance(test_pre, test_post, control_pre, control_post)
                logger.info(f"\nUnadjusted Effect: {results['unadjusted_effect']:.4f}")
                logger.info(f"CUPED Effect: {results['adjusted_effect']:.4f}")
                logger.info(f"Variance Reduction: {results['variance_reduction']*100:.1f}%")
                logger.info(f"P-value: {sig['p_value']:.4f}, Significant: {sig['is_significant']}")
                return

    # Simple comparison
    test_outcomes = outcome[treatment == 1]
    control_outcomes = outcome[treatment == 0]

    if len(test_outcomes) > 10:
        sig = test_significance(test_outcomes, control_outcomes)
        lift = calculate_lift(np.mean(test_outcomes), np.mean(control_outcomes))
        logger.info(f"Test: {np.mean(test_outcomes):.4f}, Control: {np.mean(control_outcomes):.4f}")
        logger.info(f"Lift: {lift:.2f}%, P-value: {sig['p_value']:.4f}")


def demo_tree_causal(rtb_data: Optional[Dict[str, Any]]) -> None:
    """DEMO 2: Tree-based Causal Inference

    Args:
        rtb_data: Dictionary with prepared data or None
    """
    if rtb_data is None:
        return

    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Tree-based Causal Inference")
    logger.info("=" * 60)

    X = rtb_data["features"]
    treatment = rtb_data["treatment"]
    outcome = rtb_data["outcome"]

    if len(X) < 100:
        logger.warning("Insufficient data (need >= 100 samples)")
        return

    results = robust_causal_inference(X, treatment, outcome)
    checks = results["assumption_checks"]

    # Display assumption checks
    logger.info("\nAssumption Checks:")
    logger.info(f"  Sample Size: {'PASS' if checks['sample_size_passed'] else 'FAIL'}")
    logger.info(f"  RCT Balance: {'PASS' if checks['balanced_treatment_passed'] else 'FAIL'}")
    logger.info(f"  Positivity: {'PASS' if checks['positivity_passed'] else 'FAIL'}")
    if checks["errors"]:
        for err in checks["errors"]:
            logger.warning(f"    - {err}")
    logger.info(f"  All Assumptions: {'PASS' if results['assumptions_passed'] else 'FAIL'}")

    if not results["assumptions_passed"]:
        logger.warning("\nWARNING: Some assumptions failed. Results may be invalid.")

    logger.info(f"\nAverage Treatment Effect: {results['average_treatment_effect']:.4f}")
    logger.info(f"Effect Variance: {results['effect_variance']:.4f}")
    logger.info(f"Heterogeneity: {len(results['segment_stats'])} segments identified")


def demo_uplift(rtb_data: Optional[Dict[str, Any]]) -> None:
    """DEMO 3: Uplift Modeling

    Args:
        rtb_data: Dictionary with prepared data or None
    """
    if rtb_data is None:
        return

    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Uplift Modeling")
    logger.info("=" * 60)

    X = rtb_data["features"]
    treatment = rtb_data["treatment"]
    outcome = rtb_data["outcome"]

    if len(X) < 100:
        logger.warning("Insufficient data (need >= 100 samples)")
        return

    # Check assumptions
    from demo.uplift_models import check_uplift_assumptions

    assumption_checks = check_uplift_assumptions(X, treatment, outcome)

    logger.info("\nAssumption Checks:")
    logger.info(f"  Sample Size: {'PASS' if assumption_checks['sample_size_passed'] else 'FAIL'}")
    logger.info(f"  RCT Balance: {'PASS' if assumption_checks['rct_balance_passed'] else 'FAIL'}")
    logger.info(f"  Positivity: {'PASS' if assumption_checks['positivity_passed'] else 'FAIL'}")
    logger.info(
        f"  Sufficient Per Group: {'PASS' if assumption_checks['sufficient_per_group_passed'] else 'FAIL'}"
    )
    if assumption_checks["errors"]:
        for err in assumption_checks["errors"]:
            logger.warning(f"    - {err}")
    logger.info(
        f"  All Assumptions: {'PASS' if assumption_checks['all_assumptions_passed'] else 'FAIL'}"
    )

    if not assumption_checks["all_assumptions_passed"]:
        logger.warning("\nWARNING: Some assumptions failed. Results may be invalid.")

    # Simple uplift model comparison
    models = ["T-Learner", "S-Learner", "X-Learner", "DR-Learner"]

    logger.info("\nModel Performance:")
    test_mean = outcome[treatment == 1].mean()
    control_mean = outcome[treatment == 0].mean()
    true_uplift = test_mean - control_mean

    for name in models:
        try:
            if name == "T-Learner":
                model = TLearner()
            elif name == "S-Learner":
                model = SLearner()
            elif name == "X-Learner":
                model = XLearner()
            else:
                model = DRLearner()

            model.fit(X, treatment, outcome)
            predictions = model.predict_tau(X)
            avg_uplift = predictions.mean()
            mse = np.mean((predictions - true_uplift) ** 2)
            logger.info(f"  {name}: Avg Uplift = {avg_uplift:.4f}, MSE = {mse:.4f}")
        except Exception as e:
            logger.error(f"  {name}: Failed ({str(e)[:30]})")


def demo_ghost_bidding(rtb_data: Optional[Dict[str, Any]]) -> None:
    """DEMO 4: Ghost Bidding Simulation

    Args:
        rtb_data: Dictionary with prepared data or None
    """
    if rtb_data is None:
        return

    df = rtb_data["df"]

    # Require real bid columns - return silently if missing
    required_bid_cols = ["bid_price", "our_bid", "bid", "price"]
    required_competition_cols = ["competition_bid", "competitor_bid", "second_price"]

    our_bid_col = None
    for col in required_bid_cols:
        if col in df.columns:
            our_bid_col = col
            break

    if our_bid_col is None:
        return

    competition_bid_col = None
    for col in required_competition_cols:
        if col in df.columns:
            competition_bid_col = col
            break

    if competition_bid_col is None:
        return

    if "convert" not in df.columns:
        return

    our_bids = df[our_bid_col].fillna(0).values
    competition_bids = df[competition_bid_col].fillna(0).values
    conversion_rates = df["convert"].fillna(0).values

    if len(our_bids) < 100:
        return

    if our_bids.sum() == 0:
        return

    # Check assumptions
    checks = {
        "bid_data_passed": True,
        "sample_size_passed": True,
        "valid_bids_passed": True,
        "all_assumptions_passed": True,
        "errors": [],
    }

    if len(our_bids) < 100:
        checks["sample_size_passed"] = False
        checks["errors"].append(f"Sample size {len(our_bids)} < 100")

    if our_bids.sum() == 0:
        checks["valid_bids_passed"] = False
        checks["errors"].append("All bid values are zero")

    if competition_bids.sum() == 0:
        checks["errors"].append("All competition bid values are zero")

    checks["all_assumptions_passed"] = (
        checks["bid_data_passed"] and checks["sample_size_passed"] and checks["valid_bids_passed"]
    )

    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Ghost Bidding Simulation")
    logger.info("=" * 60)

    logger.info("\nAssumption Checks:")
    logger.info(f"  Bid Data Available: {'PASS' if checks['bid_data_passed'] else 'FAIL'}")
    logger.info(f"  Sample Size: {'PASS' if checks['sample_size_passed'] else 'FAIL'}")
    logger.info(f"  Valid Bids: {'PASS' if checks['valid_bids_passed'] else 'FAIL'}")
    if checks["errors"]:
        for err in checks["errors"]:
            logger.warning(f"    - {err}")
    logger.info(f"  All Assumptions: {'PASS' if checks['all_assumptions_passed'] else 'FAIL'}")

    if not checks["all_assumptions_passed"]:
        logger.warning("\nWARNING: Some assumptions failed. Results may be invalid.")

    n_samples = min(5000, len(our_bids))
    results = ghost_bidding_simulation(
        n_samples,
        our_bids[:n_samples],
        competition_bids[:n_samples],
        conversion_rates[:n_samples],
        control_ratio=0.1,
    )

    logger.info(f"\nProcessed {n_samples} impressions with real bid data")
    logger.info(
        f"Treatment CVR: {results['treatment_cvr']:.4f}, Control CVR: {results['control_cvr']:.4f}"
    )
    logger.info(f"Incremental Conversions: {results['incremental_conversions']:.1f}")
    logger.info(f"Lift: {results['lift']*100:.2f}%, P-value: {results['p_value']:.4f}")


def demo_iroas_vs_roas(rtb_data: Optional[Dict[str, Any]]) -> None:
    """DEMO 5: iROAS vs ROAS Comparison

    Args:
        rtb_data: Dictionary with prepared data or None
    """
    if rtb_data is None:
        return

    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: iROAS vs ROAS Comparison")
    logger.info("=" * 60)

    revenue = rtb_data["revenue"]
    spend = rtb_data["spend"]
    treatment = rtb_data["treatment"]

    # Check assumptions
    checks = {
        "rct_balance_passed": True,
        "sample_size_passed": True,
        "valid_spend_passed": True,
        "valid_revenue_passed": True,
        "all_assumptions_passed": True,
        "errors": [],
    }

    n_test = np.sum(treatment == 1)
    n_control = np.sum(treatment == 0)

    if n_test < 10 or n_control < 10:
        checks["sample_size_passed"] = False
        checks["errors"].append(f"Test: {n_test}, Control: {n_control} (need >= 10 each)")

    treatment_ratio = n_test / (n_test + n_control) if (n_test + n_control) > 0 else 0.0
    if treatment_ratio < 0.1 or treatment_ratio > 0.9:
        checks["rct_balance_passed"] = False
        checks["errors"].append(f"Treatment ratio {treatment_ratio:.2f} indicates imbalance")

    test_spend = np.sum(spend[treatment == 1])
    if test_spend <= 0:
        checks["valid_spend_passed"] = False
        checks["errors"].append("Test group spend is zero or negative")

    test_revenue = np.sum(revenue[treatment == 1])
    control_revenue = np.sum(revenue[treatment == 0])

    checks["all_assumptions_passed"] = (
        checks["rct_balance_passed"]
        and checks["sample_size_passed"]
        and checks["valid_spend_passed"]
        and checks["valid_revenue_passed"]
    )

    logger.info("\nAssumption Checks:")
    logger.info(f"  RCT Balance: {'PASS' if checks['rct_balance_passed'] else 'FAIL'}")
    logger.info(f"  Sample Size: {'PASS' if checks['sample_size_passed'] else 'FAIL'}")
    logger.info(f"  Valid Spend: {'PASS' if checks['valid_spend_passed'] else 'FAIL'}")
    if checks["errors"]:
        for err in checks["errors"]:
            logger.warning(f"    - {err}")
    logger.info(f"  All Assumptions: {'PASS' if checks['all_assumptions_passed'] else 'FAIL'}")

    if not checks["all_assumptions_passed"]:
        logger.warning("\nWARNING: Some assumptions failed. Results may be invalid.")

    incremental_revenue = test_revenue - control_revenue
    iroas = calculate_iroas(incremental_revenue, test_spend)
    roas = (test_revenue / test_spend * 100.0) if test_spend > 0 else 0.0

    logger.info(f"\nTest Revenue: ${test_revenue:.2f}, Control: ${control_revenue:.2f}")
    logger.info(f"Incremental Revenue: ${incremental_revenue:.2f}")
    logger.info(f"iROAS: {iroas:.2f}%, ROAS: {roas:.2f}%")
    logger.info(f"Gap: {roas - iroas:.2f} percentage points")


def demo_workflow(rtb_data: Optional[Dict[str, Any]], video_data: Optional[Dict[str, Any]]) -> None:
    """DEMO 6: Complete Experiment Workflow

    Args:
        rtb_data: Dictionary with prepared auction data or None
        video_data: Dictionary with prepared video ads data or None
    """
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 6: Complete Experiment Workflow")
    logger.info("=" * 60)

    data = rtb_data if rtb_data is not None else video_data
    if data is None:
        logger.warning("No data available")
        return

    treatment = data["treatment"]
    outcome = data["outcome"]

    test_outcomes = outcome[treatment == 1]
    control_outcomes = outcome[treatment == 0]

    if len(test_outcomes) > 10:
        sig = test_significance(test_outcomes, control_outcomes)
        lift = calculate_lift(np.mean(test_outcomes), np.mean(control_outcomes))
        logger.info(f"Test: {len(test_outcomes)} samples, mean = {np.mean(test_outcomes):.4f}")
        logger.info(
            f"Control: {len(control_outcomes)} samples, mean = {np.mean(control_outcomes):.4f}"
        )
        logger.info(
            f"Lift: {lift:.2f}%, P-value: {sig['p_value']:.4f}, Significant: {sig['is_significant']}"
        )


def main() -> None:
    """Run all demonstrations."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "results.txt"

    # Tee output to file and console
    class Tee:
        def __init__(self, console: Any, file: Any) -> None:
            self.console, self.file = console, file

        def write(self, obj: str) -> None:
            self.console.write(obj)
            self.file.write(obj)
            self.file.flush()

        def flush(self) -> None:
            self.console.flush()
            self.file.flush()

    original_stdout = sys.stdout
    try:
        with open(results_file, "w") as f:
            sys.stdout = Tee(original_stdout, f)

            logger.info("\n" + "=" * 60)
            logger.info("INCREMENTAL ADVERTISING DEMO")
            logger.info("=" * 60)

            # Load datasets
            datasets = load_datasets()
            if not datasets:
                logger.error("\nERROR: No datasets found!")
                logger.error("Download datasets: python3 demo/run_with_kaggle.py")
                return

            # Prepare data
            logger.info("\nPreparing data...")
            # Use Real-time Auction dataset (Dataset.csv) - has revenue data
            auction_data = None
            if datasets.get("auction") is not None:
                try:
                    auction_data = prepare_data(datasets.get("auction"))
                    logger.info("Using Real-time Auction dataset (Dataset.csv) - has revenue data")
                except Exception as e:
                    logger.error(f"Error preparing auction data: {e}")

            video_data = None
            if datasets.get("video_ads") is not None:
                try:
                    video_data = prepare_data(datasets.get("video_ads"))
                except Exception as e:
                    logger.error(f"Error preparing video ads data: {e}")

            # Run demos with auction data
            if auction_data:
                demo_cuped(auction_data)
                demo_tree_causal(auction_data)
                demo_uplift(auction_data)
                # Only run ghost bidding if bid columns exist - otherwise skip silently
                demo_ghost_bidding(auction_data)
                demo_iroas_vs_roas(auction_data)
                demo_workflow(auction_data, video_data)
            elif video_data:
                logger.info("\nNo auction data available. Running with video ads data only.")
                demo_workflow(None, video_data)
            else:
                logger.error("\nERROR: No usable datasets available!")
                logger.error(
                    "Download datasets with revenue/spend data: python3 demo/run_with_kaggle.py"
                )

            logger.info("\n" + "=" * 60)
            logger.info("COMPLETE - Results saved to: demo/results/results.txt")
            logger.info("=" * 60)
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
