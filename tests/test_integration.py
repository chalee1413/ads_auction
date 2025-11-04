"""
Integration tests for main demo workflow.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: These are integration tests that test the main workflow
# We'll test data preparation and basic workflow without requiring
# actual Kaggle datasets


class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_prepare_data_structure(self):
        """Test that prepare_data returns expected structure."""
        # Create mock dataframe
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "total_revenue": np.random.uniform(0, 100, 1000),
                "total_impressions": np.random.randint(100, 10000, 1000),
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
                "feature3": np.random.randn(1000),
            }
        )

        # Import after path setup
        from demo.main import prepare_data

        result = prepare_data(df)

        assert "features" in result
        assert "treatment" in result
        assert "outcome" in result
        assert "spend" in result
        assert "revenue" in result
        assert "df" in result

        assert isinstance(result["features"], np.ndarray)
        assert isinstance(result["treatment"], np.ndarray)
        assert isinstance(result["outcome"], np.ndarray)
        assert isinstance(result["spend"], np.ndarray)
        assert isinstance(result["revenue"], np.ndarray)

        assert len(result["features"]) == len(df)
        assert len(result["treatment"]) == len(df)
        assert len(result["outcome"]) == len(df)
        assert len(result["spend"]) == len(df)
        assert len(result["revenue"]) == len(df)

    def test_prepare_data_treatment_distribution(self):
        """Test that treatment assignment is roughly balanced."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "total_revenue": np.random.uniform(0, 100, 1000),
                "total_impressions": np.random.randint(100, 10000, 1000),
                "feature1": np.random.randn(1000),
                "feature2": np.random.randn(1000),
            }
        )

        from demo.main import prepare_data

        result = prepare_data(df)
        treatment = result["treatment"]

        # Treatment should be roughly 50/50
        treatment_ratio = np.mean(treatment)
        assert 0.4 <= treatment_ratio <= 0.6

    def test_prepare_data_empty_dataframe(self):
        """Test that empty dataframe raises error."""
        df = pd.DataFrame()

        from demo.main import prepare_data

        with pytest.raises(ValueError):
            prepare_data(df)

    def test_prepare_data_missing_columns(self):
        """Test that missing required columns raises error."""
        df = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})

        from demo.main import prepare_data

        with pytest.raises(ValueError):
            prepare_data(df)


class TestIncrementalityWorkflow:
    """Tests for incrementality calculation workflow."""

    def test_end_to_end_calculation(self):
        """Test end-to-end incrementality calculation."""
        from demo.incrementality import (
            calculate_incremental_revenue,
            calculate_iroas,
            calculate_lift,
        )

        # Simulate test and control groups
        test_revenue = 1000.0
        control_revenue = 800.0
        ad_spend = 100.0

        incremental_revenue = calculate_incremental_revenue(test_revenue, control_revenue)
        iroas = calculate_iroas(incremental_revenue, ad_spend)
        lift = calculate_lift(test_revenue, control_revenue)

        assert incremental_revenue == 200.0
        assert abs(iroas - 200.0) < 1e-6  # 200%
        assert abs(lift - 25.0) < 1e-6  # 25% lift


class TestCUPEDWorkflow:
    """Tests for CUPED workflow."""

    def test_cuped_workflow(self):
        """Test CUPED adjustment workflow."""
        from demo.cuped import cuped_adjustment, calculate_theta

        np.random.seed(42)
        n = 100

        # Create correlated pre/post data
        pre = np.random.normal(50.0, 10.0, n)
        post = pre * 0.8 + np.random.normal(0.0, 5.0, n)

        test_pre = pre
        test_post = post + np.random.normal(5.0, 2.0, n)  # Treatment effect
        control_pre = pre
        control_post = post

        # Calculate Theta
        theta = calculate_theta(pre, post)

        # Run CUPED adjustment
        result = cuped_adjustment(test_pre, test_post, control_pre, control_post)

        assert "adjusted_effect" in result
        assert "unadjusted_effect" in result
        assert "variance_reduction" in result
        assert isinstance(result["adjusted_effect"], float)


class TestCausalInferenceWorkflow:
    """Tests for causal inference workflow."""

    def test_causal_inference_workflow(self):
        """Test causal inference workflow."""
        from demo.tree_causal import robust_causal_inference

        np.random.seed(42)
        n_samples = 500
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        treatment = np.random.binomial(1, 0.5, n_samples)
        outcome = (
            10.0 + np.sum(X, axis=1) * 2.0 + treatment * 5.0 + np.random.normal(0.0, 3.0, n_samples)
        )

        results = robust_causal_inference(X, treatment, outcome)

        assert "average_treatment_effect" in results
        assert results["average_treatment_effect"] is not None
        assert np.isfinite(results["average_treatment_effect"])
