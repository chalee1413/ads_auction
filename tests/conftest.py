"""
Pytest configuration and shared fixtures for tests.
"""

import numpy as np
import pytest
from typing import Dict, Tuple


@pytest.fixture
def sample_data():
    """Generate sample test and control data for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    test_revenue = np.random.normal(100.0, 20.0, n_samples)
    control_revenue = np.random.normal(90.0, 18.0, n_samples)
    test_outcomes = np.random.binomial(1, 0.1, n_samples).astype(float)
    control_outcomes = np.random.binomial(1, 0.08, n_samples).astype(float)
    
    return {
        'test_revenue': test_revenue,
        'control_revenue': control_revenue,
        'test_outcomes': test_outcomes,
        'control_outcomes': control_outcomes,
        'n_samples': n_samples
    }


@pytest.fixture
def cuped_data():
    """Generate sample pre/post data for CUPED testing."""
    np.random.seed(42)
    
    n_samples = 100
    # Create correlated pre/post data
    base_pre = np.random.normal(50.0, 10.0, n_samples)
    noise = np.random.normal(0.0, 5.0, n_samples)
    base_post = base_pre * 0.8 + noise  # Correlated with pre-period
    
    test_pre = base_pre + np.random.normal(0.0, 2.0, n_samples)
    test_post = base_post + np.random.normal(5.0, 2.0, n_samples)  # Treatment effect
    
    control_pre = base_pre + np.random.normal(0.0, 2.0, n_samples)
    control_post = base_post + np.random.normal(0.0, 2.0, n_samples)
    
    return {
        'test_pre': test_pre,
        'test_post': test_post,
        'control_pre': control_pre,
        'control_post': control_post
    }


@pytest.fixture
def causal_data():
    """Generate sample feature and treatment data for causal inference."""
    np.random.seed(42)
    
    n_samples = 500
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Create outcome with treatment effect
    outcome = (
        10.0 + 
        np.sum(X, axis=1) * 2.0 + 
        treatment * 5.0 + 
        np.random.normal(0.0, 3.0, n_samples)
    )
    
    return {
        'X': X,
        'treatment': treatment,
        'outcome': outcome,
        'n_samples': n_samples,
        'n_features': n_features
    }

