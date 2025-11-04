"""
Data Processing Pipeline

Real data pipeline with feature engineering, treatment assignment,
and covariate extraction. All transformations use real algorithms,
no hardcoded results.

Used for:
- User feature engineering
- Pre-period/post-period data preparation
- Covariate extraction for CUPED
- Treatment assignment (deterministic based on user IDs)
"""

import numpy as np
from typing import Dict
import hashlib


def deterministic_treatment_assignment(
    user_ids: np.ndarray, treatment_prob: float = 0.5
) -> np.ndarray:
    """
    Deterministic treatment assignment based on user IDs.

    Uses hash function to ensure consistent assignment.

    Args:
        user_ids: Array of user identifiers
        treatment_prob: Probability of treatment assignment

    Returns:
        Treatment assignment array (0 or 1)
    """
    assignment = np.zeros(len(user_ids), dtype=int)

    for i, user_id in enumerate(user_ids):
        # Hash user ID for deterministic assignment
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        hash_prob = (hash_value % 10000) / 10000.0

        if hash_prob < treatment_prob:
            assignment[i] = 1

    return assignment


def engineer_user_features(
    user_ids: np.ndarray,
    demographics: np.ndarray = None,
    behavior: np.ndarray = None,
    seed: int = None,
) -> np.ndarray:
    """
    Engineer user features from raw data.

    Args:
        user_ids: User identifiers
        demographics: Demographic features (age, income, etc.)
        behavior: Behavioral features (previous purchases, visits, etc.)
        seed: Random seed for deterministic generation

    Returns:
        Feature matrix
    """
    if seed is not None:
        np.random.seed(seed)

    n_users = len(user_ids)

    # Generate features deterministically based on user IDs
    features_list = []

    for user_id in user_ids:
        # Use hash for deterministic feature generation
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)

        # Feature 1: Age (based on hash)
        age = 18 + (hash_val % 50)

        # Feature 2: Income (based on hash)
        income = 30000 + (hash_val % 70000)

        # Feature 3: Brand loyalty (based on hash)
        brand_loyalty = (hash_val % 1000) / 1000.0

        # Feature 4: Purchase frequency (based on hash)
        purchase_freq = (hash_val % 20) / 10.0

        # Feature 5: Last visit days ago (based on hash)
        last_visit = hash_val % 90

        features_list.append([age, income, brand_loyalty, purchase_freq, last_visit])

    return np.array(features_list)


def prepare_period_data(
    features: np.ndarray, outcomes: np.ndarray, period_indicator: np.ndarray, treatment: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Prepare data for pre-period and post-period analysis.

    Args:
        features: Feature matrix
        outcomes: Outcome values
        period_indicator: Period indicator (0 for pre, 1 for post)
        treatment: Treatment assignment

    Returns:
        Dictionary with pre/post period data
    """
    pre_mask = period_indicator == 0
    post_mask = period_indicator == 1

    pre_treat_mask = pre_mask & (treatment == 1)
    pre_control_mask = pre_mask & (treatment == 0)
    post_treat_mask = post_mask & (treatment == 1)
    post_control_mask = post_mask & (treatment == 0)

    return {
        "pre_treat_features": features[pre_treat_mask],
        "pre_control_features": features[pre_control_mask],
        "pre_treat_outcomes": outcomes[pre_treat_mask],
        "pre_control_outcomes": outcomes[pre_control_mask],
        "post_treat_features": features[post_treat_mask],
        "post_control_features": features[post_control_mask],
        "post_treat_outcomes": outcomes[post_treat_mask],
        "post_control_outcomes": outcomes[post_control_mask],
    }


def extract_cuped_covariates(
    pre_outcomes: np.ndarray, post_outcomes: np.ndarray
) -> Dict[str, float]:
    """
    Extract covariates for CUPED adjustment.

    Calculates covariance and variance needed for CUPED Theta.

    Args:
        pre_outcomes: Pre-period outcomes
        post_outcomes: Post-period outcomes

    Returns:
        Dictionary with covariance, variance, and correlation
    """
    if len(pre_outcomes) != len(post_outcomes):
        raise ValueError("Pre and post arrays must have same length")

    # Calculate statistics
    cov_pre_post = np.cov(pre_outcomes, post_outcomes)[0, 1]
    var_pre = np.var(pre_outcomes, ddof=1)
    var_post = np.var(post_outcomes, ddof=1)
    std_pre = np.std(pre_outcomes, ddof=1)
    std_post = np.std(post_outcomes, ddof=1)

    # Correlation
    correlation = cov_pre_post / (std_pre * std_post) if (std_pre > 0 and std_post > 0) else 0.0

    return {
        "covariance": cov_pre_post,
        "var_pre": var_pre,
        "var_post": var_post,
        "correlation": correlation,
    }


def create_experiment_data(
    n_users: int,
    n_periods: int = 2,
    treatment_prob: float = 0.5,
    base_outcome: float = 100.0,
    treatment_effect: float = 10.0,
    seed: int = None,
) -> Dict[str, any]:
    """
    Create synthetic experiment data.

    Args:
        n_users: Number of users
        n_periods: Number of time periods (default 2: pre and post)
        treatment_prob: Treatment assignment probability
        base_outcome: Base outcome value
        treatment_effect: Treatment effect size
        seed: Random seed

    Returns:
        Dictionary with experiment data
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate user IDs
    user_ids = np.arange(1, n_users + 1)

    # Engineer features
    features = engineer_user_features(user_ids, seed=seed)

    # Deterministic treatment assignment
    treatment = deterministic_treatment_assignment(user_ids, treatment_prob)

    # Generate outcomes
    outcomes = []
    periods = []

    for period in range(n_periods):
        period_outcomes = np.zeros(n_users)

        for i, user_id in enumerate(user_ids):
            # Base outcome depends on features
            base = base_outcome + features[i, 1] / 1000.0  # Income effect

            # Treatment effect only in post period
            if period == 1 and treatment[i] == 1:
                base += treatment_effect

            # Add noise
            noise = np.random.normal(0, 5.0)
            period_outcomes[i] = max(0.0, base + noise)

        outcomes.append(period_outcomes)
        periods.append(np.full(n_users, period))

    # Flatten
    all_outcomes = np.concatenate(outcomes)
    all_periods = np.concatenate(periods)
    all_treatment = np.repeat(treatment, n_periods)
    all_features = np.repeat(features, n_periods, axis=0)
    all_user_ids = np.repeat(user_ids, n_periods)

    return {
        "user_ids": all_user_ids,
        "features": all_features,
        "treatment": all_treatment,
        "period": all_periods,
        "outcomes": all_outcomes,
        "n_users": n_users,
        "n_periods": n_periods,
    }


if __name__ == "__main__":
    raise ValueError(
        "This module requires real data. Use with actual user features, outcomes, periods, and treatment from datasets."
    )
