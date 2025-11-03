"""
CUPED (Conditional Unconfounded Pre-period Estimator) Implementation

CUPED reduces variance in A/B test analysis by adjusting for pre-period
covariates, allowing for more precise treatment effect estimates with
smaller sample sizes.

Algorithm based on Microsoft Research 2013 methodology.

Formula:
- Theta = Cov(pre, post) / Var(pre)
- Adjusted estimate = Treatment effect - Theta * (Pre-period diff)
- Variance reduction = 1 - (Theta^2 * Var(pre) / Var(post))

Citations:
- Microsoft Research 2013: Conditional Unconfounded Pre-period Estimator
- Conversation.md references CUPED methodology multiple times

References:
- Used in experimentation frameworks for variance reduction
- Particularly effective when pre-post correlation is high
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats


def calculate_theta(
    pre_values: np.ndarray, post_values: np.ndarray
) -> float:
    """
    Calculate Theta coefficient for CUPED adjustment.
    
    Theta represents the covariance between pre-period and post-period
    metrics, normalized by pre-period variance.
    
    Formula: Theta = Cov(pre, post) / Var(pre)
    
    Args:
        pre_values: Array of pre-period metric values
        post_values: Array of post-period metric values
        
    Returns:
        Theta coefficient for CUPED adjustment
    """
    if len(pre_values) != len(post_values):
        raise ValueError("Pre and post arrays must have same length")
    
    # Calculate covariance and variance
    cov_pre_post = np.cov(pre_values, post_values)[0, 1]
    var_pre = np.var(pre_values, ddof=1)
    
    # Avoid division by zero
    if var_pre == 0:
        return 0.0
    
    return cov_pre_post / var_pre


def cuped_adjustment(
    test_pre: np.ndarray,
    test_post: np.ndarray,
    control_pre: np.ndarray,
    control_post: np.ndarray
) -> Dict[str, float]:
    """
    Apply CUPED adjustment to test and control groups.
    
    Process:
    1. Calculate Theta from combined data
    2. Adjust test and control post-period means
    3. Calculate adjusted treatment effect
    4. Compute variance reduction
    
    DECISION RATIONALE:
    Why CUPED?
    1. Statistical Power: Reduces variance by 10-30%, enabling detection of smaller
       effects with same sample size (or same detection with smaller samples).
    2. Industry Standard: Microsoft Research 2013 methodology widely adopted in
       industry (Meta, Google, Microsoft experiments).
    3. Cost Efficiency: Smaller sample sizes = lower experimentation costs.
    
    Why Combined Theta?
    - More stable estimate than separate Thetas for test/control
    - Standard practice in CUPED literature
    - Assumes treatment doesn't affect pre-post correlation (reasonable assumption)
    
    Why Adjust Both Groups?
    - More powerful than adjusting just the difference
    - Removes baseline imbalance that inflates variance
    - Standard CUPED methodology
    
    When to Use:
    - A/B tests with pre-period baseline data
    - High pre-post correlation (user-level metrics, revenue)
    - Sample size constraints or cost-sensitive experiments
    
    When NOT to Use:
    - No pre-period data available
    - Pre-post correlation is low (limited benefit)
    - Already well-balanced randomized experiments
    
    References: Microsoft Research 2013 CUPED methodology
    
    Args:
        test_pre: Pre-period values for test group
        test_post: Post-period values for test group
        control_pre: Pre-period values for control group
        control_post: Post-period values for control group
        
    Returns:
        Dictionary with adjusted metrics and variance reduction stats
    """
    # Combine groups for Theta calculation
    # RATIONALE: More stable estimate using all data, standard CUPED practice
    all_pre = np.concatenate([test_pre, control_pre])
    all_post = np.concatenate([test_post, control_post])
    
    # Calculate Theta: Cov(pre, post) / Var(pre)
    # RATIONALE: Captures pre-post correlation normalized by pre-period variance
    theta = calculate_theta(all_pre, all_post)
    
    # Calculate unadjusted means
    test_post_mean = np.mean(test_post)
    control_post_mean = np.mean(control_post)
    test_pre_mean = np.mean(test_pre)
    control_pre_mean = np.mean(control_pre)
    
    # Calculate unadjusted treatment effect
    unadjusted_effect = test_post_mean - control_post_mean
    
    # CUPED adjustment: subtract Theta * pre-period difference
    pre_diff = test_pre_mean - control_pre_mean
    adjusted_effect = unadjusted_effect - theta * pre_diff
    
    # Calculate adjusted means
    test_adjusted_mean = test_post_mean - theta * (test_pre_mean - np.mean(all_pre))
    control_adjusted_mean = control_post_mean - theta * (control_pre_mean - np.mean(all_pre))
    
    # Variance calculations
    var_test_post = np.var(test_post, ddof=1)
    var_control_post = np.var(control_post, ddof=1)
    var_pre = np.var(all_pre, ddof=1)
    
    # Variance reduction factor
    # Variance reduction = 1 - (Theta^2 * Var(pre) / Var(post))
    # Expected reduction in variance of treatment effect estimate
    pooled_post_var = (var_test_post + var_control_post) / 2
    variance_reduction = 1.0 - (theta**2 * var_pre / pooled_post_var) if pooled_post_var > 0 else 0.0
    
    # Effective sample size increase
    effective_sample_size_multiplier = 1.0 / (1.0 - variance_reduction) if variance_reduction < 1.0 else 1.0
    
    # Calculate standard errors for unadjusted and adjusted
    n_test = len(test_post)
    n_control = len(control_post)
    se_unadjusted = np.sqrt(var_test_post / n_test + var_control_post / n_control)
    
    # Adjusted standard error (approximate)
    se_adjusted = se_unadjusted * np.sqrt(1.0 - variance_reduction) if variance_reduction > 0 else se_unadjusted
    
    return {
        'theta': theta,
        'unadjusted_effect': unadjusted_effect,
        'adjusted_effect': adjusted_effect,
        'unadjusted_test_mean': test_post_mean,
        'unadjusted_control_mean': control_post_mean,
        'adjusted_test_mean': test_adjusted_mean,
        'adjusted_control_mean': control_adjusted_mean,
        'variance_reduction': max(0.0, variance_reduction),
        'effective_sample_size_multiplier': effective_sample_size_multiplier,
        'se_unadjusted': se_unadjusted,
        'se_adjusted': se_adjusted,
        'pre_period_diff': pre_diff
    }


def statistical_significance(
    test_pre: np.ndarray,
    test_post: np.ndarray,
    control_pre: np.ndarray,
    control_post: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Calculate statistical significance for CUPED-adjusted treatment effect.
    
    Uses t-test on adjusted values to determine if treatment effect
    is statistically significant.
    
    Args:
        test_pre: Pre-period values for test group
        test_post: Post-period values for test group
        control_pre: Pre-period values for control group
        control_post: Post-period values for control group
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary with p-value, t-statistic, and significance flag
    """
    # Get CUPED adjustments
    cuped_results = cuped_adjustment(test_pre, test_post, control_pre, control_post)
    theta = cuped_results['theta']
    
    # Adjust individual values
    test_mean_pre = np.mean(test_pre)
    control_mean_pre = np.mean(control_pre)
    pooled_mean_pre = np.mean(np.concatenate([test_pre, control_pre]))
    
    # Adjust each observation
    test_adjusted = test_post - theta * (test_pre - pooled_mean_pre)
    control_adjusted = control_post - theta * (control_pre - pooled_mean_pre)
    
    # Two-sample t-test on adjusted values
    t_stat, p_value = stats.ttest_ind(test_adjusted, control_adjusted)
    
    is_significant = p_value < alpha
    
    # Confidence interval
    test_std_adj = np.std(test_adjusted, ddof=1)
    control_std_adj = np.std(control_adjusted, ddof=1)
    n_test = len(test_adjusted)
    n_control = len(control_adjusted)
    
    se_diff = np.sqrt((test_std_adj**2 / n_test) + (control_std_adj**2 / n_control))
    margin = stats.t.ppf(0.975, n_test + n_control - 2) * se_diff
    ci_lower = cuped_results['adjusted_effect'] - margin
    ci_upper = cuped_results['adjusted_effect'] + margin
    
    return {
        'p_value': p_value,
        't_statistic': t_stat,
        'is_significant': is_significant,
        'adjusted_effect': cuped_results['adjusted_effect'],
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'alpha': alpha,
        **cuped_results
    }


if __name__ == '__main__':
    raise ValueError("This module requires real data. Use with actual pre/post period outcomes from datasets.")

