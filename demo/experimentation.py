"""
Experimentation Framework: RCT, A/B Tests, Geo Experiments

Implementation of experimentation frameworks including:
- Randomized Controlled Trials (RCT)
- A/B test allocation and randomization
- Geo-holdout experiments
- Pre-period balance validation
- Power analysis and MDE calculation

References:
- ACM WWW 2021 paper on double-blind designs (conversation.md line 106, 133)
- Standard RCT methodology
"""

import numpy as np
from typing import Dict
from scipy import stats
from scipy.stats import norm


def randomize_assignment(
    n_samples: int, treatment_prob: float = 0.5, seed: int = None
) -> np.ndarray:
    """
    Randomize assignment to treatment and control groups.

    Args:
        n_samples: Number of samples to randomize
        treatment_prob: Probability of treatment assignment (default 0.5)
        seed: Random seed for reproducibility

    Returns:
        Array of treatment assignments (0 or 1)
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.binomial(1, treatment_prob, n_samples)


def balanced_assignment(
    n_samples: int, treatment_ratio: float = 0.5, seed: int = None
) -> np.ndarray:
    """
    Create balanced assignment ensuring exact treatment ratio.

    Args:
        n_samples: Number of samples
        treatment_ratio: Desired treatment ratio (default 0.5)
        seed: Random seed for reproducibility

    Returns:
        Array of treatment assignments
    """
    if seed is not None:
        np.random.seed(seed)

    n_treat = int(n_samples * treatment_ratio)
    assignment = np.zeros(n_samples)
    treat_indices = np.random.choice(n_samples, n_treat, replace=False)
    assignment[treat_indices] = 1

    return assignment.astype(int)


def pre_period_balance_test(
    X: np.ndarray, treatment: np.ndarray, alpha: float = 0.05
) -> Dict[str, any]:
    """
    Test pre-period balance between treatment and control groups.

    Checks if groups are statistically equivalent before treatment.

    Args:
        X: Feature matrix or pre-period outcomes
        treatment: Treatment assignment
        alpha: Significance level

    Returns:
        Dictionary with balance test results
    """
    X_treat = X[treatment == 1]
    X_control = X[treatment == 0]

    results = {}

    # For each feature/outcome
    if len(X.shape) > 1:
        for i in range(X.shape[1]):
            treat_mean = np.mean(X_treat[:, i])
            control_mean = np.mean(X_control[:, i])

            # T-test
            t_stat, p_value = stats.ttest_ind(X_treat[:, i], X_control[:, i])

            # Standardized mean difference
            pooled_std = np.sqrt(
                (np.var(X_treat[:, i], ddof=1) + np.var(X_control[:, i], ddof=1)) / 2
            )
            smd = (treat_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0

            results[f"feature_{i}"] = {
                "treat_mean": treat_mean,
                "control_mean": control_mean,
                "difference": treat_mean - control_mean,
                "smd": smd,
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_balanced": p_value >= alpha,
            }
    else:
        # Single outcome
        treat_mean = np.mean(X_treat)
        control_mean = np.mean(X_control)

        t_stat, p_value = stats.ttest_ind(X_treat, X_control)

        pooled_std = np.sqrt((np.var(X_treat, ddof=1) + np.var(X_control, ddof=1)) / 2)
        smd = (treat_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0

        results["outcome"] = {
            "treat_mean": treat_mean,
            "control_mean": control_mean,
            "difference": treat_mean - control_mean,
            "smd": smd,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_balanced": p_value >= alpha,
        }

    return results


def calculate_power(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
    treatment_ratio: float = 0.5,
    std: float = 1.0,
) -> float:
    """
    Calculate statistical power for given effect size and sample size.

    Power = P(reject H0 | H1 is true)

    Args:
        effect_size: Expected treatment effect (standardized)
        n_samples: Total sample size
        alpha: Type I error rate (default 0.05)
        treatment_ratio: Ratio of treatment group
        std: Standard deviation (default 1.0)

    Returns:
        Statistical power (0 to 1)
    """
    n_treat = int(n_samples * treatment_ratio)
    n_control = n_samples - n_treat

    # Standard error of difference
    se_diff = std * np.sqrt(1.0 / n_treat + 1.0 / n_control)

    # Critical value
    z_alpha = norm.ppf(1 - alpha / 2)

    # Non-centrality parameter
    ncp = effect_size / se_diff

    # Power calculation
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

    return power


def calculate_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    treatment_ratio: float = 0.5,
    std: float = 1.0,
) -> int:
    """
    Calculate required sample size for given effect size and power.

    Args:
        effect_size: Minimum detectable effect size (standardized)
        power: Desired statistical power (default 0.8)
        alpha: Type I error rate (default 0.05)
        treatment_ratio: Ratio of treatment group
        std: Standard deviation

    Returns:
        Required total sample size
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Formula for sample size
    n_per_group = 2 * ((z_alpha + z_beta) ** 2 * std**2) / (effect_size**2)
    n_total = int(n_per_group / treatment_ratio)

    return n_total


def calculate_mde(
    n_samples: int,
    power: float = 0.8,
    alpha: float = 0.05,
    treatment_ratio: float = 0.5,
    std: float = 1.0,
) -> float:
    """
    Calculate Minimum Detectable Effect (MDE).

    MDE is the smallest effect size detectable with given sample size and power.

    Args:
        n_samples: Total sample size
        power: Statistical power (default 0.8)
        alpha: Type I error rate (default 0.05)
        treatment_ratio: Ratio of treatment group
        std: Standard deviation

    Returns:
        Minimum detectable effect size
    """
    n_treat = int(n_samples * treatment_ratio)
    n_control = n_samples - n_treat

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Standard error
    se_diff = std * np.sqrt(1.0 / n_treat + 1.0 / n_control)

    # MDE
    mde = (z_alpha + z_beta) * se_diff

    return mde


def geo_holdout_experiment(
    geo_ids: np.ndarray, treatment_geo_ratio: float = 0.5, seed: int = None
) -> Dict[str, np.ndarray]:
    """
    Design geo-holdout experiment.

    Randomly assigns geographic regions to treatment and control.

    Args:
        geo_ids: Array of geographic identifiers
        treatment_geo_ratio: Ratio of geos to assign to treatment
        seed: Random seed

    Returns:
        Dictionary with geo-level treatment assignments
    """
    if seed is not None:
        np.random.seed(seed)

    unique_geos = np.unique(geo_ids)
    n_geos = len(unique_geos)
    n_treat_geos = int(n_geos * treatment_geo_ratio)

    # Randomly assign geos
    treat_geo_indices = np.random.choice(n_geos, n_treat_geos, replace=False)
    treat_geos = unique_geos[treat_geo_indices]

    # Create assignment vector
    geo_treatment = np.zeros(len(geo_ids))
    for i, geo_id in enumerate(geo_ids):
        if geo_id in treat_geos:
            geo_treatment[i] = 1

    return {
        "treatment": geo_treatment.astype(int),
        "treated_geos": treat_geos,
        "control_geos": unique_geos[~np.isin(unique_geos, treat_geos)],
    }


def rct_design(
    n_samples: int,
    features: np.ndarray = None,
    treatment_prob: float = 0.5,
    balance_check: bool = True,
    seed: int = None,
) -> Dict[str, any]:
    """
    Design a complete Randomized Controlled Trial.

    Args:
        n_samples: Number of participants
        features: Pre-treatment features for balance check
        treatment_prob: Treatment assignment probability
        balance_check: Whether to check pre-period balance
        seed: Random seed

    Returns:
        Dictionary with RCT design including assignments and balance checks
    """
    # Randomize assignment
    assignment = randomize_assignment(n_samples, treatment_prob, seed)

    results = {
        "assignment": assignment,
        "n_treated": np.sum(assignment),
        "n_control": np.sum(1 - assignment),
        "treatment_ratio": np.mean(assignment),
    }

    # Balance check if features provided
    if balance_check and features is not None:
        balance_results = pre_period_balance_test(features, assignment)
        results["balance_check"] = balance_results

        # Overall balance assessment
        all_balanced = all(result["is_balanced"] for result in balance_results.values())
        results["is_balanced"] = all_balanced

    return results


# Example usage
if __name__ == "__main__":
    # Example: Design RCT with power analysis
    np.random.seed(42)

    # Sample size calculation
    effect_size = 0.2  # Small effect
    n_required = calculate_sample_size(effect_size, power=0.8)
    print(f"Required sample size for effect={effect_size}: {n_required}")

    # MDE calculation
    n_available = 1000
    mde = calculate_mde(n_available, power=0.8)
    print(f"MDE with n={n_available}: {mde:.4f}")

    # Power calculation
    power = calculate_power(effect_size, n_required, alpha=0.05)
    print(f"Power with effect={effect_size}, n={n_required}: {power:.4f}")

    # RCT design
    n_samples = 1000
    features = np.random.normal(0, 1, (n_samples, 3))
    rct = rct_design(n_samples, features, balance_check=True)

    print("\nRCT Design:")
    print(f"Treated: {rct['n_treated']}, Control: {rct['n_control']}")
    print(f"Balance check passed: {rct.get('is_balanced', True)}")
