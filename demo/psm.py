"""
Propensity Score Matching (PSM) for Observational Studies

Propensity Score Matching algorithm for estimating treatment effects
in observational studies where randomization is not possible.

Algorithm:
1. Estimate propensity scores using logistic regression
2. Match treated units to control units based on propensity scores
3. Calculate Average Treatment Effect on the Treated (ATT)

Citation: Referenced in conversation.md line 134 for observational/causal inference
"""

import numpy as np
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from scipy import stats


def estimate_propensity_scores(X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """
    Estimate propensity scores using logistic regression.

    Args:
        X: Feature matrix
        treatment: Treatment indicator (0 or 1)

    Returns:
        Propensity scores (probability of treatment given X)
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    return propensity_scores


def nearest_neighbor_matching(
    propensity_treated: np.ndarray, propensity_control: np.ndarray, caliper: float = 0.1
) -> Dict[int, int]:
    """
    Perform 1-to-1 nearest neighbor matching with caliper.

    Args:
        propensity_treated: Propensity scores for treated units
        propensity_control: Propensity scores for control units
        caliper: Maximum allowed distance for matching

    Returns:
        Dictionary mapping treated indices to matched control indices
    """
    matches = {}

    for i, p_treat in enumerate(propensity_treated):
        # Calculate distances to all control units
        distances = np.abs(propensity_control - p_treat)

        # Find nearest neighbor
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        # Apply caliper
        if min_distance <= caliper:
            matches[i] = int(min_idx)

    return matches


def calculate_att(
    outcome_treated: np.ndarray, outcome_control: np.ndarray, matches: Dict[int, int]
) -> Dict[str, float]:
    """
    Calculate Average Treatment Effect on the Treated (ATT).

    ATT = E[Y(1) - Y(0) | T = 1]

    Args:
        outcome_treated: Outcomes for treated units
        outcome_control: Outcomes for control units
        matches: Dictionary mapping treated to control indices

    Returns:
        Dictionary with ATT and related statistics
    """
    if len(matches) == 0:
        return {"att": 0.0, "se": 0.0, "n_matched": 0}

    # Get matched pairs
    treated_outcomes = []
    control_outcomes = []

    for treat_idx, control_idx in matches.items():
        treated_outcomes.append(outcome_treated[treat_idx])
        control_outcomes.append(outcome_control[control_idx])

    treated_outcomes = np.array(treated_outcomes)
    control_outcomes = np.array(control_outcomes)

    # Calculate ATT
    differences = treated_outcomes - control_outcomes
    att = np.mean(differences)
    se = stats.sem(differences)

    # 95% confidence interval
    n = len(differences)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = att - t_crit * se
    ci_upper = att + t_crit * se

    return {
        "att": att,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_matched": n,
        "differences": differences,
    }


def balance_diagnostics(
    X_treated: np.ndarray,
    X_control: np.ndarray,
    X_matched_control: np.ndarray,
    feature_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate balance diagnostics (standardized mean differences).

    Args:
        X_treated: Features for treated group
        X_control: Features for original control group
        X_matched_control: Features for matched control group
        feature_names: Names of features

    Returns:
        Dictionary with standardized mean differences for each feature
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_treated.shape[1])]

    results = {}

    for i, name in enumerate(feature_names):
        # Original control group
        mean_treat = np.mean(X_treated[:, i])
        mean_control = np.mean(X_control[:, i])
        std_control = np.std(X_control[:, i])

        if std_control == 0:
            std_control = 1.0

        smd_original = (mean_treat - mean_control) / std_control

        # Matched control group
        mean_matched = np.mean(X_matched_control[:, i])
        smd_matched = (mean_treat - mean_matched) / std_control

        results[name] = {
            "smd_original": smd_original,
            "smd_matched": smd_matched,
            "improvement": smd_original - smd_matched,
        }

    return results


def propensity_score_matching(
    X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, caliper: float = 0.1
) -> Dict[str, any]:
    """
    Complete PSM analysis pipeline.

    Args:
        X: Feature matrix
        treatment: Treatment indicator
        outcome: Outcome values
        caliper: Matching caliper

    Returns:
        Dictionary with matching results, ATT, and balance diagnostics
    """
    # Split data
    X_treated = X[treatment == 1]
    X_control = X[treatment == 0]
    outcome_treated = outcome[treatment == 1]
    outcome_control = outcome[treatment == 0]

    # Estimate propensity scores
    propensity_scores = estimate_propensity_scores(X, treatment)
    propensity_treated = propensity_scores[treatment == 1]
    propensity_control = propensity_scores[treatment == 0]

    # Perform matching
    matches = nearest_neighbor_matching(propensity_treated, propensity_control, caliper)

    # Calculate ATT
    att_results = calculate_att(outcome_treated, outcome_control, matches)

    # Get matched control features
    matched_control_indices = list(matches.values())
    X_matched_control = (
        X_control[matched_control_indices] if len(matched_control_indices) > 0 else X_control[:0]
    )

    # Balance diagnostics
    balance = (
        balance_diagnostics(X_treated, X_control, X_matched_control)
        if len(X_matched_control) > 0
        else {}
    )

    return {
        "propensity_scores": propensity_scores,
        "matches": matches,
        "att": att_results,
        "balance": balance,
        "n_treated": len(X_treated),
        "n_control": len(X_control),
        "n_matched": len(matches),
    }


if __name__ == "__main__":
    raise ValueError(
        "This module requires real data. Use with actual features, treatment, and outcomes from datasets."
    )
