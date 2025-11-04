"""
Off-Policy Evaluation: IPS and DR Methods

Implementation of off-policy evaluation methods for ad campaigns:
- Inverse Propensity Scoring (IPS)
- Doubly Robust (DR) evaluation

Used for evaluating new policies using data from old policies.

Citation: Referenced in conversation.md line 215, 260 for IPS/DR evaluation
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression, GradientBoostingRegressor


def inverse_propensity_scoring(
    outcomes: np.ndarray,
    actions: np.ndarray,
    propensity_scores: np.ndarray,
    evaluation_policy_scores: np.ndarray = None,
) -> Dict[str, float]:
    """
    Inverse Propensity Scoring (IPS) for off-policy evaluation.

    Formula: V_IPS = (1/n) * sum( (pi_e(a|x) / pi_b(a|x)) * Y )
    where pi_e is evaluation policy, pi_b is behavior policy

    Args:
        outcomes: Observed outcomes (rewards)
        actions: Actions taken (0 or 1)
        propensity_scores: Propensity scores from behavior policy
        evaluation_policy_scores: Propensity scores from evaluation policy (optional)

    Returns:
        Dictionary with IPS estimate and variance
    """
    n = len(outcomes)

    # If evaluation policy not provided, assume same as behavior
    if evaluation_policy_scores is None:
        evaluation_policy_scores = propensity_scores.copy()

    # Calculate importance weights
    # w(a|x) = pi_e(a|x) / pi_b(a|x)
    importance_weights = np.zeros(n)

    for i in range(n):
        action = actions[i]
        pi_b = propensity_scores[i] if action == 1 else (1 - propensity_scores[i])
        pi_e = evaluation_policy_scores[i] if action == 1 else (1 - evaluation_policy_scores[i])

        if pi_b > 0:
            importance_weights[i] = pi_e / pi_b
        else:
            importance_weights[i] = 0.0

    # IPS estimate
    ips_estimate = np.mean(importance_weights * outcomes)

    # Variance
    ips_variance = np.var(importance_weights * outcomes, ddof=1) / n
    ips_se = np.sqrt(ips_variance)

    return {
        "ips_estimate": ips_estimate,
        "ips_variance": ips_variance,
        "ips_se": ips_se,
        "importance_weights": importance_weights,
        "effective_sample_size": np.sum(importance_weights) ** 2 / np.sum(importance_weights**2),
    }


def doubly_robust_evaluation(
    outcomes: np.ndarray,
    actions: np.ndarray,
    propensity_scores: np.ndarray,
    outcome_model_predictions: np.ndarray,
    evaluation_policy_scores: np.ndarray = None,
) -> Dict[str, float]:
    """
    Doubly Robust (DR) evaluation method.

    Combines IPS with outcome model predictions. Consistent if either
    propensity model or outcome model is correct.

    Formula: V_DR = (1/n) * sum( mu(x,a) + (pi_e(a|x)/pi_b(a|x)) * (Y - mu(x,a)) )

    Args:
        outcomes: Observed outcomes
        actions: Actions taken
        propensity_scores: Propensity scores from behavior policy
        outcome_model_predictions: Predicted outcomes from outcome model
        evaluation_policy_scores: Propensity scores from evaluation policy

    Returns:
        Dictionary with DR estimate and variance
    """
    n = len(outcomes)

    if evaluation_policy_scores is None:
        evaluation_policy_scores = propensity_scores.copy()

    # Calculate importance weights
    importance_weights = np.zeros(n)
    for i in range(n):
        action = actions[i]
        pi_b = propensity_scores[i] if action == 1 else (1 - propensity_scores[i])
        pi_e = evaluation_policy_scores[i] if action == 1 else (1 - evaluation_policy_scores[i])

        if pi_b > 0:
            importance_weights[i] = pi_e / pi_b
        else:
            importance_weights[i] = 0.0

    # DR estimate
    # mu(x,a) is outcome model prediction
    mu_predictions = outcome_model_predictions

    # DR score
    dr_scores = mu_predictions + importance_weights * (outcomes - mu_predictions)
    dr_estimate = np.mean(dr_scores)

    # Variance
    dr_variance = np.var(dr_scores, ddof=1) / n
    dr_se = np.sqrt(dr_variance)

    return {
        "dr_estimate": dr_estimate,
        "dr_variance": dr_variance,
        "dr_se": dr_se,
        "dr_scores": dr_scores,
        "ips_component": np.mean(importance_weights * outcomes),
        "outcome_model_component": np.mean(mu_predictions),
    }


def estimate_propensity_scores(X: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Estimate propensity scores using logistic regression.

    Args:
        X: Feature matrix
        actions: Actions taken

    Returns:
        Propensity scores (probability of action=1 given X)
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, actions)
    propensity_scores = model.predict_proba(X)[:, 1]
    return propensity_scores


def estimate_outcome_model(
    X: np.ndarray, actions: np.ndarray, outcomes: np.ndarray
) -> Tuple[any, np.ndarray]:
    """
    Estimate outcome model (Q-function).

    Args:
        X: Feature matrix
        actions: Actions taken
        outcomes: Observed outcomes

    Returns:
        Tuple of (fitted model, predictions)
    """
    # Combine features with action indicator
    X_with_action = np.column_stack([X, actions])

    model = GradientBoostingRegressor()
    model.fit(X_with_action, outcomes)

    predictions = model.predict(X_with_action)

    return model, predictions


def off_policy_evaluation(
    X: np.ndarray,
    actions: np.ndarray,
    outcomes: np.ndarray,
    evaluation_policy_scores: np.ndarray = None,
) -> Dict[str, any]:
    """
    Complete off-policy evaluation pipeline.

    Args:
        X: Feature matrix
        actions: Actions taken under behavior policy
        outcomes: Observed outcomes
        evaluation_policy_scores: Propensity scores for evaluation policy

    Returns:
        Dictionary with IPS and DR estimates
    """
    # Estimate propensity scores for behavior policy
    propensity_scores = estimate_propensity_scores(X, actions)

    # Estimate outcome model
    outcome_model, outcome_predictions = estimate_outcome_model(X, actions, outcomes)

    # IPS evaluation
    ips_results = inverse_propensity_scoring(
        outcomes, actions, propensity_scores, evaluation_policy_scores
    )

    # DR evaluation
    dr_results = doubly_robust_evaluation(
        outcomes, actions, propensity_scores, outcome_predictions, evaluation_policy_scores
    )

    return {"ips": ips_results, "dr": dr_results, "propensity_scores": propensity_scores}


if __name__ == "__main__":
    raise ValueError(
        "This module requires real data. Use with actual features, actions, outcomes, and propensity scores from datasets."
    )
