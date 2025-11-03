"""
Ghost Bidding Simulation

Implementation of ghost bidding methodology for incrementality measurement.

Ghost bidding: Withhold impressions that would have won for a randomized
subset to form a live control group during active auctions.

Citation: Moloco blog post on proving incremental ROAS (conversation.md line 30, 66, 73)

Algorithm:
- For each eligible impression, determine if it would win auction
- Randomly assign to control group (withhold impression)
- Track conversions for both test and control groups
- Calculate incremental lift using RCT framework
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy import stats


def ghost_bidding_simulation(
    impressions: int,
    bids: np.ndarray,
    competition_bids: np.ndarray,
    conversion_rates: np.ndarray,
    control_ratio: float = 0.1,
    seed: int = None
) -> Dict[str, any]:
    """
    Simulate ghost bidding to create live control group.
    
    Args:
        impressions: Number of impressions to process
        bids: Our bid values for each impression
        competition_bids: Competing bid values for each impression
        conversion_rates: Conversion probability for each impression if shown ad
        control_ratio: Ratio of impressions to withhold for control
        seed: Random seed
        
    Returns:
        Dictionary with test/control results and incremental metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Determine which impressions would win
    winning_mask = bids > competition_bids
    
    # Randomly assign winners to control (ghost bid)
    eligible_for_control = np.where(winning_mask)[0]
    n_control = int(len(eligible_for_control) * control_ratio)
    control_indices = np.random.choice(
        eligible_for_control, n_control, replace=False
    )
    control_mask = np.zeros(impressions, dtype=bool)
    control_mask[control_indices] = True
    
    # Treatment group: winners not in control
    treatment_mask = winning_mask & ~control_mask
    
    # Simulate conversions
    conversions = np.random.binomial(1, conversion_rates, impressions)
    
    # Track conversions by group
    treatment_conversions = np.sum(conversions[treatment_mask])
    control_conversions = np.sum(conversions[control_mask])
    
    # Calculate metrics
    treatment_impressions = np.sum(treatment_mask)
    control_impressions = np.sum(control_mask)
    
    treatment_cvr = treatment_conversions / treatment_impressions if treatment_impressions > 0 else 0.0
    control_cvr = control_conversions / control_impressions if control_impressions > 0 else 0.0
    
    # Incremental lift
    incremental_conversions = treatment_conversions - control_conversions
    lift = (treatment_cvr - control_cvr) / control_cvr if control_cvr > 0 else 0.0
    
    # Statistical significance
    if treatment_impressions > 0 and control_impressions > 0:
        treatment_binary = conversions[treatment_mask].astype(int)
        control_binary = conversions[control_mask].astype(int)
        
        if len(treatment_binary) > 0 and len(control_binary) > 0:
            t_stat, p_value = stats.ttest_ind(treatment_binary, control_binary)
        else:
            t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = 0.0, 1.0
    
    return {
        'treatment_impressions': treatment_impressions,
        'control_impressions': control_impressions,
        'treatment_conversions': treatment_conversions,
        'control_conversions': control_conversions,
        'treatment_cvr': treatment_cvr,
        'control_cvr': control_cvr,
        'incremental_conversions': incremental_conversions,
        'lift': lift,
        'p_value': p_value,
        't_statistic': t_stat,
        'is_significant': p_value < 0.05
    }


def rct_with_ghost_bidding(
    impressions: int,
    bids: np.ndarray,
    competition_bids: np.ndarray,
    base_conversion_rate: float,
    treatment_effect: float,
    control_ratio: float = 0.1,
    seed: int = None
) -> Dict[str, any]:
    """
    Full RCT with ghost bidding for incrementality measurement.
    
    Args:
        impressions: Number of impressions
        bids: Our bid values
        competition_bids: Competing bids
        base_conversion_rate: Base conversion rate (control)
        treatment_effect: Treatment effect on conversion rate
        control_ratio: Ratio for control group
        seed: Random seed
        
    Returns:
        Dictionary with RCT results and incremental metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Conversion rates: base + treatment effect for eligible
    winning_mask = bids > competition_bids
    conversion_rates = np.zeros(impressions)
    conversion_rates[winning_mask] = base_conversion_rate + treatment_effect
    conversion_rates[~winning_mask] = 0.0  # No chance if don't win
    
    # Ghost bidding simulation
    results = ghost_bidding_simulation(
        impressions, bids, competition_bids, conversion_rates,
        control_ratio, seed
    )
    
    # Intent-to-treat analysis
    # All eligible impressions (regardless of assignment)
    eligible_mask = winning_mask
    eligible_conversions = np.sum(
        np.random.binomial(1, conversion_rates[eligible_mask], np.sum(eligible_mask))
    )
    
    # Control: those withheld + naturally didn't win
    control_total = np.sum(~winning_mask) + results['control_impressions']
    control_total_conversions = np.sum(
        np.random.binomial(1, base_conversion_rate, control_total)
    )
    
    # ITT estimate
    itt_effect = (
        results['treatment_cvr'] - 
        (control_total_conversions / control_total if control_total > 0 else 0.0)
    )
    
    return {
        **results,
        'itt_effect': itt_effect,
        'eligible_impressions': np.sum(eligible_mask),
        'eligible_conversions': eligible_conversions
    }


if __name__ == '__main__':
    raise ValueError("This module requires real data. Use with actual bids, competition bids, and conversion rates from datasets.")

