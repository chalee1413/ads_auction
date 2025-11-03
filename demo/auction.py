"""
Auction and Bidding Algorithms

Implementation of auction mechanisms and bidding strategies:
- Second-price auction (VCG-style)
- iROAS-based bidding strategy
- ROAS vs iROAS comparison
- Bid shading for first-price auctions
- Budget pacing

References:
- IMM guide on iROAS-based bidding (conversation.md line 21, 59, 72)
- Auction theory: VCG, GSP, second-price auctions
"""

import numpy as np
from typing import Dict, List, Tuple


def second_price_auction(
    bids: List[float]
) -> Dict[str, any]:
    """
    Second-price auction clearing.
    
    Winner pays second-highest bid.
    
    Args:
        bids: List of bid values from all bidders
        
    Returns:
        Dictionary with winner index, winning bid, payment
    """
    bids_array = np.array(bids)
    
    if len(bids_array) == 0:
        return {'winner': None, 'winning_bid': 0.0, 'payment': 0.0}
    
    # Winner is highest bidder
    winner_idx = np.argmax(bids_array)
    winning_bid = bids_array[winner_idx]
    
    # Payment is second-highest bid
    sorted_bids = np.sort(bids_array)[::-1]
    payment = sorted_bids[1] if len(sorted_bids) > 1 else sorted_bids[0]
    
    return {
        'winner': int(winner_idx),
        'winning_bid': float(winning_bid),
        'payment': float(payment)
    }


def iroas_based_bidding(
    incremental_value: float,
    observed_iroas: float = None,
    target_iroas: float = 100.0
) -> float:
    """
    iROAS-based bidding strategy for second-price auctions.
    
    Strategy: Bid to observed incremental value in second-price settings.
    Per IMM guide (conversation.md line 72-73).
    
    Args:
        incremental_value: Estimated incremental value of impression
        observed_iroas: Observed iROAS (optional, for optimization)
        target_iroas: Target iROAS (default 100%)
        
    Returns:
        Optimal bid value
    """
    if observed_iroas is not None:
        # Adjust bid based on iROAS performance
        adjustment_factor = target_iroas / observed_iroas if observed_iroas > 0 else 1.0
        bid = incremental_value * adjustment_factor
    else:
        # Bid to incremental value (optimal in second-price)
        bid = incremental_value
    
    return max(0.0, bid)


def roas_vs_iroas_bidding(
    attributed_value: float,
    incremental_value: float,
    observed_roas: float = None,
    observed_iroas: float = None,
    target_iroas: float = 100.0
) -> Dict[str, float]:
    """
    Compare ROAS-based vs iROAS-based bidding.
    
    Args:
        attributed_value: Attributed value (includes non-incremental)
        incremental_value: True incremental value
        observed_roas: Observed ROAS performance
        observed_iroas: Observed iROAS performance
        target_iroas: Target iROAS
        
    Returns:
        Dictionary with bid values and expected performance
    """
    # ROAS-based bid (includes non-incremental)
    roas_bid = attributed_value
    
    # iROAS-based bid (only incremental)
    iroas_bid = iroas_based_bidding(incremental_value, observed_iroas, target_iroas)
    
    # Expected over-bidding with ROAS
    over_bid = roas_bid - iroas_bid
    over_bid_pct = (over_bid / roas_bid * 100.0) if roas_bid > 0 else 0.0
    
    return {
        'roas_bid': roas_bid,
        'iroas_bid': iroas_bid,
        'over_bid': over_bid,
        'over_bid_pct': over_bid_pct,
        'attributed_value': attributed_value,
        'incremental_value': incremental_value
    }


def bid_shading(
    valuation: float,
    competition_level: float,
    auction_type: str = 'first_price'
) -> float:
    """
    Bid shading for first-price auctions.
    
    Optimal strategy in first-price auctions is to bid below valuation.
    
    Args:
        valuation: True value of impression
        competition_level: Level of competition (0-1, higher = more competition)
        auction_type: Type of auction ('first_price' or 'second_price')
        
    Returns:
        Shaded bid value
    """
    if auction_type == 'first_price':
        # Shade bid based on competition
        # More competition = less shading needed
        shading_factor = 0.7 + 0.2 * competition_level  # Between 0.7 and 0.9
        bid = valuation * shading_factor
    else:
        # Second-price: bid at valuation
        bid = valuation
    
    return max(0.0, bid)


def budget_pacing(
    total_budget: float,
    time_remaining: float,
    spent: float,
    impressions_remaining: int,
    avg_cpc: float
) -> Dict[str, float]:
    """
    Budget pacing algorithm.
    
    Calculates optimal bid adjustment to pace budget spending.
    
    Args:
        total_budget: Total budget available
        time_remaining: Fraction of time remaining (0-1)
        spent: Amount spent so far
        impressions_remaining: Estimated impressions remaining
        avg_cpc: Average cost per click
        
    Returns:
        Dictionary with pacing metrics and bid adjustment
    """
    remaining_budget = total_budget - spent
    
    # Ideal spending rate
    ideal_spend_rate = total_budget / 1.0  # Per unit time
    current_spend_rate = spent / (1.0 - time_remaining) if time_remaining < 1.0 else spent
    
    # Pacing ratio
    pacing_ratio = ideal_spend_rate / current_spend_rate if current_spend_rate > 0 else 1.0
    
    # Budget constraint: can't exceed remaining budget
    max_spend = remaining_budget
    max_impressions = max_spend / avg_cpc if avg_cpc > 0 else impressions_remaining
    
    # Bid adjustment factor
    if pacing_ratio < 0.8:
        # Spending too fast: reduce bids
        bid_adjustment = 0.8
    elif pacing_ratio > 1.2:
        # Spending too slow: increase bids
        bid_adjustment = 1.2
    else:
        # On track
        bid_adjustment = 1.0
    
    return {
        'remaining_budget': remaining_budget,
        'pacing_ratio': pacing_ratio,
        'bid_adjustment': bid_adjustment,
        'max_impressions': max_impressions,
        'is_on_pace': 0.8 <= pacing_ratio <= 1.2
    }


def auction_simulation(
    n_auctions: int,
    our_incremental_values: np.ndarray,
    competition_bids: np.ndarray,
    iroas_strategy: bool = True,
    seed: int = None
) -> Dict[str, any]:
    """
    Simulate multiple auctions with iROAS or ROAS bidding.
    
    Args:
        n_auctions: Number of auctions
        our_incremental_values: Our incremental value estimates
        competition_bids: Competing bids for each auction
        iroas_strategy: Whether to use iROAS-based bidding (True) or ROAS (False)
        seed: Random seed
        
    Returns:
        Dictionary with auction results and performance metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    if iroas_strategy:
        our_bids = iroas_based_bidding(our_incremental_values)
    else:
        # ROAS bidding: bid to attributed value (assuming 20% over-attribution)
        attributed_values = our_incremental_values * 1.2
        our_bids = attributed_values
    
    wins = 0
    total_payment = 0.0
    total_incremental_value = 0.0
    
    for i in range(n_auctions):
        # Auction clearing
        all_bids = [our_bids[i]] + [comp_bid for comp_bid in competition_bids[i]]
        result = second_price_auction(all_bids)
        
        if result['winner'] == 0:  # We won
            wins += 1
            total_payment += result['payment']
            total_incremental_value += our_incremental_values[i]
    
    # Calculate metrics
    win_rate = wins / n_auctions if n_auctions > 0 else 0.0
    avg_cpc = total_payment / wins if wins > 0 else 0.0
    
    # Calculate iROAS
    iroas = (total_incremental_value / total_payment * 100.0) if total_payment > 0 else 0.0
    
    return {
        'wins': wins,
        'win_rate': win_rate,
        'total_payment': total_payment,
        'total_incremental_value': total_incremental_value,
        'avg_cpc': avg_cpc,
        'iroas': iroas,
        'strategy': 'iROAS' if iroas_strategy else 'ROAS'
    }


# Example usage
if __name__ == '__main__':
    # Example: iROAS vs ROAS bidding comparison
    np.random.seed(42)
    n_auctions = 1000
    
    # Incremental values
    incremental_values = np.random.normal(2.0, 0.5, n_auctions)
    incremental_values = np.clip(incremental_values, 0.1, 5.0)
    
    # Competition bids (2-3 bidders per auction)
    competition_bids = [
        np.random.normal(1.5, 0.4, np.random.randint(1, 4))
        for _ in range(n_auctions)
    ]
    
    # Compare strategies
    iroas_results = auction_simulation(
        n_auctions, incremental_values, competition_bids, iroas_strategy=True
    )
    roas_results = auction_simulation(
        n_auctions, incremental_values, competition_bids, iroas_strategy=False
    )
    
    print("Auction Simulation Results:")
    print(f"\niROAS Strategy:")
    print(f"  Wins: {iroas_results['wins']}, Win Rate: {iroas_results['win_rate']:.2%}")
    print(f"  Total Payment: ${iroas_results['total_payment']:.2f}")
    print(f"  iROAS: {iroas_results['iroas']:.2f}%")
    
    print(f"\nROAS Strategy:")
    print(f"  Wins: {roas_results['wins']}, Win Rate: {roas_results['win_rate']:.2%}")
    print(f"  Total Payment: ${roas_results['total_payment']:.2f}")
    print(f"  iROAS: {roas_results['iroas']:.2f}%")
    
    print(f"\nDifference:")
    print(f"  Payment Difference: ${roas_results['total_payment'] - iroas_results['total_payment']:.2f}")
    print(f"  iROAS Difference: {roas_results['iroas'] - iroas_results['iroas']:.2f} percentage points")

