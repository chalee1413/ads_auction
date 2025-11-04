"""
Ad Campaign Simulator

Realistic campaign simulation with user segments, conversion probabilities,
and performance tracking. Demonstrates attribution vs incremental conversion
separation and treatment effect heterogeneity.

References:
- Brand loyalists have low incrementality (conversation.md line 74)
- Attribution vs incremental conversion separation
"""

import numpy as np
from typing import Dict, List
from demo.incrementality import calculate_iroas, calculate_lift


class UserSegment:
    """User segment with conversion probabilities."""

    def __init__(self, name: str, base_conversion_rate: float, incrementality: float, size: int):
        """
        Initialize user segment.

        Args:
            name: Segment name
            base_conversion_rate: Base conversion rate (without ads)
            incrementality: Incremental lift from ads (0-1)
            size: Number of users in segment
        """
        self.name = name
        self.base_conversion_rate = base_conversion_rate
        self.incrementality = incrementality
        self.size = size

    def conversion_probability(self, with_ad: bool) -> float:
        """Calculate conversion probability with or without ad."""
        if with_ad:
            return self.base_conversion_rate * (1.0 + self.incrementality)
        else:
            return self.base_conversion_rate


class CampaignSimulator:
    """Simulate ad campaign with user segments."""

    def __init__(self, segments: List[UserSegment], seed: int = None):
        """Initialize campaign simulator."""
        if seed is not None:
            np.random.seed(seed)
        self.segments = segments
        self.total_users = sum(seg.size for seg in segments)

    def simulate(
        self,
        treatment_prob: float = 0.5,
        ad_spend_per_user: float = 1.0,
        revenue_per_conversion: float = 100.0,
        attribution_over_rate: float = 0.2,
    ) -> Dict[str, any]:
        """
        Simulate campaign with test/control groups.

        Args:
            treatment_prob: Probability of treatment assignment
            ad_spend_per_user: Ad spend per user exposed
            revenue_per_conversion: Revenue per conversion
            attribution_over_rate: Over-attribution rate (non-incremental conversions attributed)

        Returns:
            Dictionary with campaign results and metrics
        """
        # Simulate users
        all_treatments = []
        all_segments = []
        all_conversions = []
        all_revenues = []

        for segment in self.segments:
            # Treatment assignment
            n_treat = int(segment.size * treatment_prob)
            n_control = segment.size - n_treat

            # Conversions
            conversions_treat = np.random.binomial(1, segment.conversion_probability(True), n_treat)
            conversions_control = np.random.binomial(
                1, segment.conversion_probability(False), n_control
            )

            # Revenues
            revenues_treat = conversions_treat * revenue_per_conversion
            revenues_control = conversions_control * revenue_per_conversion

            # Track
            treatments = np.concatenate(
                [np.ones(n_treat, dtype=int), np.zeros(n_control, dtype=int)]
            )
            conversions = np.concatenate([conversions_treat, conversions_control])
            revenues = np.concatenate([revenues_treat, revenues_control])
            segments = [segment.name] * segment.size

            all_treatments.extend(treatments)
            all_conversions.extend(conversions)
            all_revenues.extend(revenues)
            all_segments.extend(segments)

        all_treatments = np.array(all_treatments)
        all_conversions = np.array(all_conversions)
        all_revenues = np.array(all_revenues)

        # Calculate metrics
        test_mask = all_treatments == 1
        control_mask = all_treatments == 0

        test_revenue = np.sum(all_revenues[test_mask])
        control_revenue = np.sum(all_revenues[control_mask])
        test_conversions = np.sum(all_conversions[test_mask])
        control_conversions = np.sum(all_conversions[control_mask])

        # Ad spend
        test_users = np.sum(test_mask)
        total_ad_spend = test_users * ad_spend_per_user

        # Incremental metrics
        incremental_revenue = test_revenue - control_revenue
        incremental_conversions = test_conversions - control_conversions

        # Calculate iROAS
        iroas = calculate_iroas(incremental_revenue, total_ad_spend)
        lift = calculate_lift(test_revenue, control_revenue)

        # Attribution (includes non-incremental)
        # Over-attribution: some non-incremental conversions are attributed
        non_incremental_control = control_revenue * attribution_over_rate
        attributed_revenue = incremental_revenue + non_incremental_control
        roas = (attributed_revenue / total_ad_spend * 100.0) if total_ad_spend > 0 else 0.0

        # Segment-level analysis
        segment_results = {}
        for segment in self.segments:
            seg_mask = np.array(all_segments) == segment.name
            seg_treat = all_treatments[seg_mask]
            seg_conv = all_conversions[seg_mask]
            seg_rev = all_revenues[seg_mask]

            seg_test_mask = seg_treat == 1
            seg_control_mask = seg_treat == 0

            seg_test_rev = np.sum(seg_rev[seg_test_mask])
            seg_control_rev = np.sum(seg_rev[seg_control_mask])
            seg_incremental = seg_test_rev - seg_control_rev

            segment_results[segment.name] = {
                "test_revenue": seg_test_rev,
                "control_revenue": seg_control_rev,
                "incremental_revenue": seg_incremental,
                "incrementality": segment.incrementality,
            }

        return {
            "test_revenue": test_revenue,
            "control_revenue": control_revenue,
            "incremental_revenue": incremental_revenue,
            "test_conversions": test_conversions,
            "control_conversions": control_conversions,
            "incremental_conversions": incremental_conversions,
            "total_ad_spend": total_ad_spend,
            "iroas": iroas,
            "roas": roas,
            "lift": lift,
            "attributed_revenue": attributed_revenue,
            "segment_results": segment_results,
            "test_users": test_users,
            "control_users": np.sum(control_mask),
        }


# Example usage
if __name__ == "__main__":
    # Create segments
    segments = [
        UserSegment("brand_loyalists", 0.15, 0.05, 2000),  # Low incrementality
        UserSegment("new_users", 0.03, 0.50, 3000),  # High incrementality
        UserSegment("occasional_users", 0.08, 0.20, 5000),  # Medium incrementality
    ]

    # Simulate campaign
    simulator = CampaignSimulator(segments, seed=42)
    results = simulator.simulate(
        treatment_prob=0.5,
        ad_spend_per_user=1.0,
        revenue_per_conversion=100.0,
        attribution_over_rate=0.2,
    )

    print("Campaign Simulation Results:")
    print(f"Test Revenue: ${results['test_revenue']:,.2f}")
    print(f"Control Revenue: ${results['control_revenue']:,.2f}")
    print(f"Incremental Revenue: ${results['incremental_revenue']:,.2f}")
    print(f"Attributed Revenue: ${results['attributed_revenue']:,.2f}")
    print(f"Total Ad Spend: ${results['total_ad_spend']:,.2f}")
    print(f"\niROAS: {results['iroas']:.2f}%")
    print(f"ROAS: {results['roas']:.2f}%")
    print(f"Lift: {results['lift']:.2f}%")
    print(f"\nROAS-iROAS Gap: {results['roas'] - results['iroas']:.2f} percentage points")

    print("\nSegment Results:")
    for seg_name, seg_stats in results["segment_results"].items():
        print(f"{seg_name}:")
        print(f"  Incremental Revenue: ${seg_stats['incremental_revenue']:,.2f}")
        print(f"  Incrementality: {seg_stats['incrementality']*100:.1f}%")
