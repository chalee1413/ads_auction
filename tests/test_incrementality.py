"""
Unit tests for incrementality module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.incrementality import (
    calculate_incremental_revenue,
    calculate_iroas,
    calculate_lift,
    calculate_roas,
    test_significance as check_significance,
    compare_roas_iroas
)


class TestIncrementalRevenue:
    """Tests for incremental revenue calculation."""
    
    def test_basic_calculation(self):
        """Test basic incremental revenue calculation."""
        test_revenue = 1000.0
        control_revenue = 800.0
        expected = 200.0
        
        result = calculate_incremental_revenue(test_revenue, control_revenue)
        assert result == expected
    
    def test_negative_incremental(self):
        """Test case where test group has lower revenue."""
        test_revenue = 800.0
        control_revenue = 1000.0
        expected = -200.0
        
        result = calculate_incremental_revenue(test_revenue, control_revenue)
        assert result == expected
    
    def test_zero_control(self):
        """Test case with zero control revenue."""
        test_revenue = 1000.0
        control_revenue = 0.0
        expected = 1000.0
        
        result = calculate_incremental_revenue(test_revenue, control_revenue)
        assert result == expected


class TestIROAS:
    """Tests for iROAS calculation."""
    
    def test_basic_calculation(self):
        """Test basic iROAS calculation."""
        incremental_revenue = 200.0
        ad_spend = 100.0
        expected = 200.0  # 200%
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert abs(result - expected) < 1e-6
    
    def test_breakeven(self):
        """Test iROAS at breakeven (100%)."""
        incremental_revenue = 100.0
        ad_spend = 100.0
        expected = 100.0
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert abs(result - expected) < 1e-6
    
    def test_profitable(self):
        """Test profitable iROAS (>100%)."""
        incremental_revenue = 300.0
        ad_spend = 100.0
        expected = 300.0
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert abs(result - expected) < 1e-6
    
    def test_unprofitable(self):
        """Test unprofitable iROAS (<100%)."""
        incremental_revenue = 50.0
        ad_spend = 100.0
        expected = 50.0
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert abs(result - expected) < 1e-6
    
    def test_zero_spend(self):
        """Test edge case with zero spend."""
        incremental_revenue = 100.0
        ad_spend = 0.0
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert result == float('inf')
    
    def test_zero_spend_zero_revenue(self):
        """Test edge case with zero spend and zero revenue."""
        incremental_revenue = 0.0
        ad_spend = 0.0
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert result == 0.0
    
    def test_negative_incremental(self):
        """Test case with negative incremental revenue."""
        incremental_revenue = -50.0
        ad_spend = 100.0
        expected = -50.0
        
        result = calculate_iroas(incremental_revenue, ad_spend)
        assert abs(result - expected) < 1e-6


class TestLift:
    """Tests for lift calculation."""
    
    def test_basic_calculation(self):
        """Test basic lift calculation."""
        test_metric = 110.0
        control_metric = 100.0
        expected = 10.0  # 10% lift
        
        result = calculate_lift(test_metric, control_metric)
        assert abs(result - expected) < 1e-6
    
    def test_negative_lift(self):
        """Test case with negative lift."""
        test_metric = 90.0
        control_metric = 100.0
        expected = -10.0
        
        result = calculate_lift(test_metric, control_metric)
        assert abs(result - expected) < 1e-6
    
    def test_zero_control(self):
        """Test edge case with zero control metric."""
        test_metric = 100.0
        control_metric = 0.0
        
        result = calculate_lift(test_metric, control_metric)
        assert result == float('inf')
    
    def test_zero_control_zero_test(self):
        """Test edge case with both zero."""
        test_metric = 0.0
        control_metric = 0.0
        
        result = calculate_lift(test_metric, control_metric)
        assert result == 0.0


class TestROAS:
    """Tests for ROAS calculation."""
    
    def test_basic_calculation(self):
        """Test basic ROAS calculation."""
        attributed_revenue = 500.0
        ad_spend = 100.0
        expected = 500.0  # 500%
        
        result = calculate_roas(attributed_revenue, ad_spend)
        assert abs(result - expected) < 1e-6
    
    def test_zero_spend(self):
        """Test edge case with zero spend."""
        attributed_revenue = 100.0
        ad_spend = 0.0
        
        result = calculate_roas(attributed_revenue, ad_spend)
        assert result == float('inf')


class TestSignificance:
    """Tests for statistical significance testing."""
    
    def test_basic_test(self, sample_data):
        """Test basic significance test."""
        result = check_significance(
            sample_data['test_outcomes'],
            sample_data['control_outcomes']
        )
        
        assert 'p_value' in result
        assert 't_statistic' in result
        assert 'is_significant' in result
        assert 'test_mean' in result
        assert 'control_mean' in result
        assert 'difference' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'alpha' in result
        
        assert 0.0 <= result['p_value'] <= 1.0
        assert isinstance(result['is_significant'], bool)
        assert result['alpha'] == 0.05
    
    def test_significant_difference(self):
        """Test with clearly significant difference."""
        np.random.seed(42)
        test_values = np.random.normal(10.0, 1.0, 1000)
        control_values = np.random.normal(8.0, 1.0, 1000)
        
        result = check_significance(test_values, control_values)
        
        # Should be significant with large sample and clear difference
        assert result['is_significant'] == True
        assert result['p_value'] < 0.05
    
    def test_insignificant_difference(self):
        """Test with no significant difference."""
        np.random.seed(42)
        test_values = np.random.normal(10.0, 1.0, 100)
        control_values = np.random.normal(10.0, 1.0, 100)
        
        result = check_significance(test_values, control_values, alpha=0.05)
        
        # Should not be significant with same distribution
        assert result['is_significant'] == False
        assert result['p_value'] >= 0.05


class TestROASvsIROAS:
    """Tests for ROAS vs iROAS comparison."""
    
    def test_basic_comparison(self):
        """Test basic ROAS vs iROAS comparison."""
        test_revenue = 1000.0
        control_revenue = 800.0
        attributed_revenue = 1050.0  # Over-attributed
        ad_spend = 100.0
        
        result = compare_roas_iroas(
            test_revenue,
            control_revenue,
            attributed_revenue,
            ad_spend
        )
        
        assert 'roas' in result
        assert 'iroas' in result
        assert 'roas_iroas_gap' in result
        assert 'incremental_revenue' in result
        assert 'attributed_revenue' in result
        assert 'over_attribution' in result
        assert 'over_attribution_pct' in result
        
        # iROAS should be 200% (200 incremental / 100 spend)
        assert abs(result['iroas'] - 200.0) < 1e-6
        
        # ROAS should be 1050% (1050 attributed / 100 spend)
        assert abs(result['roas'] - 1050.0) < 1e-6
        
        # Gap should be positive (ROAS > iROAS)
        assert result['roas_iroas_gap'] > 0
        
        # Over-attribution should be 850 (1050 attributed - 200 incremental)
        # This represents how much more is attributed than is actually incremental
        assert abs(result['over_attribution'] - 850.0) < 1e-6

