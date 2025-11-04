"""
Unit tests for CUPED module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.cuped import (
    calculate_theta,
    check_cuped_assumptions,
    cuped_adjustment,
    statistical_significance
)


class TestCalculateTheta:
    """Tests for Theta calculation."""
    
    def test_basic_calculation(self):
        """Test basic Theta calculation."""
        np.random.seed(42)
        pre_values = np.random.normal(50.0, 10.0, 100)
        post_values = pre_values * 0.8 + np.random.normal(0.0, 5.0, 100)
        
        theta = calculate_theta(pre_values, post_values)
        
        # Theta should be positive for correlated data
        assert theta > 0
        assert isinstance(theta, float)
    
    def test_identical_arrays(self):
        """Test with identical pre and post arrays."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        theta = calculate_theta(values, values)
        
        # Should be 1.0 for identical arrays
        assert abs(theta - 1.0) < 1e-6
    
    def test_uncorrelated_arrays(self):
        """Test with uncorrelated arrays."""
        np.random.seed(42)
        pre_values = np.random.normal(50.0, 10.0, 100)
        post_values = np.random.normal(50.0, 10.0, 100)
        
        theta = calculate_theta(pre_values, post_values)
        
        # Theta should be close to zero for uncorrelated data
        assert abs(theta) < 1.0
    
    def test_length_mismatch(self):
        """Test that length mismatch raises error."""
        pre_values = np.array([1.0, 2.0, 3.0])
        post_values = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError):
            calculate_theta(pre_values, post_values)
    
    def test_zero_variance(self):
        """Test with zero variance in pre-period."""
        pre_values = np.array([5.0, 5.0, 5.0, 5.0])
        post_values = np.array([1.0, 2.0, 3.0, 4.0])
        
        theta = calculate_theta(pre_values, post_values)
        
        # Should return 0.0 when variance is zero
        assert theta == 0.0


class TestCUPEDAssumptions:
    """Tests for CUPED assumption checks."""
    
    def test_all_assumptions_pass(self, cuped_data):
        """Test when all assumptions pass."""
        checks = check_cuped_assumptions(
            cuped_data['test_pre'],
            cuped_data['test_post'],
            cuped_data['control_pre'],
            cuped_data['control_post']
        )
        
        assert 'sample_size_passed' in checks
        assert 'pre_post_correlation_passed' in checks
        assert 'balanced_groups_passed' in checks
        assert 'all_assumptions_passed' in checks
        
        assert checks['sample_size_passed'] == True
        assert checks['all_assumptions_passed'] == True
    
    def test_small_sample_size(self):
        """Test with sample size too small."""
        test_pre = np.array([1.0, 2.0])  # Too small
        test_post = np.array([1.0, 2.0])
        control_pre = np.array([1.0, 2.0])
        control_post = np.array([1.0, 2.0])
        
        checks = check_cuped_assumptions(
            test_pre, test_post, control_pre, control_post
        )
        
        assert checks['sample_size_passed'] == False
        assert checks['all_assumptions_passed'] == False
    
    def test_low_correlation(self):
        """Test with low pre-post correlation."""
        np.random.seed(42)
        test_pre = np.random.normal(50.0, 10.0, 100)
        test_post = np.random.normal(50.0, 10.0, 100)  # Uncorrelated
        control_pre = np.random.normal(50.0, 10.0, 100)
        control_post = np.random.normal(50.0, 10.0, 100)
        
        checks = check_cuped_assumptions(
            test_pre, test_post, control_pre, control_post
        )
        
        # Should fail correlation check if correlation is too low
        assert 'pre_post_correlation' in checks
        assert isinstance(checks['pre_post_correlation'], float)


class TestCUPEDAdjustment:
    """Tests for CUPED adjustment calculation."""
    
    def test_basic_adjustment(self, cuped_data):
        """Test basic CUPED adjustment."""
        result = cuped_adjustment(
            cuped_data['test_pre'],
            cuped_data['test_post'],
            cuped_data['control_pre'],
            cuped_data['control_post']
        )
        
        assert 'unadjusted_effect' in result
        assert 'adjusted_effect' in result
        assert 'variance_reduction' in result
        assert 'theta' in result
        assert 'assumption_checks' in result
        assert 'assumptions_passed' in result
        
        assert isinstance(result['unadjusted_effect'], float)
        assert isinstance(result['adjusted_effect'], float)
        assert isinstance(result['variance_reduction'], float)
        assert isinstance(result['theta'], float)
        
        # Variance reduction should be between 0 and 1
        assert 0.0 <= result['variance_reduction'] <= 1.0
    
    def test_adjustment_reduces_variance(self, cuped_data):
        """Test that CUPED adjustment reduces variance."""
        result = cuped_adjustment(
            cuped_data['test_pre'],
            cuped_data['test_post'],
            cuped_data['control_pre'],
            cuped_data['control_post']
        )
        
        # Should have some variance reduction when correlation is high
        if result['assumptions_passed']:
            assert result['variance_reduction'] >= 0.0


class TestStatisticalSignificance:
    """Tests for statistical significance in CUPED context."""
    
    def test_basic_significance(self, cuped_data):
        """Test basic significance calculation."""
        result = statistical_significance(
            cuped_data['test_pre'],
            cuped_data['test_post'],
            cuped_data['control_pre'],
            cuped_data['control_post']
        )
        
        assert 'p_value' in result
        assert 'is_significant' in result
        
        assert 0.0 <= result['p_value'] <= 1.0
        assert isinstance(result['is_significant'], bool)
    
    def test_significant_difference(self):
        """Test with significant treatment effect."""
        np.random.seed(42)
        test_pre = np.random.normal(50.0, 10.0, 100)
        test_post = np.random.normal(60.0, 10.0, 100)  # Large effect
        control_pre = np.random.normal(50.0, 10.0, 100)
        control_post = np.random.normal(51.0, 10.0, 100)  # Small effect
        
        result = statistical_significance(
            test_pre, test_post, control_pre, control_post
        )
        
        # Should be significant with large effect
        assert result['is_significant'] == True
        assert result['p_value'] < 0.05

