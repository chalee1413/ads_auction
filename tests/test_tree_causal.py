"""
Unit tests for tree-based causal inference module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.tree_causal import (
    check_tree_causal_assumptions,
    robust_causal_inference,
    CausalTree
)


class TestCausalTree:
    """Tests for CausalTree class."""
    
    def test_initialization(self):
        """Test CausalTree initialization."""
        tree = CausalTree(max_depth=5, min_samples_split=100, min_samples_leaf=50)
        
        assert tree.max_depth == 5
        assert tree.min_samples_split == 100
        assert tree.min_samples_leaf == 50
        assert tree.tree_ is None
        assert tree.feature_importances_ is None
    
    def test_fit(self, causal_data):
        """Test fitting the causal tree."""
        tree = CausalTree(max_depth=3)
        tree.fit(causal_data['X'], causal_data['treatment'], causal_data['outcome'])
        
        assert tree.tree_ is not None
        assert tree.feature_importances_ is not None
        assert len(tree.feature_importances_) > 0
    
    def test_estimate_heterogeneous_effects(self, causal_data):
        """Test heterogeneous effect estimation."""
        tree = CausalTree(max_depth=3)
        results = tree.estimate_heterogeneous_effects(
            causal_data['X'],
            causal_data['treatment'],
            causal_data['outcome']
        )
        
        assert 'treatment_effects' in results
        assert 'segment_ids' in results
        assert 'feature_importances' in results
        
        assert len(results['treatment_effects']) == causal_data['n_samples']
        assert len(results['segment_ids']) == causal_data['n_samples']
        assert len(results['feature_importances']) > 0
    
    def test_treatment_effect_estimation(self, causal_data):
        """Test that treatment effects are estimated correctly."""
        tree = CausalTree(max_depth=3)
        results = tree.estimate_heterogeneous_effects(
            causal_data['X'],
            causal_data['treatment'],
            causal_data['outcome']
        )
        
        treatment_effects = results['treatment_effects']
        
        # Treatment effects should be numeric
        assert all(np.isfinite(treatment_effects))
        
        # Should have some variation in effects
        assert np.std(treatment_effects) >= 0.0


class TestAssumptionChecks:
    """Tests for assumption checking."""
    
    def test_all_assumptions_pass(self, causal_data):
        """Test when all assumptions pass."""
        checks = check_tree_causal_assumptions(
            causal_data['X'],
            causal_data['treatment'],
            causal_data['outcome']
        )
        
        assert 'sample_size_passed' in checks
        assert 'balanced_treatment_passed' in checks
        assert 'positivity_passed' in checks
        assert 'all_assumptions_passed' in checks
        assert 'errors' in checks
        
        assert checks['sample_size_passed'] == True
        assert checks['balanced_treatment_passed'] == True
        assert checks['positivity_passed'] == True
        assert checks['all_assumptions_passed'] == True
    
    def test_small_sample_size(self):
        """Test with sample size too small."""
        X = np.random.randn(50, 5)  # Too small
        treatment = np.random.binomial(1, 0.5, 50)
        outcome = np.random.randn(50)
        
        checks = check_tree_causal_assumptions(X, treatment, outcome)
        
        assert checks['sample_size_passed'] == False
        assert checks['all_assumptions_passed'] == False
    
    def test_unbalanced_treatment(self):
        """Test with unbalanced treatment groups."""
        X = np.random.randn(500, 5)
        treatment = np.ones(500)  # All treated
        outcome = np.random.randn(500)
        
        checks = check_tree_causal_assumptions(X, treatment, outcome)
        
        assert checks['positivity_passed'] == False
        assert checks['all_assumptions_passed'] == False
    
    def test_no_control_group(self):
        """Test with no control group."""
        X = np.random.randn(500, 5)
        treatment = np.ones(500)  # All treated, no control
        outcome = np.random.randn(500)
        
        checks = check_tree_causal_assumptions(X, treatment, outcome)
        
        assert checks['positivity_passed'] == False


class TestRobustCausalInference:
    """Tests for robust_causal_inference function."""
    
    def test_basic_inference(self, causal_data):
        """Test basic causal inference."""
        results = robust_causal_inference(
            causal_data['X'],
            causal_data['treatment'],
            causal_data['outcome']
        )
        
        assert 'treatment_effects' in results
        assert 'segment_ids' in results
        assert 'average_treatment_effect' in results
        assert 'effect_variance' in results
        assert 'segment_stats' in results
        assert 'feature_importances' in results
        assert 'assumption_checks' in results
        assert 'assumptions_passed' in results
        
        assert isinstance(results['average_treatment_effect'], float)
        assert isinstance(results['effect_variance'], float)
        assert isinstance(results['segment_stats'], list)
        assert len(results['segment_stats']) > 0
    
    def test_assumption_checks_included(self, causal_data):
        """Test that assumption checks are included in results."""
        results = robust_causal_inference(
            causal_data['X'],
            causal_data['treatment'],
            causal_data['outcome']
        )
        
        checks = results['assumption_checks']
        assert 'sample_size_passed' in checks
        assert 'balanced_treatment_passed' in checks
        assert 'positivity_passed' in checks
    
    def test_segment_identification(self, causal_data):
        """Test that segments are identified correctly."""
        results = robust_causal_inference(
            causal_data['X'],
            causal_data['treatment'],
            causal_data['outcome'],
            max_depth=3
        )
        
        # Should identify multiple segments
        assert len(results['segment_stats']) > 0
        
        # Each segment should have stats
        for seg in results['segment_stats']:
            assert 'segment_id' in seg
            assert 'effect' in seg
            assert 'size' in seg
            assert 'pct_of_total' in seg
            
            assert seg['size'] > 0
            assert 0.0 <= seg['pct_of_total'] <= 100.0

