"""
Unit tests for temporal cross-validation methods
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustcv.splitters.temporal import (
    TimeSeriesSplit, RollingWindowCV, ExpandingWindowCV,
    BlockedTimeSeriesCV, PurgedKFoldCV, CombinatorialPurgedCV,
    NestedTemporalCV
)


class TestTemporalMethods:
    """Test suite for temporal cross-validation methods"""
    
    @classmethod
    def setup_class(cls):
        """Set up temporal test data"""
        np.random.seed(42)
        
        # Create time series data
        n_samples = 200
        n_features = 10
        
        # Generate autocorrelated features
        cls.X = np.zeros((n_samples, n_features))
        for i in range(n_features):
            cls.X[0, i] = np.random.randn()
            for t in range(1, n_samples):
                cls.X[t, i] = 0.7 * cls.X[t-1, i] + np.random.randn()
        
        # Generate target with trend
        trend = np.linspace(0, 2, n_samples)
        signal = np.sum(cls.X[:, :3], axis=1) + trend
        cls.y = (signal > np.median(signal)).astype(int)
        
        # Create timestamps
        cls.timestamps = pd.date_range(
            start='2023-01-01',
            periods=n_samples,
            freq='D'
        )
        
        # Create DataFrame for testing
        cls.df = pd.DataFrame(cls.X)
        cls.df['y'] = cls.y
        cls.df['timestamp'] = cls.timestamps
    
    def test_time_series_split(self):
        """Test basic time series split"""
        cv = TimeSeriesSplit(n_splits=5)
        splits = list(cv.split(self.X))
        
        assert len(splits) == 5, "Should produce 5 splits"
        
        # Check that training size increases
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert all(train_sizes[i] < train_sizes[i+1] for i in range(len(train_sizes)-1)), \
            "Training size should increase with each split"
        
        # Check temporal order
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx), "Test should come after train"
    
    def test_rolling_window_cv(self):
        """Test rolling window cross-validation"""
        cv = RollingWindowCV(window_size=50, step_size=25)
        splits = list(cv.split(self.X))
        
        assert len(splits) > 0, "Should produce at least one split"
        
        # Check window properties
        for i, (train_idx, test_idx) in enumerate(splits):
            # Training window should be of fixed size (except possibly the first)
            if i > 0:
                assert len(train_idx) == 50, f"Window size should be 50, got {len(train_idx)}"
            
            # Test set should follow training set
            assert max(train_idx) < min(test_idx), "Test should follow train"
    
    def test_rolling_window_with_gap(self):
        """Test rolling window with gap between train and test"""
        cv = RollingWindowCV(window_size=50, step_size=25, gap=5)
        splits = list(cv.split(self.X))
        
        for train_idx, test_idx in splits:
            # Check gap exists
            gap_size = min(test_idx) - max(train_idx) - 1
            assert gap_size >= 5, f"Gap should be at least 5, got {gap_size}"
    
    def test_expanding_window_cv(self):
        """Test expanding window cross-validation"""
        cv = ExpandingWindowCV(initial_size=50, step_size=25)
        splits = list(cv.split(self.X))
        
        assert len(splits) > 0, "Should produce at least one split"
        
        # Check expanding property
        train_sizes = []
        for train_idx, test_idx in splits:
            train_sizes.append(len(train_idx))
            assert max(train_idx) < min(test_idx), "Test should follow train"
        
        # Training size should increase
        assert all(train_sizes[i] <= train_sizes[i+1] for i in range(len(train_sizes)-1)), \
            "Training size should expand or stay the same"
        
        # First window should be initial_size
        assert train_sizes[0] >= 50, "Initial window should be at least initial_size"
    
    def test_blocked_time_series_cv(self):
        """Test blocked time series cross-validation"""
        cv = BlockedTimeSeriesCV(n_splits=4)
        splits = list(cv.split(self.X))

        assert len(splits) == 4, "Should produce 4 splits"

        # Check temporal order within each split (train before test)
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx), "Test block should follow train blocks"

        # Check expanding training set (time series property)
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert all(train_sizes[i] <= train_sizes[i+1] for i in range(len(train_sizes)-1)), \
            "Training set should grow or stay the same over splits"
    
    def test_purged_kfold_cv(self):
        """Test purged K-fold cross-validation"""
        cv = PurgedKFoldCV(n_splits=5, purge_size=10)
        splits = list(cv.split(self.X, self.y))
        
        assert len(splits) == 5, "Should produce 5 splits"
        
        for train_idx, test_idx in splits:
            # Check purging: no training samples should be within purge_size of test samples
            for test_i in test_idx:
                for train_i in train_idx:
                    assert abs(test_i - train_i) > 10, \
                        f"Training sample {train_i} too close to test sample {test_i}"
    
    def test_purged_kfold_with_embargo(self):
        """Test purged K-fold with embargo"""
        cv = PurgedKFoldCV(n_splits=5, purge_size=10, embargo_size=5)
        splits = list(cv.split(self.X, self.y))
        
        for train_idx, test_idx in splits:
            # Check embargo: test samples should be embargoed after appearing
            test_set = set(test_idx)
            for test_i in test_idx:
                # Check that embargoed samples are not in training
                embargo_range = range(test_i + 1, min(test_i + 6, len(self.X)))
                for embargo_i in embargo_range:
                    assert embargo_i not in train_idx, \
                        f"Embargoed sample {embargo_i} found in training"
    
    def test_combinatorial_purged_cv(self):
        """Test combinatorial purged cross-validation"""
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_size=10)
        splits = list(cv.split(self.X, self.y))
        
        # Should produce C(n_splits, n_test_splits) combinations
        from math import comb
        expected_splits = comb(6, 2)  # C(6,2) = 15
        assert len(splits) <= expected_splits, f"Should produce at most {expected_splits} splits"
        
        for train_idx, test_idx in splits:
            # Check purging
            for test_i in test_idx:
                for train_i in train_idx:
                    assert abs(test_i - train_i) > 10, \
                        "Purging not applied correctly in CPCV"
    
    def test_nested_temporal_cv(self):
        """Test nested temporal cross-validation"""
        outer_cv = TimeSeriesSplit(n_splits=3)
        inner_cv = TimeSeriesSplit(n_splits=2)
        
        cv = NestedTemporalCV(outer_cv=outer_cv, inner_cv=inner_cv)
        
        # Get outer splits
        outer_splits = list(cv.split(self.X[:100], self.y[:100]))
        assert len(outer_splits) == 3, "Should have 3 outer splits"
        
        for train_idx, test_idx in outer_splits:
            # Ensure temporal order in outer split
            assert max(train_idx) < min(test_idx), "Outer test should follow train"
            
            # Check that inner CV can be applied
            X_train = self.X[:100][train_idx]
            y_train = self.y[:100][train_idx]
            inner_splits = list(inner_cv.split(X_train, y_train))
            
            assert len(inner_splits) == 2, "Should have 2 inner splits"
            
            for inner_train, inner_test in inner_splits:
                assert max(inner_train) < min(inner_test), \
                    "Inner test should follow inner train"
    
    def test_temporal_order_preservation(self):
        """Test that all methods preserve temporal order"""
        cv_methods = [
            TimeSeriesSplit(n_splits=3),
            RollingWindowCV(window_size=50, step_size=25),
            ExpandingWindowCV(initial_size=50, step_size=25),
            BlockedTimeSeriesCV(n_splits=3),
        ]
        
        for cv in cv_methods:
            splits = list(cv.split(self.X))
            for train_idx, test_idx in splits:
                # Training data should come before test data
                if len(train_idx) > 0 and len(test_idx) > 0:
                    assert max(train_idx) < min(test_idx), \
                        f"{cv.__class__.__name__} violates temporal order"
    
    def test_no_future_leakage(self):
        """Test that no future information leaks into training"""
        # Note: PurgedKFoldCV is a k-fold method that adds purging but doesn't
        # enforce forward-only temporal order, so it's not included here
        cv_methods = [
            TimeSeriesSplit(n_splits=3),
            RollingWindowCV(window_size=50, step_size=25),
            ExpandingWindowCV(initial_size=50, step_size=25),
            BlockedTimeSeriesCV(n_splits=3),
        ]
        
        for cv in cv_methods:
            splits = list(cv.split(self.X, self.y))
            for train_idx, test_idx in splits:
                # No test index should appear in or before training indices
                test_min = min(test_idx) if len(test_idx) > 0 else float('inf')
                train_max = max(train_idx) if len(train_idx) > 0 else -1
                
                assert train_max < test_min, \
                    f"{cv.__class__.__name__} has future leakage"
    
    def test_window_cv_edge_cases(self):
        """Test window-based CV methods with edge cases"""
        # Small dataset
        X_small = self.X[:30]
        
        # Rolling window larger than half the data
        cv1 = RollingWindowCV(window_size=20, step_size=5)
        splits1 = list(cv1.split(X_small))
        assert len(splits1) >= 1, "Should produce at least one split with small data"
        
        # Expanding window with large initial size
        cv2 = ExpandingWindowCV(initial_size=25, step_size=5)
        splits2 = list(cv2.split(X_small))
        assert len(splits2) >= 1, "Should handle large initial size"
    
    def test_purging_effectiveness(self):
        """Test that purging actually removes correlated samples"""
        # Create highly autocorrelated data
        n = 100
        X_auto = np.zeros((n, 1))
        X_auto[0] = np.random.randn()
        for i in range(1, n):
            X_auto[i] = 0.95 * X_auto[i-1] + 0.05 * np.random.randn()
        
        y_auto = (X_auto.flatten() > 0).astype(int)
        
        # Test with and without purging
        cv_no_purge = PurgedKFoldCV(n_splits=5, purge_size=0)
        cv_with_purge = PurgedKFoldCV(n_splits=5, purge_size=10)
        
        # The purged version should have fewer training samples
        splits_no_purge = list(cv_no_purge.split(X_auto, y_auto))
        splits_with_purge = list(cv_with_purge.split(X_auto, y_auto))
        
        for (train_np, test_np), (train_p, test_p) in zip(splits_no_purge, splits_with_purge):
            assert len(train_p) <= len(train_np), \
                "Purged version should have fewer or equal training samples"
    
    def test_cv_with_timestamps(self):
        """Test that temporal CV methods work with timestamp data"""
        # Create a DataFrame with timestamps
        df_test = pd.DataFrame({
            'feature1': self.X[:50, 0],
            'feature2': self.X[:50, 1],
            'target': self.y[:50],
            'timestamp': self.timestamps[:50]
        })
        
        # Methods should work with array data regardless of timestamps
        X_array = df_test[['feature1', 'feature2']].values
        y_array = df_test['target'].values
        
        cv = TimeSeriesSplit(n_splits=3)
        splits = list(cv.split(X_array, y_array))
        
        assert len(splits) == 3, "Should work with array data"
        
        # Verify temporal order is maintained
        for train_idx, test_idx in splits:
            train_times = df_test.iloc[train_idx]['timestamp']
            test_times = df_test.iloc[test_idx]['timestamp']
            
            assert train_times.max() < test_times.min(), \
                "Temporal order should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])