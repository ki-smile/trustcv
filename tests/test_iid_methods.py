"""
Unit tests for I.I.D. cross-validation methods
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustcv.splitters.iid import (
    HoldOut, KFoldMedical, StratifiedKFoldMedical,
    RepeatedKFold, LOOCV, LPOCV, BootstrapValidation,
    MonteCarloCV, NestedCV
)


class TestIIDMethods:
    """Test suite for I.I.D. cross-validation methods"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data"""
        np.random.seed(42)
        cls.X, cls.y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        cls.X_small, cls.y_small = cls.X[:30], cls.y[:30]
    
    def test_holdout_split(self):
        """Test Hold-Out split"""
        cv = HoldOut(test_size=0.3, random_state=42)
        splits = list(cv.split(self.X, self.y))
        
        assert len(splits) == 1, "Hold-out should produce exactly 1 split"
        
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 70, "Training set should have 70% of samples"
        assert len(test_idx) == 30, "Test set should have 30% of samples"
        assert len(set(train_idx) & set(test_idx)) == 0, "No overlap between train and test"
        
    def test_holdout_stratified(self):
        """Test stratified Hold-Out split"""
        cv = HoldOut(test_size=0.3, stratify=True, random_state=42)
        splits = list(cv.split(self.X, self.y))
        
        train_idx, test_idx = splits[0]
        train_ratio = np.mean(self.y[train_idx])
        test_ratio = np.mean(self.y[test_idx])
        
        # Check stratification (ratios should be similar)
        assert abs(train_ratio - test_ratio) < 0.1, "Stratification not working properly"
    
    def test_kfold_medical(self):
        """Test K-Fold Medical implementation"""
        cv = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(self.X, self.y))
        
        assert len(splits) == 5, "Should produce 5 splits"
        
        # Check fold sizes
        for train_idx, test_idx in splits:
            assert len(train_idx) == 80, "Each training fold should have 80 samples"
            assert len(test_idx) == 20, "Each test fold should have 20 samples"
            assert len(set(train_idx) & set(test_idx)) == 0, "No overlap in fold"
        
        # Check that all samples are used
        all_test_idx = []
        for _, test_idx in splits:
            all_test_idx.extend(test_idx)
        assert len(set(all_test_idx)) == 100, "All samples should be tested exactly once"
    
    def test_stratified_kfold_medical(self):
        """Test Stratified K-Fold Medical"""
        cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(self.X, self.y))
        
        assert len(splits) == 5, "Should produce 5 splits"
        
        # Check stratification in each fold
        overall_ratio = np.mean(self.y)
        for train_idx, test_idx in splits:
            test_ratio = np.mean(self.y[test_idx])
            assert abs(test_ratio - overall_ratio) < 0.15, "Stratification not maintained"
    
    def test_repeated_kfold(self):
        """Test Repeated K-Fold"""
        cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
        splits = list(cv.split(self.X, self.y))
        
        assert len(splits) == 6, "Should produce n_splits * n_repeats splits"
        
        # Check that repeats are different
        first_repeat_test = [splits[i][1] for i in range(3)]
        second_repeat_test = [splits[i][1] for i in range(3, 6)]
        
        # At least one fold should be different between repeats
        different = False
        for f1, f2 in zip(first_repeat_test, second_repeat_test):
            if not np.array_equal(sorted(f1), sorted(f2)):
                different = True
                break
        assert different, "Repeats should produce different splits"
    
    def test_loocv(self):
        """Test Leave-One-Out CV"""
        cv = LOOCV()
        splits = list(cv.split(self.X_small, self.y_small))
        
        assert len(splits) == 30, "LOOCV should produce n_samples splits"
        
        for train_idx, test_idx in splits:
            assert len(train_idx) == 29, "Training set should have n-1 samples"
            assert len(test_idx) == 1, "Test set should have exactly 1 sample"
    
    def test_lpocv(self):
        """Test Leave-P-Out CV"""
        cv = LPOCV(p=2)
        splits = list(cv.split(self.X_small[:10], self.y_small[:10]))
        
        # For n=10, p=2: C(10,2) = 45 combinations
        assert len(splits) == 45, "LPOCV should produce C(n,p) splits"
        
        for train_idx, test_idx in splits:
            assert len(train_idx) == 8, "Training set should have n-p samples"
            assert len(test_idx) == 2, "Test set should have exactly p samples"
    
    def test_bootstrap_validation(self):
        """Test Bootstrap Validation"""
        cv = BootstrapValidation(n_iterations=10, estimator='.632', random_state=42)
        splits = list(cv.split(self.X_small, self.y_small))

        assert len(splits) == 10, "Should produce n_iterations bootstrap samples"
        
        # Check bootstrap properties
        for train_idx, test_idx in splits:
            assert len(train_idx) == 30, "Bootstrap sample should be same size as original"
            assert len(test_idx) > 0, "OOB samples should exist"
            assert len(set(train_idx) | set(test_idx)) <= 30, "All indices from original data"
            
            # Check for duplicates in bootstrap sample (characteristic of bootstrap)
            assert len(train_idx) != len(set(train_idx)), "Bootstrap should have duplicates"
    
    def test_bootstrap_632_estimator(self):
        """Test Bootstrap .632 estimator calculation"""
        cv = BootstrapValidation(n_iterations=100, estimator='.632', random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Calculate score using the bootstrap CV
        scores = []
        for train_idx, test_idx in cv.split(self.X_small, self.y_small):
            if len(test_idx) > 0:
                model.fit(self.X_small[train_idx], self.y_small[train_idx])
                score = model.score(self.X_small[test_idx], self.y_small[test_idx])
                scores.append(score)
        
        assert len(scores) > 0, "Should produce valid scores"
        assert 0 <= np.mean(scores) <= 1, "Scores should be valid probabilities"
    
    def test_monte_carlo_cv(self):
        """Test Monte Carlo Cross-Validation"""
        cv = MonteCarloCV(n_iterations=5, test_size=0.2, random_state=42)
        splits = list(cv.split(self.X, self.y))

        assert len(splits) == 5, "Should produce n_iterations random splits"
        
        for train_idx, test_idx in splits:
            assert len(train_idx) == 80, "Training set should have 80% of samples"
            assert len(test_idx) == 20, "Test set should have 20% of samples"
            assert len(set(train_idx) & set(test_idx)) == 0, "No overlap between train and test"
        
        # Check that splits are different (Monte Carlo property)
        test_sets = [set(splits[i][1]) for i in range(5)]
        assert len(test_sets) == len(set(map(frozenset, test_sets))), "Splits should be random"
    
    def test_nested_cv_structure(self):
        """Test Nested CV structure"""
        outer_cv = KFoldMedical(n_splits=3, random_state=42)
        inner_cv = KFoldMedical(n_splits=2, random_state=42)
        cv = NestedCV(outer_cv=outer_cv, inner_cv=inner_cv)

        # Get outer splits from outer_cv
        outer_splits = list(cv.outer_cv.split(self.X_small, self.y_small))
        assert len(outer_splits) == 3, "Should have 3 outer splits"

        # Check inner CV can be applied to outer training data
        for train_idx, test_idx in outer_splits:
            X_train = self.X_small[train_idx]
            y_train = self.y_small[train_idx]

            inner_splits = list(cv.inner_cv.split(X_train, y_train))
            assert len(inner_splits) == 2, "Should have 2 inner splits"

            # Verify no leakage between outer test and inner data
            assert len(set(test_idx) & set(train_idx)) == 0, "No outer test in inner CV"
    
    def test_nested_cv_with_model(self):
        """Test Nested CV with actual model training"""
        from sklearn.model_selection import GridSearchCV
        
        outer_cv = KFoldMedical(n_splits=3)
        inner_cv = KFoldMedical(n_splits=2)
        
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [5, 10], 'max_depth': [2, 3]}
        
        outer_scores = []
        for train_idx, test_idx in outer_cv.split(self.X_small, self.y_small):
            X_train, X_test = self.X_small[train_idx], self.X_small[test_idx]
            y_train, y_test = self.y_small[train_idx], self.y_small[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=inner_cv)
            grid_search.fit(X_train, y_train)
            
            # Evaluate on outer test set
            score = grid_search.score(X_test, y_test)
            outer_scores.append(score)
        
        assert len(outer_scores) == 3, "Should have scores for each outer fold"
        assert all(0 <= s <= 1 for s in outer_scores), "Scores should be valid"
    
    def test_cv_with_different_data_sizes(self):
        """Test CV methods with different data sizes"""
        sizes = [20, 50, 100, 200]
        
        for size in sizes:
            X_test = self.X[:size]
            y_test = self.y[:size]
            
            # Test methods that should work with any size
            cv_methods = [
                HoldOut(test_size=0.3),
                MonteCarloCV(n_iterations=3, test_size=0.2),
            ]
            
            if size >= 50:
                cv_methods.append(KFoldMedical(n_splits=5))
                cv_methods.append(StratifiedKFoldMedical(n_splits=5))
            
            for cv in cv_methods:
                splits = list(cv.split(X_test, y_test))
                assert len(splits) > 0, f"{cv.__class__.__name__} failed with size {size}"
    
    def test_reproducibility(self):
        """Test that methods with random_state are reproducible"""
        cv_methods = [
            HoldOut(test_size=0.3, random_state=42),
            KFoldMedical(n_splits=5, shuffle=True, random_state=42),
            MonteCarloCV(n_iterations=5, test_size=0.2, random_state=42),
            BootstrapValidation(n_iterations=10, random_state=42),
        ]
        
        for cv in cv_methods:
            splits1 = list(cv.split(self.X, self.y))
            splits2 = list(cv.split(self.X, self.y))
            
            for s1, s2 in zip(splits1, splits2):
                assert np.array_equal(s1[0], s2[0]), f"{cv.__class__.__name__} not reproducible"
                assert np.array_equal(s1[1], s2[1]), f"{cv.__class__.__name__} not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])