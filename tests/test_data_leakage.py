"""
Unit tests for data leakage detection across all CV methods
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustcv.splitters.iid import (
    KFoldMedical, StratifiedKFoldMedical, BootstrapValidation
)
from trustcv.splitters.temporal import (
    TimeSeriesSplit, PurgedKFoldCV, CombinatorialPurgedCV
)
from trustcv.splitters.grouped import (
    GroupKFoldMedical, LeaveOneGroupOut
)
from trustcv.splitters.spatial import (
    SpatialBlockCV, BufferedSpatialCV
)
from trustcv.checkers.leakage import DataLeakageChecker
from trustcv.checkers.balance import BalanceChecker


class TestDataLeakage:
    """Comprehensive tests for data leakage detection"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data with known leakage patterns"""
        np.random.seed(42)
        
        # Create dataset with multiple levels of structure
        n_samples = 300
        n_features = 10
        n_patients = 50
        n_hospitals = 5
        
        # Basic features
        cls.X = np.random.randn(n_samples, n_features)
        
        # Patient IDs (multiple measurements per patient)
        cls.patient_ids = np.repeat(np.arange(n_patients), n_samples // n_patients)
        
        # Hospital IDs (patients grouped in hospitals)
        cls.hospital_ids = np.repeat(np.arange(n_hospitals), n_samples // n_hospitals)
        
        # Temporal structure
        cls.timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        # Spatial structure
        cls.coordinates = np.random.uniform(0, 100, (n_samples, 2))
        
        # Target with dependencies
        cls.y = np.zeros(n_samples)
        for i in range(n_samples):
            # Target depends on patient, hospital, and time
            patient_effect = cls.patient_ids[i] * 0.1
            hospital_effect = cls.hospital_ids[i] * 0.2
            temporal_effect = i * 0.01
            
            cls.y[i] = (patient_effect + hospital_effect + temporal_effect + 
                       np.random.randn() > 2)
        
        cls.y = cls.y.astype(int)
        
        # Initialize checkers
        cls.leakage_checker = DataLeakageChecker()
        cls.balance_checker = BalanceChecker()
    
    def test_patient_level_leakage(self):
        """Test detection of patient-level data leakage"""
        # Standard K-fold (WRONG - has leakage)
        cv_leaky = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
        
        # Group K-fold (CORRECT - no leakage)
        cv_correct = GroupKFoldMedical(n_splits=5)
        
        # Check leaky CV
        leaky_has_leakage = False
        for train_idx, test_idx in cv_leaky.split(self.X, self.y):
            train_patients = self.patient_ids[train_idx]
            test_patients = self.patient_ids[test_idx]
            
            # Check for patient overlap
            if len(set(train_patients) & set(test_patients)) > 0:
                leaky_has_leakage = True
                break
        
        assert leaky_has_leakage, "Standard K-fold should have patient leakage"
        
        # Check correct CV
        correct_has_leakage = False
        for train_idx, test_idx in cv_correct.split(self.X, self.y, groups=self.patient_ids):
            train_patients = self.patient_ids[train_idx]
            test_patients = self.patient_ids[test_idx]
            
            # Check for patient overlap
            if len(set(train_patients) & set(test_patients)) > 0:
                correct_has_leakage = True
                break
        
        assert not correct_has_leakage, "Group K-fold should not have patient leakage"
    
    def test_temporal_leakage(self):
        """Test detection of temporal data leakage"""
        # Create temporal features with lookahead
        X_temporal = self.X.copy()
        
        # Add future information (LEAKAGE!)
        for i in range(len(X_temporal) - 1):
            X_temporal[i, 0] = self.y[i + 1]  # Next timestep's target
        
        # Standard K-fold (has temporal leakage)
        cv_leaky = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
        
        # Time series split (no temporal leakage)
        cv_correct = TimeSeriesSplit(n_splits=5)
        
        # Check for temporal leakage in standard K-fold
        for train_idx, test_idx in cv_leaky.split(X_temporal, self.y):
            train_times = self.timestamps[train_idx]
            test_times = self.timestamps[test_idx]
            
            # Check if any training data comes after test data
            if train_times.max() > test_times.min():
                temporal_leakage = True
                break
        else:
            temporal_leakage = False
        
        assert temporal_leakage, "Standard K-fold should have temporal leakage"
        
        # Check time series split
        for train_idx, test_idx in cv_correct.split(X_temporal, self.y):
            train_times = self.timestamps[train_idx]
            test_times = self.timestamps[test_idx]
            
            # Training should always come before test
            assert train_times.max() < test_times.min(), \
                "Time series split should prevent temporal leakage"
    
    def test_spatial_leakage(self):
        """Test detection of spatial data leakage"""
        # Create spatially autocorrelated features
        X_spatial = self.X.copy()
        
        # Make features depend on location
        for i in range(len(X_spatial)):
            X_spatial[i, 0] = np.sin(self.coordinates[i, 0] / 10)
            X_spatial[i, 1] = np.cos(self.coordinates[i, 1] / 10)
        
        # Standard K-fold (ignores spatial structure)
        cv_no_spatial = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
        
        # Spatial block CV (respects spatial structure)
        cv_spatial = SpatialBlockCV(n_splits=4)
        
        # Check for spatial proximity in standard CV
        min_distances = []
        for train_idx, test_idx in cv_no_spatial.split(X_spatial, self.y):
            train_coords = self.coordinates[train_idx]
            test_coords = self.coordinates[test_idx]
            
            # Find minimum distance between train and test
            for test_coord in test_coords[:10]:  # Sample for speed
                distances = [
                    np.linalg.norm(test_coord - train_coord)
                    for train_coord in train_coords[:10]
                ]
                if distances:
                    min_distances.append(min(distances))
        
        avg_min_dist_no_spatial = np.mean(min_distances)
        
        # Check spatial CV
        min_distances_spatial = []
        for train_idx, test_idx in cv_spatial.split(X_spatial, self.y, 
                                                   coordinates=self.coordinates):
            train_coords = self.coordinates[train_idx]
            test_coords = self.coordinates[test_idx]
            
            # Find minimum distance between train and test
            for test_coord in test_coords[:10]:
                distances = [
                    np.linalg.norm(test_coord - train_coord)
                    for train_coord in train_coords[:10]
                ]
                if distances:
                    min_distances_spatial.append(min(distances))
        
        avg_min_dist_spatial = np.mean(min_distances_spatial) if min_distances_spatial else 0
        
        # Spatial CV should maintain larger distances
        assert avg_min_dist_spatial > avg_min_dist_no_spatial * 0.8, \
            "Spatial CV should maintain spatial separation"
    
    def test_hierarchical_leakage(self):
        """Test detection of hierarchical data leakage"""
        # Patients within hospitals - need to respect both levels
        
        # Check patient-only grouping (ignores hospital structure)
        cv_patient_only = GroupKFoldMedical(n_splits=5)
        
        # Check if hospitals are mixed
        hospital_mixing = False
        for train_idx, test_idx in cv_patient_only.split(self.X, self.y, 
                                                        groups=self.patient_ids):
            train_hospitals = set(self.hospital_ids[train_idx])
            test_hospitals = set(self.hospital_ids[test_idx])
            
            # Check if same hospital appears in both
            if len(train_hospitals & test_hospitals) > 0:
                hospital_mixing = True
                break
        
        assert hospital_mixing, \
            "Patient-only grouping should allow hospital mixing"
        
        # Check hospital-level grouping (respects hierarchy)
        cv_hospital = GroupKFoldMedical(n_splits=3)
        
        hospital_leakage = False
        for train_idx, test_idx in cv_hospital.split(self.X, self.y, 
                                                    groups=self.hospital_ids):
            train_hospitals = set(self.hospital_ids[train_idx])
            test_hospitals = set(self.hospital_ids[test_idx])
            
            # No hospital should appear in both
            if len(train_hospitals & test_hospitals) > 0:
                hospital_leakage = True
                break
        
        assert not hospital_leakage, \
            "Hospital-level grouping should prevent hospital leakage"
    
    def test_feature_leakage(self):
        """Test detection of feature-based leakage"""
        # Create features that directly contain target information
        X_leaky = self.X.copy()
        
        # Add target as a feature (obvious leakage)
        X_leaky[:, 0] = self.y
        
        # Add highly correlated feature (subtle leakage)
        X_leaky[:, 1] = self.y + np.random.randn(len(self.y)) * 0.1
        
        # Check correlation with target
        correlations = []
        for i in range(X_leaky.shape[1]):
            corr = np.corrcoef(X_leaky[:, i], self.y)[0, 1]
            correlations.append(abs(corr))
        
        # First two features should have high correlation
        assert correlations[0] > 0.9, "Direct target copy should have high correlation"
        assert correlations[1] > 0.8, "Noisy target copy should have high correlation"
        
        # Detect using leakage checker
        leakage_report = self.leakage_checker.check_feature_target_leakage(X_leaky, self.y)
        
        assert leakage_report['has_leakage'], "Should detect feature leakage"
        # suspicious_features is a list of dicts with 'index' key
        suspicious_indices = [f['index'] for f in leakage_report['suspicious_features']]
        assert 0 in suspicious_indices, \
            "Should identify feature 0 as suspicious"
        assert 1 in suspicious_indices, \
            "Should identify feature 1 as suspicious"
    
    def test_bootstrap_leakage(self):
        """Test that bootstrap methods handle duplicates correctly"""
        cv = BootstrapValidation(n_splits=10, random_state=42)
        
        for train_idx, test_idx in cv.split(self.X[:50], self.y[:50]):
            # Bootstrap training should have duplicates
            assert len(train_idx) != len(set(train_idx)), \
                "Bootstrap should have duplicate samples"
            
            # But no sample should be in both train and test
            assert len(set(train_idx) & set(test_idx)) == 0, \
                "No sample should be in both bootstrap and OOB"
            
            # All indices should be valid
            assert max(train_idx) < 50, "Invalid training index"
            if len(test_idx) > 0:
                assert max(test_idx) < 50, "Invalid test index"
    
    def test_purging_effectiveness(self):
        """Test that purging removes information leakage"""
        # Create overlapping observations
        n = 100
        X_overlap = np.zeros((n, 5))
        
        # Each observation uses information from neighbors
        for i in range(n):
            if i > 0:
                X_overlap[i, 0] = X_overlap[i-1, 0] * 0.5
            if i < n-1:
                X_overlap[i, 1] = np.random.randn()  # Will affect next
            
            X_overlap[i, 2:] = np.random.randn(3)
        
        y_overlap = (X_overlap[:, 0] > 0).astype(int)
        
        # Without purging
        cv_no_purge = KFoldMedical(n_splits=5, shuffle=False)
        
        # With purging
        cv_purged = PurgedKFoldCV(n_splits=5, purge_size=5)
        
        # Check information leakage without purging
        for train_idx, test_idx in cv_no_purge.split(X_overlap, y_overlap):
            # Check for adjacent samples
            for test_i in test_idx:
                if test_i > 0 and (test_i - 1) in train_idx:
                    adjacent_leakage = True
                    break
                if test_i < n-1 and (test_i + 1) in train_idx:
                    adjacent_leakage = True
                    break
            else:
                continue
            break
        else:
            adjacent_leakage = False
        
        assert adjacent_leakage, "Standard CV should have adjacent samples"
        
        # Check purged CV
        for train_idx, test_idx in cv_purged.split(X_overlap, y_overlap):
            # Check purging gap
            for test_i in test_idx:
                for train_i in train_idx:
                    assert abs(test_i - train_i) > 5, \
                        f"Purging violated: {test_i} and {train_i} too close"
    
    def test_class_balance_preservation(self):
        """Test that stratified methods preserve class balance"""
        # Create imbalanced dataset
        y_imbalanced = np.concatenate([
            np.zeros(80),
            np.ones(20)
        ])
        X_imbalanced = np.random.randn(100, 5)
        
        # Shuffle
        shuffle_idx = np.random.permutation(100)
        y_imbalanced = y_imbalanced[shuffle_idx]
        X_imbalanced = X_imbalanced[shuffle_idx]
        
        overall_balance = np.mean(y_imbalanced)
        
        # Non-stratified CV
        cv_regular = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
        
        # Stratified CV
        cv_stratified = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
        
        # Check balance in regular CV
        regular_balances = []
        for train_idx, test_idx in cv_regular.split(X_imbalanced, y_imbalanced):
            test_balance = np.mean(y_imbalanced[test_idx])
            regular_balances.append(test_balance)
        
        regular_variance = np.var(regular_balances)
        
        # Check balance in stratified CV
        stratified_balances = []
        for train_idx, test_idx in cv_stratified.split(X_imbalanced, y_imbalanced):
            test_balance = np.mean(y_imbalanced[test_idx])
            stratified_balances.append(test_balance)
            
            # Each fold should maintain balance
            assert abs(test_balance - overall_balance) < 0.1, \
                "Stratified CV should maintain class balance"
        
        stratified_variance = np.var(stratified_balances)
        
        # Stratified should have lower variance
        assert stratified_variance < regular_variance, \
            "Stratified CV should have more consistent class balance"
    
    def test_comprehensive_leakage_check(self):
        """Test comprehensive leakage detection across all aspects"""
        # Create complex dataset with all types of potential leakage
        n = 200
        
        # Features with various leakage patterns
        X_complex = np.random.randn(n, 10)
        
        # Add leaky features
        y_complex = np.random.randint(0, 2, n)
        X_complex[:, 0] = y_complex * 2 + np.random.randn(n) * 0.1  # Direct leakage
        
        # Temporal structure
        for i in range(1, n):
            X_complex[i, 1] = X_complex[i-1, 1] * 0.8 + np.random.randn()
        
        # Group structure
        groups = np.repeat(np.arange(20), 10)
        
        # Spatial structure
        coords = np.random.uniform(0, 50, (n, 2))
        
        # Run comprehensive check
        report = self.leakage_checker.comprehensive_check(
            X=X_complex,
            y=y_complex,
            groups=groups,
            timestamps=np.arange(n),
            coordinates=coords
        )
        
        # Should detect feature leakage
        assert report['feature_leakage']['has_leakage'], \
            "Should detect feature leakage"
        
        # Should identify problematic features
        assert 0 in report['feature_leakage']['suspicious_features'], \
            "Should identify leaky feature"
        
        # Should provide recommendations
        assert len(report['recommendations']) > 0, \
            "Should provide recommendations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])