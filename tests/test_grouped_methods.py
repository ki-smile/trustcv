"""
Unit tests for grouped cross-validation methods
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustcv.splitters.grouped import (
    GroupKFoldMedical, StratifiedGroupKFold,
    LeaveOneGroupOut, RepeatedGroupKFold,
    NestedGroupedCV
)


class TestGroupedMethods:
    """Test suite for grouped cross-validation methods"""
    
    @classmethod
    def setup_class(cls):
        """Set up grouped test data"""
        np.random.seed(42)
        
        # Create multi-center clinical trial data
        n_patients = 150
        n_hospitals = 6
        n_measurements_per_patient = 3
        n_features = 10
        
        cls.X = []
        cls.y = []
        cls.patient_ids = []
        cls.hospital_ids = []
        
        patient_counter = 0
        
        for hospital_id in range(n_hospitals):
            # Hospital-specific effect
            hospital_effect = np.random.randn() * 0.5
            
            # Determine number of patients for this hospital
            n_patients_hospital = n_patients // n_hospitals
            
            for _ in range(n_patients_hospital):
                patient_id = patient_counter
                patient_counter += 1
                
                # Patient-specific baseline
                patient_baseline = np.random.randn(n_features) + hospital_effect
                
                for _ in range(n_measurements_per_patient):
                    # Add measurement noise
                    features = patient_baseline + np.random.randn(n_features) * 0.3
                    
                    # Binary outcome based on features and hospital effect
                    outcome_prob = 1 / (1 + np.exp(-(np.sum(features[:3]) + hospital_effect)))
                    outcome = np.random.binomial(1, outcome_prob)
                    
                    cls.X.append(features)
                    cls.y.append(outcome)
                    cls.patient_ids.append(patient_id)
                    cls.hospital_ids.append(hospital_id)
        
        cls.X = np.array(cls.X)
        cls.y = np.array(cls.y)
        cls.patient_ids = np.array(cls.patient_ids)
        cls.hospital_ids = np.array(cls.hospital_ids)
        
        # Create a small dataset for exhaustive tests
        cls.X_small = cls.X[:30]
        cls.y_small = cls.y[:30]
        cls.groups_small = cls.patient_ids[:30]
    
    def test_group_kfold_medical(self):
        """Test Group K-Fold Medical implementation"""
        cv = GroupKFoldMedical(n_splits=5)
        splits = list(cv.split(self.X, self.y, groups=self.patient_ids))
        
        assert len(splits) <= 5, "Should produce at most 5 splits"
        
        # Check that no patient appears in both train and test
        for train_idx, test_idx in splits:
            train_patients = set(self.patient_ids[train_idx])
            test_patients = set(self.patient_ids[test_idx])
            
            assert len(train_patients & test_patients) == 0, \
                "No patient should appear in both train and test"
            
            # All measurements from same patient should be together
            for patient_id in test_patients:
                patient_mask = self.patient_ids == patient_id
                patient_indices = np.where(patient_mask)[0]
                
                # All indices for this patient should be in test set
                assert all(idx in test_idx for idx in patient_indices), \
                    f"All measurements from patient {patient_id} should be in same split"
    
    def test_stratified_group_kfold(self):
        """Test Stratified Group K-Fold"""
        cv = StratifiedGroupKFold(n_splits=3)
        splits = list(cv.split(self.X, self.y, groups=self.patient_ids))
        
        assert len(splits) <= 3, "Should produce at most 3 splits"
        
        # Check stratification
        overall_positive_rate = np.mean(self.y)
        
        for train_idx, test_idx in splits:
            # Check no group leakage
            train_patients = set(self.patient_ids[train_idx])
            test_patients = set(self.patient_ids[test_idx])
            assert len(train_patients & test_patients) == 0, "No patient in both sets"
            
            # Check stratification is maintained (approximately)
            test_positive_rate = np.mean(self.y[test_idx])
            assert abs(test_positive_rate - overall_positive_rate) < 0.2, \
                "Stratification should be maintained"
    
    def test_leave_one_group_out(self):
        """Test Leave One Group Out"""
        unique_patients = np.unique(self.patient_ids)
        
        cv = LeaveOneGroupOut()
        splits = list(cv.split(self.X, self.y, groups=self.patient_ids))
        
        assert len(splits) == len(unique_patients), \
            "Should produce one split per unique group"
        
        tested_patients = []
        for train_idx, test_idx in splits:
            test_patients = np.unique(self.patient_ids[test_idx])
            
            # Each split should test exactly one patient
            assert len(test_patients) == 1, "Should test exactly one patient"
            tested_patients.append(test_patients[0])
            
            # That patient should not be in training
            train_patients = self.patient_ids[train_idx]
            assert test_patients[0] not in train_patients, \
                "Test patient should not be in training"
        
        # All patients should be tested exactly once
        assert set(tested_patients) == set(unique_patients), \
            "All patients should be tested exactly once"
    
    def test_repeated_group_kfold(self):
        """Test Repeated Group K-Fold"""
        cv = RepeatedGroupKFold(n_splits=3, n_repeats=2)
        splits = list(cv.split(self.X_small, self.y_small, groups=self.groups_small))
        
        # Should produce approximately n_splits * n_repeats
        assert len(splits) >= 3, "Should produce multiple splits"
        
        # Collect test sets from each repeat
        n_splits_per_repeat = 3
        first_repeat_tests = []
        second_repeat_tests = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Check no group leakage
            train_groups = set(self.groups_small[train_idx])
            test_groups = set(self.groups_small[test_idx])
            assert len(train_groups & test_groups) == 0, "No group in both sets"
            
            # Store test sets
            if i < n_splits_per_repeat:
                first_repeat_tests.append(set(test_idx))
            elif i < 2 * n_splits_per_repeat:
                second_repeat_tests.append(set(test_idx))
        
        # Repeats should produce different splits
        if len(first_repeat_tests) > 0 and len(second_repeat_tests) > 0:
            assert first_repeat_tests != second_repeat_tests, \
                "Repeats should produce different splits"
    
    def test_nested_grouped_cv(self):
        """Test Nested Grouped CV"""
        outer_cv = GroupKFoldMedical(n_splits=3)
        inner_cv = GroupKFoldMedical(n_splits=2)

        cv = NestedGroupedCV(outer_cv=outer_cv, inner_cv=inner_cv)

        # NestedGroupedCV doesn't have split() - use outer_cv.split()
        outer_splits = list(cv.outer_cv.split(
            self.X_small,
            self.y_small,
            groups=self.groups_small
        ))
        
        assert len(outer_splits) <= 3, "Should have at most 3 outer splits"
        
        for train_idx, test_idx in outer_splits:
            # Check no group leakage in outer split
            train_groups = set(self.groups_small[train_idx])
            test_groups = set(self.groups_small[test_idx])
            assert len(train_groups & test_groups) == 0, \
                "No group should be in both outer train and test"
            
            # Inner CV should work on outer training data
            X_train = self.X_small[train_idx]
            y_train = self.y_small[train_idx]
            groups_train = self.groups_small[train_idx]
            
            inner_splits = list(inner_cv.split(X_train, y_train, groups=groups_train))
            assert len(inner_splits) > 0, "Inner CV should produce splits"
            
            for inner_train, inner_test in inner_splits:
                inner_train_groups = set(groups_train[inner_train])
                inner_test_groups = set(groups_train[inner_test])
                assert len(inner_train_groups & inner_test_groups) == 0, \
                    "No group leakage in inner CV"
    
    def test_hospital_level_grouping(self):
        """Test grouping at hospital level"""
        cv = GroupKFoldMedical(n_splits=3)
        splits = list(cv.split(self.X, self.y, groups=self.hospital_ids))
        
        assert len(splits) <= 3, "Should produce at most 3 splits"
        
        for train_idx, test_idx in splits:
            train_hospitals = set(self.hospital_ids[train_idx])
            test_hospitals = set(self.hospital_ids[test_idx])
            
            # No hospital should appear in both train and test
            assert len(train_hospitals & test_hospitals) == 0, \
                "No hospital should appear in both train and test"
            
            # All patients from same hospital should be together
            for hospital_id in test_hospitals:
                hospital_mask = self.hospital_ids == hospital_id
                hospital_indices = np.where(hospital_mask)[0]
                assert all(idx in test_idx for idx in hospital_indices), \
                    f"All data from hospital {hospital_id} should be in same split"
    
    def test_nested_grouping_levels(self):
        """Test with nested grouping (patients within hospitals)"""
        # Create composite group IDs (hospital_patient)
        composite_groups = [
            f"{h}_{p}" for h, p in zip(self.hospital_ids, self.patient_ids)
        ]
        composite_groups = np.array(composite_groups)
        
        cv = GroupKFoldMedical(n_splits=5)
        splits = list(cv.split(self.X, self.y, groups=composite_groups))
        
        for train_idx, test_idx in splits:
            train_composites = set(composite_groups[train_idx])
            test_composites = set(composite_groups[test_idx])
            
            # No composite group in both sets
            assert len(train_composites & test_composites) == 0, \
                "No composite group should be in both train and test"
    
    def test_group_cv_with_imbalanced_groups(self):
        """Test grouped CV with imbalanced group sizes"""
        # Create imbalanced groups
        imbalanced_groups = []
        group_sizes = [5, 10, 3, 20, 2, 15, 1, 8]  # Very different sizes
        
        for group_id, size in enumerate(group_sizes):
            imbalanced_groups.extend([group_id] * size)
        
        X_imbalanced = np.random.randn(len(imbalanced_groups), 5)
        y_imbalanced = np.random.randint(0, 2, len(imbalanced_groups))
        groups_imbalanced = np.array(imbalanced_groups)
        
        cv = GroupKFoldMedical(n_splits=3)
        splits = list(cv.split(X_imbalanced, y_imbalanced, groups=groups_imbalanced))
        
        assert len(splits) > 0, "Should handle imbalanced groups"
        
        for train_idx, test_idx in splits:
            train_groups = set(groups_imbalanced[train_idx])
            test_groups = set(groups_imbalanced[test_idx])
            assert len(train_groups & test_groups) == 0, \
                "No group leakage with imbalanced groups"
    
    def test_single_measurement_per_group(self):
        """Test when each group has only one measurement"""
        X_single = np.random.randn(20, 5)
        y_single = np.random.randint(0, 2, 20)
        groups_single = np.arange(20)  # Each sample is its own group
        
        cv = GroupKFoldMedical(n_splits=4)
        splits = list(cv.split(X_single, y_single, groups=groups_single))
        
        assert len(splits) == 4, "Should work with single measurement per group"
        
        # Should behave like regular K-fold in this case
        all_test_indices = []
        for train_idx, test_idx in splits:
            all_test_indices.extend(test_idx)
            assert len(set(train_idx) & set(test_idx)) == 0, \
                "No overlap between train and test"
        
        # Each sample tested once
        assert len(set(all_test_indices)) == 20, \
            "Each sample should be tested exactly once"
    
    def test_group_cv_preserves_prevalence(self):
        """Test that stratified group CV preserves class prevalence"""
        cv = StratifiedGroupKFold(n_splits=3)
        splits = list(cv.split(self.X, self.y, groups=self.patient_ids))
        
        overall_prevalence = np.mean(self.y)
        
        prevalences = []
        for train_idx, test_idx in splits:
            test_prevalence = np.mean(self.y[test_idx])
            prevalences.append(test_prevalence)
        
        # Average prevalence across folds should be close to overall
        avg_prevalence = np.mean(prevalences)
        assert abs(avg_prevalence - overall_prevalence) < 0.1, \
            "Average fold prevalence should match overall prevalence"
    
    def test_group_cv_with_model(self):
        """Test grouped CV with actual model training"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        cv = GroupKFoldMedical(n_splits=3)
        
        # Manual cross-validation
        scores = []
        for train_idx, test_idx in cv.split(self.X_small, self.y_small, 
                                           groups=self.groups_small):
            model.fit(self.X_small[train_idx], self.y_small[train_idx])
            score = model.score(self.X_small[test_idx], self.y_small[test_idx])
            scores.append(score)
        
        assert len(scores) > 0, "Should produce scores"
        assert all(0 <= s <= 1 for s in scores), "Scores should be valid"
        
        # Verify no data leakage improved performance unrealistically
        assert np.mean(scores) < 0.95, \
            "Scores shouldn't be unrealistically high (possible leakage)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])