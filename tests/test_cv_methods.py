"""
Unit tests for trustcv cross-validation methods
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import trustcv components
from trustcv import MedicalValidator
from trustcv.splitters import PatientGroupKFold, StratifiedGroupKFold
from trustcv.checkers import DataLeakageChecker
from trustcv.metrics import ClinicalMetrics, calculate_nnt


class TestMedicalValidator:
    """Test MedicalValidator class"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.7, 0.3],
            random_state=42
        )
        self.X = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y = pd.Series(self.y)
        
        # Create patient IDs with some patients having multiple records
        self.patient_ids = pd.Series(
            [f'P{i//3:03d}' for i in range(200)]
        )
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = MedicalValidator(
            method='stratified_kfold',
            n_splits=5,
            check_leakage=True
        )
        assert validator.method == 'stratified_kfold'
        assert validator.n_splits == 5
        assert validator.check_leakage == True
    
    def test_basic_validation(self):
        """Test basic cross-validation"""
        validator = MedicalValidator(
            method='kfold',
            n_splits=3,
            check_leakage=False
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = validator.fit_validate(model, self.X, self.y)
        
        assert results.mean_scores is not None
        assert len(results.fold_details) == 3
        assert 'accuracy' in results.mean_scores
    
    def test_patient_grouped_validation(self):
        """Test patient-grouped cross-validation"""
        validator = MedicalValidator(
            method='patient_grouped_kfold',
            n_splits=3,
            check_leakage=True
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = validator.fit_validate(
            model, self.X, self.y, 
            patient_ids=self.patient_ids
        )
        
        assert results.leakage_check.get('no_patient_leakage', False)
    
    def test_method_suggestion(self):
        """Test automatic method suggestion"""
        validator = MedicalValidator()
        
        # Test with grouped data
        suggested = validator.suggest_best_method(
            self.X, self.y, patient_ids=self.patient_ids
        )
        assert suggested == 'patient_grouped_kfold'
        
        # Test with imbalanced data
        y_imbalanced = pd.Series([0] * 190 + [1] * 10)
        suggested = validator.suggest_best_method(self.X, y_imbalanced)
        assert suggested == 'stratified_kfold'


class TestPatientGroupKFold:
    """Test PatientGroupKFold splitter"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randint(0, 2, 100)
        self.groups = np.array([f'P{i//5:02d}' for i in range(100)])
    
    def test_patient_grouping(self):
        """Test that patients stay together"""
        pgkf = PatientGroupKFold(n_splits=3)
        
        for train_idx, test_idx in pgkf.split(self.X, self.y, self.groups):
            train_patients = set(self.groups[train_idx])
            test_patients = set(self.groups[test_idx])
            
            # Check no overlap
            assert len(train_patients.intersection(test_patients)) == 0
    
    def test_n_splits(self):
        """Test number of splits"""
        pgkf = PatientGroupKFold(n_splits=5)
        splits = list(pgkf.split(self.X, self.y, self.groups))
        assert len(splits) == 5
    
    def test_error_on_missing_groups(self):
        """Test error when groups not provided"""
        pgkf = PatientGroupKFold(n_splits=3)
        with pytest.raises(ValueError):
            list(pgkf.split(self.X, self.y, groups=None))


class TestDataLeakageChecker:
    """Test DataLeakageChecker"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(80, 5))
        self.X_test = pd.DataFrame(np.random.randn(20, 5))
        self.y_train = pd.Series(np.random.randint(0, 2, 80))
        self.y_test = pd.Series(np.random.randint(0, 2, 20))
    
    def test_no_leakage(self):
        """Test when no leakage exists"""
        checker = DataLeakageChecker(verbose=False)
        
        patient_ids_train = pd.Series([f'P{i:03d}' for i in range(80)])
        patient_ids_test = pd.Series([f'P{i:03d}' for i in range(80, 100)])
        
        report = checker.check_cv_splits(
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            patient_ids_train, patient_ids_test
        )
        
        assert not report.has_leakage
        assert report.severity == 'none'
    
    def test_patient_leakage_detection(self):
        """Test detection of patient leakage"""
        checker = DataLeakageChecker(verbose=False)
        
        # Create overlapping patient IDs
        patient_ids_train = pd.Series([f'P{i:03d}' for i in range(80)])
        patient_ids_test = pd.Series([f'P{i:03d}' for i in range(70, 90)])
        
        report = checker.check_cv_splits(
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            patient_ids_train, patient_ids_test
        )
        
        assert report.has_leakage
        assert 'patient' in report.leakage_types
        assert report.severity == 'critical'
    
    def test_duplicate_sample_detection(self):
        """Test detection of duplicate samples"""
        checker = DataLeakageChecker(verbose=False)
        
        # Add duplicate rows
        self.X_test.iloc[0] = self.X_train.iloc[0]
        
        report = checker.check_cv_splits(
            self.X_train, self.X_test,
            self.y_train, self.y_test
        )
        
        assert report.has_leakage
        assert 'duplicate' in report.leakage_types


class TestClinicalMetrics:
    """Test ClinicalMetrics calculator"""
    
    def setup_method(self):
        """Set up test predictions"""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        self.y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        self.y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.2, 0.7, 0.9, 0.1])
    
    def test_basic_metrics(self):
        """Test basic metric calculation"""
        metrics_calc = ClinicalMetrics()
        metrics = metrics_calc.calculate_all(self.y_true, self.y_pred, self.y_proba)
        
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'ppv' in metrics
        assert 'npv' in metrics
        assert 'auc_roc' in metrics
        
        # Check value ranges
        assert 0 <= metrics['sensitivity'] <= 1
        assert 0 <= metrics['specificity'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        metrics_calc = ClinicalMetrics(confidence_level=0.95)
        metrics = metrics_calc.calculate_all(self.y_true, self.y_pred)
        
        assert 'sensitivity_ci' in metrics
        assert 'specificity_ci' in metrics
        
        # Check CI format
        assert len(metrics['sensitivity_ci']) == 2
        assert metrics['sensitivity_ci'][0] <= metrics['sensitivity']
        assert metrics['sensitivity_ci'][1] >= metrics['sensitivity']
    
    def test_nnt_calculation(self):
        """Test NNT calculation"""
        nnt = calculate_nnt(
            sensitivity=0.8,
            specificity=0.9,
            prevalence=0.1
        )
        assert nnt is not None
        assert nnt > 0
    
    def test_clinical_significance(self):
        """Test clinical significance assessment"""
        metrics_calc = ClinicalMetrics()
        metrics = metrics_calc.calculate_all(self.y_true, self.y_pred)
        
        assert 'clinical_significance' in metrics
        assert 'recommendations' in metrics['clinical_significance']


class TestTemporalSplitters:
    """Test temporal cross-validation splitters"""
    
    def setup_method(self):
        """Set up temporal test data"""
        np.random.seed(42)
        self.n_samples = 100
        self.X = np.random.randn(self.n_samples, 5)
        self.y = np.random.randint(0, 2, self.n_samples)
        self.timestamps = pd.date_range('2020-01-01', periods=self.n_samples, freq='D')
    
    def test_temporal_ordering(self):
        """Test that temporal order is preserved"""
        from trustcv.splitters import TemporalClinical
        
        tscv = TemporalClinical(n_splits=3)
        
        for train_idx, test_idx in tscv.split(self.X, timestamps=self.timestamps):
            train_times = self.timestamps[train_idx]
            test_times = self.timestamps[test_idx]
            
            # All training data should come before test data
            assert train_times.max() < test_times.min()
    
    def test_temporal_gap(self):
        """Test temporal gap between train and test"""
        from trustcv.splitters import TemporalClinical
        
        gap_days = 7
        tscv = TemporalClinical(n_splits=3, gap=gap_days)
        
        for train_idx, test_idx in tscv.split(self.X, timestamps=self.timestamps):
            if len(train_idx) > 0 and len(test_idx) > 0:
                train_end = self.timestamps[train_idx].max()
                test_start = self.timestamps[test_idx].min()
                
                # Check gap is respected
                actual_gap = (test_start - train_end).days
                assert actual_gap >= gap_days


if __name__ == '__main__':
    pytest.main([__file__, '-v'])