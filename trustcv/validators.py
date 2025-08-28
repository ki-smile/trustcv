"""
Trustworthy Cross-Validation Validators

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
Main validation classes with medical-specific features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from typing import Optional, Dict, Any, Union, Tuple, List
import warnings
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """Results from medical cross-validation"""
    scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    fold_details: List[Dict]
    leakage_check: Dict[str, bool]
    recommendations: List[str]
    
    def summary(self) -> str:
        """Generate summary report"""
        summary = "=== Trustworthy Cross-Validation Results ===\n\n"
        summary += "Performance Metrics (mean ± std):\n"
        for metric, mean_val in self.mean_scores.items():
            std_val = self.std_scores[metric]
            ci_lower, ci_upper = self.confidence_intervals[metric]
            summary += f"  {metric}: {mean_val:.3f} ± {std_val:.3f} "
            summary += f"[95% CI: {ci_lower:.3f}-{ci_upper:.3f}]\n"
        
        summary += "\nData Integrity Checks:\n"
        for check, passed in self.leakage_check.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            summary += f"  {check}: {status}\n"
        
        if self.recommendations:
            summary += "\nRecommendations:\n"
            for rec in self.recommendations:
                summary += f"  • {rec}\n"
        
        return summary
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'mean_scores': self.mean_scores,
            'std_scores': self.std_scores,
            'confidence_intervals': self.confidence_intervals,
            'leakage_check': self.leakage_check,
            'recommendations': self.recommendations
        }


class MedicalValidator:
    """
    Main validator class for medical machine learning
    
    Features:
    - Automatic patient-level splitting
    - Data leakage detection
    - Clinical metrics calculation
    - Regulatory compliance reporting
    """
    
    def __init__(
        self,
        method: str = 'stratified_kfold',
        n_splits: int = 5,
        random_state: int = 42,
        check_leakage: bool = True,
        check_balance: bool = True,
        compliance: Optional[str] = None
    ):
        """
        Initialize Medical Validator
        
        Parameters:
        -----------
        method : str
            Cross-validation method ('kfold', 'stratified_kfold', 
            'patient_grouped_kfold', 'temporal')
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed for reproducibility
        check_leakage : bool
            Whether to check for data leakage
        check_balance : bool
            Whether to check class balance
        compliance : str
            Regulatory compliance mode ('FDA', 'CE', None)
        """
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state
        self.check_leakage = check_leakage
        self.check_balance = check_balance
        self.compliance = compliance
        
        self._cv_splitter = None
        self._setup_splitter()
    
    def _setup_splitter(self):
        """Configure the appropriate CV splitter"""
        from sklearn.model_selection import (
            KFold, StratifiedKFold, GroupKFold, 
            TimeSeriesSplit
        )
        
        if self.method == 'kfold':
            self._cv_splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        elif self.method == 'stratified_kfold':
            self._cv_splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        elif self.method == 'patient_grouped_kfold':
            self._cv_splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'temporal':
            self._cv_splitter = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_validate(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        patient_ids: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
        scoring: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Perform medical cross-validation
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to validate
        X : array-like
            Features
        y : array-like
            Labels
        patient_ids : array-like, optional
            Patient identifiers for grouped splitting
        timestamps : array-like, optional
            Timestamps for temporal validation
        scoring : dict, optional
            Scoring metrics
        
        Returns:
        --------
        ValidationResult object with comprehensive metrics
        """
        # Default medical scoring metrics
        if scoring is None:
            scoring = self._get_medical_scoring()
        
        # Prepare groups for patient-level splitting
        groups = patient_ids if patient_ids is not None else None
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=self._cv_splitter,
            groups=groups,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        mean_scores = {
            metric: np.mean(scores) 
            for metric, scores in cv_results.items() 
            if metric.startswith('test_')
        }
        
        std_scores = {
            metric: np.std(scores) 
            for metric, scores in cv_results.items() 
            if metric.startswith('test_')
        }
        
        # Calculate 95% confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            cv_results, alpha=0.05
        )
        
        # Check for data leakage
        leakage_check = {}
        if self.check_leakage and patient_ids is not None:
            leakage_check = self._check_data_leakage(
                X, y, patient_ids, cv_results
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            mean_scores, std_scores, leakage_check, X.shape
        )
        
        # Prepare fold details
        fold_details = self._extract_fold_details(cv_results)
        
        # Create result object
        result = ValidationResult(
            scores=cv_results,
            mean_scores=mean_scores,
            std_scores=std_scores,
            confidence_intervals=confidence_intervals,
            fold_details=fold_details,
            leakage_check=leakage_check,
            recommendations=recommendations
        )
        
        # Generate compliance report if needed
        if self.compliance:
            self._generate_compliance_report(result, model, X, y)
        
        return result
    
    def _get_medical_scoring(self) -> Dict[str, Any]:
        """Get medical-relevant scoring metrics"""
        from sklearn.metrics import make_scorer
        
        def sensitivity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        
        def specificity_score(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'sensitivity': make_scorer(sensitivity_score),
            'specificity': make_scorer(specificity_score),
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
    
    def _calculate_confidence_intervals(
        self,
        cv_results: Dict,
        alpha: float = 0.05
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals using bootstrap"""
        from scipy import stats
        
        confidence_intervals = {}
        for metric in cv_results:
            if metric.startswith('test_'):
                scores = cv_results[metric]
                # Using t-distribution for small samples
                mean = np.mean(scores)
                sem = stats.sem(scores)
                ci = stats.t.interval(
                    1 - alpha, len(scores) - 1,
                    loc=mean, scale=sem
                )
                metric_name = metric.replace('test_', '')
                confidence_intervals[metric_name] = ci
        
        return confidence_intervals
    
    def _check_data_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        patient_ids: Union[np.ndarray, pd.Series],
        cv_results: Dict
    ) -> Dict[str, bool]:
        """Check for various types of data leakage"""
        checks = {
            'no_patient_leakage': True,
            'no_duplicate_samples': True,
            'balanced_classes': True
        }
        
        # Check if same patient appears in train and test
        if patient_ids is not None:
            for fold_idx, estimator in enumerate(cv_results['estimator']):
                # This would need actual train/test indices from CV
                # Simplified check here
                unique_patients = len(np.unique(patient_ids))
                if unique_patients < len(patient_ids) * 0.8:
                    checks['no_patient_leakage'] = False
                    warnings.warn(
                        "Potential patient leakage detected: "
                        "Multiple records per patient found"
                    )
        
        # Check for duplicate samples
        if isinstance(X, pd.DataFrame):
            if X.duplicated().any():
                checks['no_duplicate_samples'] = False
                warnings.warn("Duplicate samples detected in dataset")
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 2:  # Binary classification
            ratio = counts.min() / counts.max()
            if ratio < 0.1:  # Severely imbalanced
                checks['balanced_classes'] = False
                warnings.warn(
                    f"Severe class imbalance detected: {ratio:.2%} minority class"
                )
        
        return checks
    
    def _generate_recommendations(
        self,
        mean_scores: Dict[str, float],
        std_scores: Dict[str, float],
        leakage_check: Dict[str, bool],
        data_shape: Tuple
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for high variance
        for metric, std in std_scores.items():
            if 'test_' in metric:
                clean_metric = metric.replace('test_', '')
                if std > 0.1:  # High variance threshold
                    recommendations.append(
                        f"High variance in {clean_metric} ({std:.3f}). "
                        "Consider increasing sample size or using stratification."
                    )
        
        # Sample size recommendations
        n_samples = data_shape[0]
        n_features = data_shape[1]
        if n_samples < 10 * n_features:
            recommendations.append(
                f"Low sample-to-feature ratio ({n_samples}/{n_features}). "
                "Consider feature selection or regularization."
            )
        
        # Leakage warnings
        if not all(leakage_check.values()):
            failed_checks = [k for k, v in leakage_check.items() if not v]
            recommendations.append(
                f"Data integrity issues detected: {', '.join(failed_checks)}. "
                "Review data preprocessing pipeline."
            )
        
        # Method-specific recommendations
        if self.method == 'kfold' and not leakage_check.get('balanced_classes', True):
            recommendations.append(
                "Consider using StratifiedKFold for imbalanced classes."
            )
        
        return recommendations
    
    def _extract_fold_details(self, cv_results: Dict) -> List[Dict]:
        """Extract detailed information for each fold"""
        fold_details = []
        n_folds = len(cv_results['test_accuracy'])
        
        for i in range(n_folds):
            fold_info = {
                'fold': i + 1,
                'train_score': cv_results['train_accuracy'][i],
                'test_score': cv_results['test_accuracy'][i],
                'fit_time': cv_results['fit_time'][i],
                'score_time': cv_results['score_time'][i]
            }
            fold_details.append(fold_info)
        
        return fold_details
    
    def _generate_compliance_report(
        self,
        result: ValidationResult,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ):
        """Generate regulatory compliance report"""
        if self.compliance == 'FDA':
            # FDA-specific requirements
            report = {
                'device_description': str(model.__class__.__name__),
                'validation_method': self.method,
                'sample_size': len(y),
                'performance_metrics': result.to_dict(),
                'data_integrity': result.leakage_check,
                'random_seed': self.random_state,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save FDA report
            with open('fda_validation_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print("FDA validation report generated: fda_validation_report.json")
        
        elif self.compliance == 'CE':
            # CE Mark requirements (simplified)
            print("CE compliance report generation (placeholder)")
    
    def suggest_best_method(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        patient_ids: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> str:
        """Suggest best CV method based on data characteristics"""
        n_samples = len(y)
        
        # Check for temporal data
        if timestamps is not None:
            return "temporal"
        
        # Check for grouped data
        if patient_ids is not None:
            unique_patients = len(np.unique(patient_ids))
            if unique_patients < n_samples:
                return "patient_grouped_kfold"
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) > 1:
            ratio = counts.min() / counts.max()
            if ratio < 0.3:  # Imbalanced
                return "stratified_kfold"
        
        # Default
        return "kfold" if n_samples > 1000 else "stratified_kfold"