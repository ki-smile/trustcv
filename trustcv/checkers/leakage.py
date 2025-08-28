"""
Data Leakage Detection for Medical Machine Learning

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team

Common leakage sources in medical data:
1. Patient data in both train and test sets
2. Temporal leakage (future information in training)
3. Data preprocessing on full dataset
4. Feature engineering using test information
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass


@dataclass
class LeakageReport:
    """Detailed leakage detection report"""
    has_leakage: bool
    leakage_types: List[str]
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    details: Dict[str, any]
    recommendations: List[str]
    
    def __str__(self):
        if not self.has_leakage:
            return "✅ No data leakage detected"
        
        report = f"⚠️ Data Leakage Detected (Severity: {self.severity})\n"
        report += f"Types: {', '.join(self.leakage_types)}\n"
        report += "Recommendations:\n"
        for rec in self.recommendations:
            report += f"  • {rec}\n"
        return report


class DataLeakageChecker:
    """
    Comprehensive data leakage detection for ML
    
    Checks for:
    - Patient-level leakage
    - Temporal leakage
    - Feature leakage
    - Preprocessing leakage
    - Duplicate samples
    
    Examples
    --------
    >>> checker = DataLeakageChecker()
    >>> report = checker.check_cv_splits(X_train, X_test, y_train, y_test, patient_ids)
    >>> if report.has_leakage:
    >>>     print(report)
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def check_cv_splits(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        patient_ids_train: Optional[Union[np.ndarray, pd.Series]] = None,
        patient_ids_test: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps_train: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> LeakageReport:
        """
        Check for data leakage between train and test sets
        
        Parameters
        ----------
        X_train, X_test : array-like
            Training and test features
        y_train, y_test : array-like, optional
            Training and test labels
        patient_ids_train, patient_ids_test : array-like, optional
            Patient identifiers
        timestamps_train, timestamps_test : array-like, optional
            Temporal information
            
        Returns
        -------
        LeakageReport
            Detailed leakage analysis
        """
        leakage_types = []
        details = {}
        recommendations = []
        severity = 'none'
        
        # Check 1: Patient-level leakage
        if patient_ids_train is not None and patient_ids_test is not None:
            patient_leakage = self._check_patient_leakage(
                patient_ids_train, patient_ids_test
            )
            if patient_leakage['has_leakage']:
                leakage_types.append('patient')
                details['patient_leakage'] = patient_leakage
                recommendations.append(
                    "Use PatientGroupKFold to ensure patient-level separation"
                )
                severity = 'critical'
        
        # Check 2: Duplicate samples
        duplicate_leakage = self._check_duplicate_samples(X_train, X_test)
        if duplicate_leakage['has_leakage']:
            leakage_types.append('duplicate')
            details['duplicate_leakage'] = duplicate_leakage
            recommendations.append(
                "Remove duplicate samples before splitting"
            )
            severity = max(severity, 'high') if severity != 'none' else 'high'
        
        # Check 3: Temporal leakage
        if timestamps_train is not None and timestamps_test is not None:
            temporal_leakage = self._check_temporal_leakage(
                timestamps_train, timestamps_test
            )
            if temporal_leakage['has_leakage']:
                leakage_types.append('temporal')
                details['temporal_leakage'] = temporal_leakage
                recommendations.append(
                    "Use TemporalClinical splitter for time-series data"
                )
                severity = max(severity, 'high') if severity != 'none' else 'high'
        
        # Check 4: Feature statistics leakage
        feature_leakage = self._check_feature_statistics(X_train, X_test)
        if feature_leakage['suspicious']:
            leakage_types.append('feature_statistics')
            details['feature_leakage'] = feature_leakage
            recommendations.append(
                "Check if preprocessing was done before train-test split"
            )
            severity = max(severity, 'medium') if severity != 'none' else 'medium'
        
        # Check 5: Label distribution (if severely different, might indicate issue)
        if y_train is not None and y_test is not None:
            label_check = self._check_label_distribution(y_train, y_test)
            if label_check['suspicious']:
                details['label_distribution'] = label_check
                recommendations.append(
                    "Consider using stratified splitting for balanced class distribution"
                )
                severity = max(severity, 'low') if severity != 'none' else 'low'
        
        return LeakageReport(
            has_leakage=len(leakage_types) > 0,
            leakage_types=leakage_types,
            severity=severity,
            details=details,
            recommendations=recommendations
        )
    
    def _check_patient_leakage(
        self,
        patient_ids_train: Union[np.ndarray, pd.Series],
        patient_ids_test: Union[np.ndarray, pd.Series]
    ) -> Dict:
        """Check if same patients appear in train and test"""
        train_patients = set(patient_ids_train)
        test_patients = set(patient_ids_test)
        
        overlap = train_patients.intersection(test_patients)
        
        result = {
            'has_leakage': len(overlap) > 0,
            'overlapping_patients': list(overlap)[:10],  # Show first 10
            'overlap_count': len(overlap),
            'train_unique': len(train_patients),
            'test_unique': len(test_patients),
            'overlap_percentage': len(overlap) / len(train_patients.union(test_patients)) * 100
        }
        
        if result['has_leakage'] and self.verbose:
            warnings.warn(
                f"CRITICAL: {result['overlap_count']} patients "
                f"({result['overlap_percentage']:.1f}%) appear in both train and test sets!"
            )
        
        return result
    
    def _check_duplicate_samples(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> Dict:
        """Check for duplicate samples between train and test"""
        # Convert to DataFrame for easier handling
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        
        # Create hash of each row
        train_hashes = pd.util.hash_pandas_object(X_train)
        test_hashes = pd.util.hash_pandas_object(X_test)
        
        # Find overlaps
        common_hashes = set(train_hashes).intersection(set(test_hashes))
        
        result = {
            'has_leakage': len(common_hashes) > 0,
            'duplicate_count': len(common_hashes),
            'duplicate_percentage': len(common_hashes) / len(test_hashes) * 100
        }
        
        if result['has_leakage'] and self.verbose:
            warnings.warn(
                f"Found {result['duplicate_count']} duplicate samples "
                f"({result['duplicate_percentage']:.1f}%) between train and test!"
            )
        
        return result
    
    def _check_temporal_leakage(
        self,
        timestamps_train: Union[np.ndarray, pd.Series],
        timestamps_test: Union[np.ndarray, pd.Series]
    ) -> Dict:
        """Check if test data comes before training data (future leakage)"""
        # Convert to datetime if needed
        if isinstance(timestamps_train, (np.ndarray, list)):
            timestamps_train = pd.Series(timestamps_train)
        if isinstance(timestamps_test, (np.ndarray, list)):
            timestamps_test = pd.Series(timestamps_test)
        
        try:
            timestamps_train = pd.to_datetime(timestamps_train)
            timestamps_test = pd.to_datetime(timestamps_test)
        except:
            return {'has_leakage': False, 'error': 'Could not parse timestamps'}
        
        # Check if any test data is before earliest training data
        min_train = timestamps_train.min()
        max_train = timestamps_train.max()
        min_test = timestamps_test.min()
        max_test = timestamps_test.max()
        
        # Test data should generally come after training data
        has_leakage = min_test < max_train
        
        result = {
            'has_leakage': has_leakage,
            'train_period': f"{min_train} to {max_train}",
            'test_period': f"{min_test} to {max_test}",
            'overlap_exists': has_leakage,
            'test_before_train': min_test < min_train
        }
        
        if result['test_before_train'] and self.verbose:
            warnings.warn(
                "CRITICAL: Test data contains dates before training data! "
                "This causes temporal leakage."
            )
        
        return result
    
    def _check_feature_statistics(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> Dict:
        """Check if feature statistics are suspiciously similar"""
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        
        # Calculate statistics
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        train_std = np.std(X_train, axis=0)
        test_std = np.std(X_test, axis=0)
        
        # Check if means and stds are almost identical (might indicate preprocessing leakage)
        mean_diff = np.abs(train_mean - test_mean)
        std_diff = np.abs(train_std - test_std)
        
        # Suspicious if differences are extremely small
        suspicious_features = np.where(
            (mean_diff < 1e-10) & (std_diff < 1e-10) & (train_std > 0)
        )[0]
        
        result = {
            'suspicious': len(suspicious_features) > X_train.shape[1] * 0.5,
            'suspicious_features': len(suspicious_features),
            'total_features': X_train.shape[1],
            'likely_normalized_together': len(suspicious_features) > 0
        }
        
        if result['suspicious'] and self.verbose:
            warnings.warn(
                f"{result['suspicious_features']} features have identical statistics. "
                "Data might have been preprocessed before splitting!"
            )
        
        return result
    
    def _check_label_distribution(
        self,
        y_train: Union[np.ndarray, pd.Series],
        y_test: Union[np.ndarray, pd.Series]
    ) -> Dict:
        """Check if label distributions are significantly different"""
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        train_dist = counts_train / len(y_train)
        test_dist = counts_test / len(y_test)
        
        # Calculate distribution difference
        max_diff = 0
        for i, label in enumerate(unique_train):
            if label in unique_test:
                idx_test = np.where(unique_test == label)[0][0]
                diff = abs(train_dist[i] - test_dist[idx_test])
                max_diff = max(max_diff, diff)
        
        result = {
            'suspicious': max_diff > 0.2,  # >20% difference is suspicious
            'max_difference': max_diff,
            'train_distribution': dict(zip(unique_train, train_dist)),
            'test_distribution': dict(zip(unique_test, test_dist))
        }
        
        if result['suspicious'] and self.verbose:
            warnings.warn(
                f"Large class distribution difference ({max_diff:.1%}). "
                "Consider using stratified splitting."
            )
        
        return result
    
    def check_preprocessing_leakage(
        self,
        X_original: Union[np.ndarray, pd.DataFrame],
        X_processed: Union[np.ndarray, pd.DataFrame],
        split_indices: Tuple[np.ndarray, np.ndarray]
    ) -> bool:
        """
        Check if preprocessing was done before or after splitting
        
        Parameters
        ----------
        X_original : array-like
            Original data before preprocessing
        X_processed : array-like
            Data after preprocessing
        split_indices : tuple
            (train_indices, test_indices)
            
        Returns
        -------
        bool
            True if leakage detected
        """
        train_idx, test_idx = split_indices
        
        # Check if normalization used global statistics
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.values
            
        global_mean = np.mean(X_processed, axis=0)
        train_mean = np.mean(X_processed[train_idx], axis=0)
        test_mean = np.mean(X_processed[test_idx], axis=0)
        
        # If train mean equals global mean, likely preprocessed together
        mean_similarity = np.allclose(train_mean, global_mean, rtol=1e-10)
        
        if mean_similarity and self.verbose:
            warnings.warn(
                "Preprocessing appears to use global statistics. "
                "This causes data leakage! Fit preprocessing only on training data."
            )
        
        return mean_similarity
    
    def check_feature_target_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        threshold: float = 0.95
    ) -> Dict:
        """
        Check if any features are too correlated with target (potential leakage)
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        threshold : float
            Correlation threshold above which to flag as suspicious
            
        Returns
        -------
        dict
            Leakage detection results
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Calculate correlations
        correlations = []
        suspicious_features = []
        
        for i in range(X_array.shape[1]):
            # Handle both continuous and categorical targets
            try:
                # For continuous targets
                from scipy.stats import pearsonr
                corr, _ = pearsonr(X_array[:, i], y_array)
                correlations.append(abs(corr))
                
                if abs(corr) > threshold:
                    suspicious_features.append({
                        'index': i,
                        'name': feature_names[i],
                        'correlation': corr
                    })
            except:
                # For categorical, use mutual information
                from sklearn.feature_selection import mutual_info_classif
                mi = mutual_info_classif(X_array[:, i:i+1], y_array, random_state=42)[0]
                # Normalize MI to [0, 1] range for comparison
                normalized_mi = min(mi, 1.0)
                correlations.append(normalized_mi)
                
                if normalized_mi > threshold:
                    suspicious_features.append({
                        'index': i,
                        'name': feature_names[i],
                        'mutual_info': normalized_mi
                    })
        
        result = {
            'has_leakage': len(suspicious_features) > 0,
            'suspicious_features': suspicious_features,
            'max_correlation': max(correlations) if correlations else 0,
            'num_suspicious': len(suspicious_features)
        }
        
        if result['has_leakage'] and self.verbose:
            warnings.warn(
                f"Found {len(suspicious_features)} features with suspiciously high "
                f"correlation to target (>{threshold}). Possible target leakage!"
            )
            for feat in suspicious_features[:3]:  # Show top 3
                print(f"  - {feat['name']}: {feat.get('correlation', feat.get('mutual_info', 0)):.3f}")
        
        return result