#!/usr/bin/env python
"""
Heart Disease Classification with Proper Cross-Validation

This example demonstrates:
- Loading medical data with patient grouping
- Checking for data leakage
- Using stratified patient-grouped CV
- Calculating clinical metrics
- Generating FDA-style report
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import trustcv components
from trustcv import MedicalValidator
from trustcv.datasets import load_heart_disease
from trustcv.checkers import DataLeakageChecker
from trustcv.metrics import ClinicalMetrics


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("HEART DISEASE CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # 1. Load Data
    print("\n1. Loading heart disease dataset...")
    X, y, patient_ids = load_heart_disease()
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Unique patients: {patient_ids.nunique()}")
    print(f"   Disease prevalence: {y.mean():.1%}")
    print(f"   Records per patient: {len(X) / patient_ids.nunique():.1f}")
    
    # 2. Check for Data Quality Issues
    print("\n2. Checking data quality...")
    
    # Check for missing values
    missing = X.isnull().sum()
    if missing.any():
        print(f"   ⚠️ Missing values detected: {missing[missing > 0]}")
    else:
        print("   ✅ No missing values")
    
    # Check class balance
    class_dist = y.value_counts(normalize=True)
    print(f"   Class distribution: {dict(class_dist)}")
    
    if min(class_dist) < 0.1:
        print("   ⚠️ Severe class imbalance detected!")
    
    # 3. Set Up Cross-Validation
    print("\n3. Setting up medical-aware cross-validation...")
    
    validator = MedicalValidator(
        method='stratified_group_kfold',  # Handles both grouping and imbalance
        n_splits=5,
        check_leakage=True,
        check_balance=True,
        compliance='FDA'
    )
    
    # 4. Create ML Pipeline
    print("\n4. Creating ML pipeline...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,  # Prevent overfitting
            random_state=42,
            class_weight='balanced'  # Handle imbalance
        ))
    ])
    
    # 5. Perform Cross-Validation
    print("\n5. Running cross-validation...")
    print("   This may take a moment...")
    
    results = validator.fit_validate(
        model=pipeline,
        X=X,
        y=y,
        patient_ids=patient_ids
    )
    
    # 6. Display Results
    print("\n6. Cross-Validation Results:")
    print("-" * 40)
    print(results.summary())
    
    # 7. Calculate Clinical Metrics
    print("\n7. Calculating clinical metrics...")
    
    # Get predictions from the last fold for demonstration
    from sklearn.model_selection import train_test_split
    
    # Create a single train-test split respecting patient grouping
    unique_patients = patient_ids.unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42, stratify=None
    )
    
    train_mask = patient_ids.isin(train_patients)
    test_mask = patient_ids.isin(test_patients)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate clinical metrics
    clinical_metrics = ClinicalMetrics(prevalence=y.mean())
    metrics = clinical_metrics.calculate_all(y_test, y_pred, y_proba)
    
    # Display clinical report
    print("\n" + clinical_metrics.format_report(metrics))
    
    # 8. Check for Data Leakage
    print("\n8. Checking for data leakage...")
    
    checker = DataLeakageChecker()
    leakage_report = checker.check_cv_splits(
        X_train, X_test,
        y_train, y_test,
        patient_ids[train_mask], patient_ids[test_mask]
    )
    
    if leakage_report.has_leakage:
        print(f"   ❌ {leakage_report}")
    else:
        print("   ✅ No data leakage detected")
    
    # 9. Feature Importance
    print("\n9. Top 10 Most Important Features:")
    print("-" * 40)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.3f}")
    
    # 10. Generate Recommendations
    print("\n10. Clinical Recommendations:")
    print("-" * 40)
    
    if metrics['sensitivity'] > 0.90:
        print("   ✅ High sensitivity - suitable for screening")
    else:
        print("   ⚠️ Consider improving sensitivity for screening use")
    
    if metrics['specificity'] > 0.90:
        print("   ✅ High specificity - suitable for confirmation")
    else:
        print("   ⚠️ Consider improving specificity to reduce false positives")
    
    if metrics['auc_roc'] > 0.80:
        print("   ✅ Good overall discrimination (AUC > 0.80)")
    else:
        print("   ⚠️ Model discrimination needs improvement")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Save results
    print("\nSaving results to 'heart_disease_results.json'...")
    import json
    with open('heart_disease_results.json', 'w') as f:
        json.dump({
            'dataset_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_patients': patient_ids.nunique(),
                'prevalence': float(y.mean())
            },
            'cv_results': results.to_dict(),
            'clinical_metrics': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
                if not isinstance(v, tuple) and not isinstance(v, dict)
            }
        }, f, indent=2)
    
    print("✅ Results saved successfully!")
    
    return results, metrics


if __name__ == "__main__":
    results, metrics = main()