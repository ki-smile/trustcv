#!/usr/bin/env python3
"""
Heart Disease Prediction with Proper Cross-Validation
======================================================
This example demonstrates how to properly validate a heart disease
prediction model using various CV strategies.

Dataset: UCI Heart Disease Dataset (simulated)
Task: Binary classification (disease/no disease)
Challenge: Class imbalance, feature selection
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our CV methods
import sys
sys.path.append('..')
from trustcv.splitters.iid import (
    HoldOut, KFoldMedical, StratifiedKFoldMedical,
    BootstrapValidation, NestedCV
)

# Set random seed for reproducibility
np.random.seed(42)

def create_heart_disease_data(n_samples=1000):
    """
    Create synthetic heart disease dataset with realistic features
    """
    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=13,  # Similar to real heart disease dataset
        n_informative=8,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.65, 0.35],  # 35% disease prevalence
        flip_y=0.02,  # Add some noise
        random_state=42
    )
    
    # Create feature names similar to real heart disease features
    feature_names = [
        'age', 'sex', 'chest_pain_type', 'resting_bp',
        'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
        'max_heart_rate', 'exercise_angina', 'st_depression',
        'st_slope', 'num_vessels', 'thalassemia'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Make features more realistic
    df['age'] = 30 + df['age'] * 15  # Age 30-75
    df['sex'] = (df['sex'] > 0).astype(int)  # Binary
    df['chest_pain_type'] = np.abs(df['chest_pain_type'] * 2).astype(int) % 4  # 0-3
    df['resting_bp'] = 100 + df['resting_bp'] * 20  # 80-140
    df['cholesterol'] = 150 + df['cholesterol'] * 50  # 100-250
    df['fasting_blood_sugar'] = (df['fasting_blood_sugar'] > 0).astype(int)
    df['max_heart_rate'] = 150 + df['max_heart_rate'] * 30  # 120-200
    df['exercise_angina'] = (df['exercise_angina'] > 0).astype(int)
    df['st_depression'] = np.abs(df['st_depression'] * 2)
    df['num_vessels'] = np.abs(df['num_vessels']).astype(int) % 4
    
    return df, y


def evaluate_cv_strategies(X, y, model):
    """
    Compare different CV strategies for heart disease prediction
    """
    results = {}
    
    print("=" * 60)
    print("COMPARING CROSS-VALIDATION STRATEGIES")
    print("=" * 60)
    
    # 1. Hold-Out Validation
    print("\n1. Hold-Out Validation (70/30 split)")
    print("-" * 40)
    holdout = HoldOut(test_size=0.3, random_state=42, stratify=y)
    for train_idx, test_idx in holdout.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred_proba)
        
        results['Hold-Out'] = {'scores': [score], 'mean': score, 'std': 0}
        print(f"ROC-AUC: {score:.4f}")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    
    # 2. Standard K-Fold
    print("\n2. Standard K-Fold (k=5)")
    print("-" * 40)
    kfold = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        model.fit(X[train_idx], y[train_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        scores.append(score)
        print(f"Fold {fold}: {score:.4f}")
    
    results['K-Fold'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
    print(f"Mean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 3. Stratified K-Fold
    print("\n3. Stratified K-Fold (k=5)")
    print("-" * 40)
    stratified_kfold = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y), 1):
        model.fit(X[train_idx], y[train_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        scores.append(score)
        
        # Check class distribution
        train_pos = y[train_idx].mean()
        test_pos = y[test_idx].mean()
        print(f"Fold {fold}: {score:.4f} (train pos: {train_pos:.2%}, test pos: {test_pos:.2%})")
    
    results['Stratified K-Fold'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
    print(f"Mean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 4. Bootstrap Validation
    print("\n4. Bootstrap Validation (.632 estimator)")
    print("-" * 40)
    bootstrap = BootstrapValidation(n_iterations=50, estimator='.632', random_state=42)
    train_scores = []
    test_scores = []
    
    for i, (train_idx, test_idx) in enumerate(bootstrap.split(X, y)):
        if i >= 50:  # Limit iterations for demo
            break
        model.fit(X[train_idx], y[train_idx])
        
        # Training score
        y_train_pred = model.predict_proba(X[train_idx])[:, 1]
        train_score = roc_auc_score(y[train_idx], y_train_pred)
        train_scores.append(train_score)
        
        # Test (OOB) score
        y_test_pred = model.predict_proba(X[test_idx])[:, 1]
        test_score = roc_auc_score(y[test_idx], y_test_pred)
        test_scores.append(test_score)
    
    # Calculate .632 estimate
    train_err = 1 - np.mean(train_scores)
    test_err = 1 - np.mean(test_scores)
    bootstrap_632 = 0.368 * train_err + 0.632 * test_err
    
    results['Bootstrap .632'] = {
        'scores': test_scores,
        'mean': 1 - bootstrap_632,
        'std': np.std(test_scores)
    }
    print(f"Bootstrap .632 ROC-AUC: {1 - bootstrap_632:.4f}")
    print(f"OOB ROC-AUC: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
    
    return results


def visualize_results(results):
    """
    Visualize CV comparison results
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Box plot of scores
    ax1 = axes[0]
    data_for_box = []
    labels = []
    for method, res in results.items():
        if len(res['scores']) > 1:
            data_for_box.append(res['scores'])
            labels.append(method)
    
    if data_for_box:
        bp = ax1.boxplot(data_for_box, labels=labels, patch_artist=True)
        colors = ['#870052', '#FF876F', '#4CAF50', '#2196F3']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax1.set_ylabel('ROC-AUC Score')
    ax1.set_title('Cross-Validation Strategy Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot 2: Bar plot with error bars
    ax2 = axes[1]
    methods = list(results.keys())
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    
    x_pos = np.arange(len(methods))
    colors = ['#870052', '#FF876F', '#4CAF50', '#2196F3', '#FFA500']
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=10,
                   color=colors[:len(methods)], alpha=0.7,
                   edgecolor='black', linewidth=2)
    
    ax2.set_xlabel('CV Strategy')
    ax2.set_ylabel('Mean ROC-AUC Score')
    ax2.set_title('Mean Performance ± Standard Deviation')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def nested_cv_model_selection(X, y):
    """
    Use nested CV to select best model for heart disease
    """
    print("\n" + "=" * 60)
    print("NESTED CV FOR MODEL SELECTION")
    print("=" * 60)
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        }
    }
    
    # Perform nested CV for each model
    from sklearn.model_selection import GridSearchCV
    from trustcv.splitters.iid import StratifiedKFoldMedical

    outer_cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFoldMedical(n_splits=3, shuffle=True, random_state=42)
    
    results = {}
    
    for name, config in models.items():
        print(f"\nEvaluating {name}...")
        outer_scores = []
        
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'], config['params'],
                cv=inner_cv, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Evaluate on outer test set
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            outer_scores.append(score)
        
        results[name] = {
            'scores': outer_scores,
            'mean': np.mean(outer_scores),
            'std': np.std(outer_scores)
        }
        
        print(f"  Nested CV Score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['mean'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Score: {results[best_model]['mean']:.4f} ± {results[best_model]['std']:.4f}")
    
    return results


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("HEART DISEASE PREDICTION - CV COMPARISON")
    print("=" * 60)
    
    # Create dataset
    print("\n📊 Creating synthetic heart disease dataset...")
    df, y = create_heart_disease_data(n_samples=1000)
    X = df.values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y) / len(y)}")
    print(f"Features: {list(df.columns)}")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Compare CV strategies
    results = evaluate_cv_strategies(X, y, model)
    
    # Visualize results
    visualize_results(results)
    
    # Nested CV for model selection
    nested_results = nested_cv_model_selection(X, y)
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Stratified K-Fold maintains class balance → more stable")
    print("2. Bootstrap provides confidence intervals")
    print("3. Nested CV gives unbiased model selection")
    print("4. Choice depends on dataset size and requirements")
    print("\n✅ Always validate ML models properly!")


if __name__ == "__main__":
    main()