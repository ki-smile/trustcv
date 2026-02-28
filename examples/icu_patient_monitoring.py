#!/usr/bin/env python3
"""
ICU Patient Monitoring - Temporal Cross-Validation
====================================================
This example demonstrates proper validation of time-series models
for ICU patient monitoring and deterioration prediction.

Dataset: Simulated ICU vital signs data
Task: Predict patient deterioration in next 6 hours
Challenge: Temporal dependencies, varying patient stays
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our temporal CV methods
import sys
sys.path.append('..')
from trustcv.splitters.temporal import (
    TimeSeriesSplit, RollingWindowCV, ExpandingWindowCV,
    BlockedTimeSeries, PurgedKFoldCV
)

# Set random seed for reproducibility
np.random.seed(42)

# Fast mode by default for CI/test environments
FAST_MODE = os.environ.get("TRUSTCV_FULL_EXAMPLE", "").lower() not in ("1", "true", "yes")
SHOW_PLOTS = os.environ.get("TRUSTCV_PLOT", "").lower() in ("1", "true", "yes")


def _safe_auc(y_true, y_score, y_pred=None):
    """Return ROC-AUC when possible, otherwise fallback to accuracy."""
    if len(np.unique(y_true)) < 2:
        if y_pred is not None:
            return float(accuracy_score(y_true, y_pred)), True
        return np.nan, True
    return float(roc_auc_score(y_true, y_score)), False

def create_icu_monitoring_data(n_patients=100, avg_hours=72):
    """
    Create synthetic ICU monitoring dataset with temporal patterns
    
    Features include:
    - Vital signs (HR, BP, RR, SpO2, Temperature)
    - Lab values (WBC, Lactate, Creatinine)
    - Clinical scores (SOFA, GCS)
    """
    all_data = []
    
    for patient_id in range(n_patients):
        # Random length of stay (24-120 hours)
        stay_hours = np.random.randint(24, min(120, avg_hours * 2))
        
        # Generate hourly measurements
        timestamps = pd.date_range(
            start=datetime(2024, 1, 1) + timedelta(days=patient_id),
            periods=stay_hours,
            freq='H'
        )
        
        # Patient baseline characteristics
        age = np.random.randint(30, 85)
        severity = np.random.uniform(0, 1)  # Overall illness severity
        
        # Initialize vital signs with realistic ranges
        hr_baseline = 70 + severity * 20
        bp_sys_baseline = 120 - severity * 30
        rr_baseline = 16 + severity * 8
        spo2_baseline = 98 - severity * 10
        temp_baseline = 37 + severity * 1.5
        
        for t, timestamp in enumerate(timestamps):
            # Add temporal trends and noise
            trend = t / stay_hours
            circadian = np.sin(2 * np.pi * t / 24)
            
            # Vital signs with temporal correlation
            hr = hr_baseline + trend * 10 + circadian * 5 + np.random.randn() * 5
            bp_sys = bp_sys_baseline - trend * 10 + circadian * 3 + np.random.randn() * 8
            bp_dia = bp_sys * 0.6 + np.random.randn() * 5
            rr = rr_baseline + trend * 3 + np.random.randn() * 2
            spo2 = np.clip(spo2_baseline - trend * 3 + np.random.randn() * 2, 85, 100)
            temp = temp_baseline + trend * 0.5 + circadian * 0.2 + np.random.randn() * 0.3
            
            # Lab values (less frequent, interpolated)
            if t % 6 == 0:  # Every 6 hours
                wbc = 10 + severity * 10 + trend * 5 + np.random.randn() * 2
                lactate = 1 + severity * 3 + trend * 2 + np.random.randn() * 0.5
                creatinine = 1 + severity * 1 + trend * 0.5 + np.random.randn() * 0.2
            
            # Clinical scores
            sofa_score = int(severity * 10 + trend * 5 + np.random.randn())
            gcs_score = int(15 - severity * 5 - trend * 2 + np.random.randn())
            
            # Deterioration event (binary outcome)
            # Higher probability with worse vitals and later in stay
            deterioration_prob = 1 / (1 + np.exp(-(-5 + severity * 3 + trend * 2 +
                                                   (hr > 100) * 0.5 + (bp_sys < 90) * 1 +
                                                   (spo2 < 92) * 1 + (lactate > 2) * 0.5)))
            
            deteriorates_in_6h = np.random.random() < deterioration_prob
            
            all_data.append({
                'patient_id': patient_id,
                'timestamp': timestamp,
                'hour_in_stay': t,
                'age': age,
                'heart_rate': hr,
                'bp_systolic': bp_sys,
                'bp_diastolic': bp_dia,
                'respiratory_rate': rr,
                'spo2': spo2,
                'temperature': temp,
                'wbc': wbc if t % 6 == 0 else np.nan,
                'lactate': lactate if t % 6 == 0 else np.nan,
                'creatinine': creatinine if t % 6 == 0 else np.nan,
                'sofa_score': sofa_score,
                'gcs_score': gcs_score,
                'deteriorates_6h': deteriorates_in_6h
            })
    
    df = pd.DataFrame(all_data)
    
    # Forward fill missing lab values
    df['wbc'] = df.groupby('patient_id')['wbc'].ffill()
    df['lactate'] = df.groupby('patient_id')['lactate'].ffill()
    df['creatinine'] = df.groupby('patient_id')['creatinine'].ffill()
    
    return df


# Backward-compat names expected by tests
def create_icu_data(n_patients=100, avg_hours=72):
    return create_icu_monitoring_data(n_patients=n_patients, avg_hours=avg_hours)


def engineer_features(df):
    """
    Create temporal features for ICU monitoring
    """
    features_df = df.copy()
    
    # Sort by patient and time
    features_df = features_df.sort_values(['patient_id', 'timestamp'])
    
    # Calculate rolling statistics (6-hour windows)
    window_size = 6
    
    vital_cols = ['heart_rate', 'bp_systolic', 'respiratory_rate', 'spo2', 'temperature']
    
    for col in vital_cols:
        # Rolling mean
        features_df[f'{col}_mean_6h'] = features_df.groupby('patient_id')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        
        # Rolling std (variability)
        features_df[f'{col}_std_6h'] = features_df.groupby('patient_id')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).std()
        )
        
        # Trend (current - 6h ago)
        features_df[f'{col}_trend'] = features_df.groupby('patient_id')[col].transform(
            lambda x: x.diff(window_size)
        )
    
    # Calculate shock index
    features_df['shock_index'] = features_df['heart_rate'] / features_df['bp_systolic']
    
    # Calculate MAP (Mean Arterial Pressure)
    features_df['map'] = (features_df['bp_systolic'] + 2 * features_df['bp_diastolic']) / 3
    
    # Time-based features
    features_df['hour_of_day'] = features_df['timestamp'].dt.hour
    features_df['day_of_stay'] = features_df['hour_in_stay'] // 24
    
    return features_df


def evaluate_temporal_cv_strategies(X, y, timestamps, patient_ids, fast_mode=True):
    """
    Compare different temporal CV strategies for ICU monitoring
    """
    results = {}
    
    print("="*60)
    print("TEMPORAL CROSS-VALIDATION FOR ICU MONITORING")
    print("="*60)
    
    # 1. Standard Time Series Split
    print("\n1. Time Series Split (Train on Past, Test on Future)")
    print("-"*40)
    n_splits = 3 if fast_mode else 5
    tss = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tss.split(X), 1):
        model = RandomForestClassifier(n_estimators=30 if fast_mode else 50, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score, used_fallback = _safe_auc(y[test_idx], y_pred_proba, y_pred)
        if np.isnan(score):
            print(f"Fold {fold}: Skipped (single class)")
            continue
        scores.append(score)
        
        train_time_range = (pd.Timestamp(timestamps[train_idx].min()), pd.Timestamp(timestamps[train_idx].max()))
        test_time_range = (pd.Timestamp(timestamps[test_idx].min()), pd.Timestamp(timestamps[test_idx].max()))

        metric_name = "Accuracy (fallback)" if used_fallback else "ROC-AUC"
        print(f"Fold {fold}: {metric_name} = {score:.4f}")
        print(f"  Train: {train_time_range[0].strftime('%Y-%m-%d')} to {train_time_range[1].strftime('%Y-%m-%d')}")
        print(f"  Test:  {test_time_range[0].strftime('%Y-%m-%d')} to {test_time_range[1].strftime('%Y-%m-%d')}")
    
    if scores:
        results['Time Series Split'] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        print(f"\nMean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 2. Rolling Window CV
    window_size = 24 if fast_mode else 48
    step_size = 12 if fast_mode else 24
    forecast_horizon = 12
    print(f"\n2. Rolling Window CV (Fixed {window_size}-hour Training Window)")
    print("-"*40)

    # Convert to hourly indices for rolling window
    unique_hours = np.unique(timestamps)
    hour_to_idx = {hour: idx for idx, hour in enumerate(unique_hours)}
    time_indices = np.array([hour_to_idx[t] for t in timestamps])
    
    rolling_cv = RollingWindowCV(
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        step_size=step_size
    )
    
    scores = []
    for fold, (train_idx, test_idx) in enumerate(rolling_cv.split(time_indices), 1):
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Skip if not enough classes in train or test
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                print(f"Window {fold}: Skipped (single class)")
                continue

            model = RandomForestClassifier(n_estimators=30 if fast_mode else 50, random_state=42)
            model.fit(X[train_idx], y[train_idx])

            y_pred = model.predict(X[test_idx])
            y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
            score, used_fallback = _safe_auc(y[test_idx], y_pred_proba, y_pred)
            if np.isnan(score):
                print(f"Window {fold}: Skipped (single class)")
                continue
            scores.append(score)

            metric_name = "Accuracy (fallback)" if used_fallback else "ROC-AUC"
            print(f"Window {fold}: {metric_name} = {score:.4f} "
                  f"(train size: {len(train_idx)}, test size: {len(test_idx)})")
    
    if scores:
        results['Rolling Window'] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        print(f"\nMean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 3. Expanding Window CV
    print("\n3. Expanding Window CV (Growing Training Set)")
    print("-"*40)
    
    expanding_cv = ExpandingWindowCV(
        min_train_size=60 if fast_mode else 100,
        forecast_horizon=30 if fast_mode else 50,
        step_size=30 if fast_mode else 50
    )
    
    scores = []
    train_sizes = []
    
    for fold, (train_idx, test_idx) in enumerate(expanding_cv.split(X), 1):
        if len(train_idx) >= (60 if fast_mode else 100) and len(test_idx) > 0:
            model = RandomForestClassifier(n_estimators=30 if fast_mode else 50, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
            score, used_fallback = _safe_auc(y[test_idx], y_pred_proba, y_pred)
            if np.isnan(score):
                print(f"Fold {fold}: Skipped (single class)")
                continue
            scores.append(score)
            train_sizes.append(len(train_idx))

            metric_name = "Accuracy (fallback)" if used_fallback else "ROC-AUC"
            print(f"Fold {fold}: {metric_name} = {score:.4f} "
                  f"(train size: {len(train_idx)}, test size: {len(test_idx)})")
    
    if scores:
        results['Expanding Window'] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'train_sizes': train_sizes
        }
        print(f"\nMean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 4. Blocked Time Series (Daily Blocks)
    print("\n4. Blocked Time Series CV (Daily Blocks)")
    print("-"*40)

    # Create day blocks
    days = pd.to_datetime(timestamps).normalize()

    blocked_cv = BlockedTimeSeries(n_splits=n_splits, block_size='day')
    scores = []

    for fold, (train_idx, test_idx) in enumerate(blocked_cv.split(X, timestamps=timestamps), 1):
        model = RandomForestClassifier(n_estimators=30 if fast_mode else 50, random_state=42)
        model.fit(X[train_idx], y[train_idx])

        y_pred = model.predict(X[test_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score, used_fallback = _safe_auc(y[test_idx], y_pred_proba, y_pred)
        if np.isnan(score):
            print(f"Fold {fold}: Skipped (single class)")
            continue
        scores.append(score)

        train_days = np.unique(days[train_idx])
        test_days = np.unique(days[test_idx])

        metric_name = "Accuracy (fallback)" if used_fallback else "ROC-AUC"
        print(f"Fold {fold}: {metric_name} = {score:.4f}")
        print(f"  Train days: {len(train_days)}, Test days: {len(test_days)}")
    
    if scores:
        results['Blocked Time Series'] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        print(f"\nMean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 5. Purged K-Fold (with temporal gap)
    print("\n5. Purged K-Fold CV (6-hour Gap Between Train/Test)")
    print("-"*40)
    
    purged_cv = PurgedKFoldCV(n_splits=n_splits, purge_gap=3 if fast_mode else 6)  # 6-hour gap
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X), 1):
        model = RandomForestClassifier(n_estimators=30 if fast_mode else 50, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score, used_fallback = _safe_auc(y[test_idx], y_pred_proba, y_pred)
        if np.isnan(score):
            print(f"Fold {fold}: Skipped (single class)")
            continue
        scores.append(score)
        
        metric_name = "Accuracy (fallback)" if used_fallback else "ROC-AUC"
        print(f"Fold {fold}: {metric_name} = {score:.4f} "
              f"(train: {len(train_idx)}, test: {len(test_idx)})")
    
    if scores:
        results['Purged K-Fold'] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        print(f"\nMean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return results


# Backward-compat name expected by tests
def temporal_cv_comparison(X, y, timestamps, patient_ids):
    return evaluate_temporal_cv_strategies(X, y, timestamps, patient_ids, fast_mode=FAST_MODE)


def visualize_temporal_results(results, df):
    """
    Visualize temporal CV results and patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: CV Strategy Comparison
    ax1 = axes[0, 0]
    methods = list(results.keys())
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    
    x_pos = np.arange(len(methods))
    colors_list = ['#870052', '#FF876F', '#4CAF50', '#2196F3', '#FFA500']
    
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=10,
                   color=colors_list[:len(methods)], alpha=0.7,
                   edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('CV Strategy')
    ax1.set_ylabel('Mean ROC-AUC Score')
    ax1.set_title('Temporal CV Strategy Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    # Plot 2: Learning Curve (for Expanding Window)
    ax2 = axes[0, 1]
    if 'Expanding Window' in results and 'train_sizes' in results['Expanding Window']:
        train_sizes = results['Expanding Window']['train_sizes']
        scores = results['Expanding Window']['scores']
        
        ax2.plot(train_sizes, scores, 'o-', color='#870052', linewidth=2, markersize=8)
        ax2.fill_between(train_sizes,
                         [s - 0.02 for s in scores],
                         [s + 0.02 for s in scores],
                         alpha=0.3, color='#870052')
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('ROC-AUC Score')
        ax2.set_title('Learning Curve: Expanding Window CV')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temporal Pattern of Deterioration
    ax3 = axes[1, 0]
    hourly_deterioration = df.groupby('hour_in_stay')['deteriorates_6h'].mean()
    ax3.plot(hourly_deterioration.index[:72], hourly_deterioration.values[:72],
             color='#870052', linewidth=2)
    ax3.fill_between(hourly_deterioration.index[:72], 0, hourly_deterioration.values[:72],
                      alpha=0.3, color='#870052')
    ax3.set_xlabel('Hours in ICU')
    ax3.set_ylabel('Deterioration Rate')
    ax3.set_title('Temporal Pattern of Patient Deterioration')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance Over Time
    ax4 = axes[1, 1]
    # Simulate feature importance changing over time
    time_windows = ['0-24h', '24-48h', '48-72h', '>72h']
    features = ['Heart Rate', 'BP', 'SpO2', 'Lactate']
    importance_data = np.random.rand(4, 4) * 0.3 + 0.1
    importance_data = importance_data / importance_data.sum(axis=0)
    
    x = np.arange(len(time_windows))
    width = 0.2
    
    for i, feature in enumerate(features):
        ax4.bar(x + i * width, importance_data[i], width,
                label=feature, alpha=0.8)
    
    ax4.set_xlabel('Time Window')
    ax4.set_ylabel('Feature Importance')
    ax4.set_title('Feature Importance Evolution')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(time_windows)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def compare_models_temporal(X, y, timestamps, fast_mode=True):
    """
    Compare different models using temporal CV
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON WITH TEMPORAL CV")
    print("="*60)
    
    if fast_mode:
        print("\nSkipping model comparison in fast mode. Set TRUSTCV_FULL_EXAMPLE=1 for full run.")
        return {}

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Use Time Series Split for fair comparison
    tss = TimeSeriesSplit(n_splits=5)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        scores = []
        
        for train_idx, test_idx in tss.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
            score, _ = _safe_auc(y[test_idx], y_pred_proba, y_pred)
            if np.isnan(score):
                continue
            scores.append(score)
        
        results[name] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        
        if scores:
            print(f"  Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['mean'])
    print(f"\n🏆 Best Model for ICU Monitoring: {best_model}")
    print(f"   Score: {results[best_model]['mean']:.4f} ± {results[best_model]['std']:.4f}")
    
    return results


def main():
    """
    Main execution function
    """
    print("="*60)
    print("ICU PATIENT MONITORING - TEMPORAL CV")
    print("="*60)
    
    # Create ICU monitoring dataset
    print("\n📊 Creating synthetic ICU monitoring dataset...")
    n_patients = 40 if FAST_MODE else 100
    avg_hours = 48 if FAST_MODE else 72
    df = create_icu_monitoring_data(n_patients=n_patients, avg_hours=avg_hours)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Deterioration rate: {df['deteriorates_6h'].mean():.2%}")
    
    # Engineer features
    print("\n🔧 Engineering temporal features...")
    df_features = engineer_features(df)
    
    # Prepare features and target
    feature_cols = [col for col in df_features.columns 
                   if col not in ['patient_id', 'timestamp', 'deteriorates_6h']]
    
    X = df_features[feature_cols].fillna(0).values
    y = df_features['deteriorates_6h'].values
    timestamps = df_features['timestamp'].values
    patient_ids = df_features['patient_id'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Selected features: {len(feature_cols)}")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Evaluate temporal CV strategies
    results = evaluate_temporal_cv_strategies(X, y, timestamps, patient_ids, fast_mode=FAST_MODE)
    
    # Visualize results
    if SHOW_PLOTS:
        visualize_temporal_results(results, df_features)
    else:
        print("\n📈 Plotting skipped (set TRUSTCV_PLOT=1 to enable).")
    
    # Compare models
    model_results = compare_models_temporal(X, y, timestamps, fast_mode=FAST_MODE)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR ICU MONITORING")
    print("="*60)
    print("1. Time Series Split respects temporal order → no future leakage")
    print("2. Rolling Window maintains constant training size → stable performance")
    print("3. Expanding Window uses all historical data → improving accuracy")
    print("4. Blocked CV preserves daily patterns → realistic evaluation")
    print("5. Purged K-Fold prevents information leakage → conservative estimates")
    print("\n⚠️  Never use random splits for temporal medical data!")
    print("✅ Always validate temporal models with future data!")


if __name__ == "__main__":
    main()
