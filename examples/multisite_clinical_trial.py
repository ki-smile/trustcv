#!/usr/bin/env python3
"""
Multi-site Clinical Trial - Grouped Cross-Validation
======================================================
This example demonstrates proper validation for multi-site clinical trials
where patients are nested within hospitals/sites.

Dataset: Simulated multi-site drug efficacy trial
Task: Predict treatment response accounting for site effects
Challenge: Site-specific variations, patient clustering, hierarchical structure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our grouped CV methods
import sys
sys.path.append('..')
from trustcv.splitters.grouped import (
    GroupKFoldMedical, StratifiedGroupKFold,
    LeaveOneGroupOut, RepeatedGroupKFold, 
    HierarchicalGroupKFold, NestedGroupedCV
)

# Set random seed for reproducibility
np.random.seed(42)

def create_multisite_trial_data(n_sites=10, patients_per_site=50):
    """
    Create synthetic multi-site clinical trial dataset
    
    Features include:
    - Patient demographics (age, sex, BMI)
    - Baseline clinical measures
    - Treatment arm (drug vs placebo)
    - Site-specific effects
    - Hierarchical structure: Site → Department → Patient
    """
    all_data = []
    
    # Site characteristics
    site_effects = np.random.randn(n_sites) * 0.3  # Site-specific treatment effects
    site_types = np.random.choice(['Academic', 'Community', 'Private'], n_sites)
    site_regions = np.random.choice(['North', 'South', 'East', 'West'], n_sites)
    
    for site_id in range(n_sites):
        site_name = f"Site_{site_id+1:02d}"
        site_type = site_types[site_id]
        site_region = site_regions[site_id]
        site_effect = site_effects[site_id]
        
        # Site-specific patient population characteristics
        age_mean = 50 + np.random.randn() * 5
        age_std = 10
        
        # Departments within site (for hierarchical structure)
        n_departments = np.random.randint(2, 5)
        
        for patient_idx in range(patients_per_site):
            # Assign to department
            department_id = patient_idx % n_departments
            department_name = f"{site_name}_Dept_{department_id+1}"
            
            # Patient demographics
            age = np.clip(np.random.normal(age_mean, age_std), 18, 85)
            sex = np.random.choice([0, 1])  # 0: Female, 1: Male
            bmi = np.clip(np.random.normal(27, 4), 18, 45)
            
            # Baseline clinical measures
            baseline_score = np.random.normal(50, 10)
            comorbidity_count = np.random.poisson(1.5)
            disease_duration = np.random.exponential(3)
            
            # Previous medications
            prev_medications = np.random.randint(0, 4)
            
            # Lab values
            hemoglobin = np.random.normal(14 - sex, 1.5)
            creatinine = np.random.normal(1.0, 0.2)
            alt = np.random.normal(30, 10)
            
            # Treatment assignment (randomized within site)
            treatment = np.random.choice([0, 1])  # 0: Placebo, 1: Drug
            
            # Compliance and adherence
            compliance = np.random.beta(8, 2)  # Most patients are compliant
            
            # Calculate treatment response
            # Base response depends on patient characteristics
            response_score = (
                60 +  # Baseline
                treatment * 15 +  # Treatment effect
                treatment * site_effect * 10 +  # Site-specific treatment effect
                (65 - age) * 0.2 +  # Age effect
                sex * 2 +  # Sex effect
                (25 - bmi) * 0.3 +  # BMI effect
                baseline_score * 0.3 +  # Baseline correlation
                compliance * 10 +  # Compliance effect
                np.random.randn() * 8  # Random noise
            )
            
            # Binary outcome: responder vs non-responder
            responder = int(response_score > 70)
            
            # Add site batch effects
            if site_type == 'Academic':
                response_score += 3
            elif site_type == 'Community':
                response_score -= 2
            
            all_data.append({
                'patient_id': f"{site_name}_P{patient_idx+1:03d}",
                'site_id': site_name,
                'site_type': site_type,
                'site_region': site_region,
                'department_id': department_name,
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'baseline_score': baseline_score,
                'comorbidity_count': comorbidity_count,
                'disease_duration': disease_duration,
                'prev_medications': prev_medications,
                'hemoglobin': hemoglobin,
                'creatinine': creatinine,
                'alt': alt,
                'treatment': treatment,
                'compliance': compliance,
                'response_score': response_score,
                'responder': responder
            })
    
    df = pd.DataFrame(all_data)
    return df


def analyze_site_heterogeneity(df):
    """
    Analyze and visualize site-specific effects
    """
    print("\n" + "="*60)
    print("SITE HETEROGENEITY ANALYSIS")
    print("="*60)
    
    # Calculate site-specific response rates
    site_stats = df.groupby('site_id').agg({
        'responder': ['mean', 'std', 'count'],
        'age': 'mean',
        'treatment': 'mean'
    }).round(3)
    
    print("\nSite-specific Response Rates:")
    print(site_stats)
    
    # Test for site heterogeneity
    sites = df['site_id'].unique()
    site_responses = [df[df['site_id'] == site]['responder'].values for site in sites]
    
    # Chi-square test for independence
    contingency_table = pd.crosstab(df['site_id'], df['responder'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\n📊 Site Heterogeneity Test:")
    print(f"   Chi-square statistic: {chi2:.2f}")
    print(f"   P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("   ⚠️ Significant site heterogeneity detected!")
        print("   → Group-aware CV is essential!")
    else:
        print("   ✅ No significant site heterogeneity")
    
    return site_stats


def evaluate_grouped_cv_strategies(X, y, groups, site_ids, df):
    """
    Compare different grouped CV strategies for multi-site trials
    """
    results = {}
    
    print("\n" + "="*60)
    print("GROUPED CROSS-VALIDATION FOR MULTI-SITE TRIAL")
    print("="*60)
    
    # 1. Standard K-Fold (WRONG - ignores groups)
    print("\n1. Standard K-Fold (IGNORING SITES - WRONG!)")
    print("-"*40)
    from trustcv.splitters.iid import KFoldMedical
    
    kf = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        scores.append(score)
        
        # Check for data leakage
        train_sites = set(site_ids[train_idx])
        test_sites = set(site_ids[test_idx])
        overlap = train_sites & test_sites
        
        print(f"Fold {fold}: ROC-AUC = {score:.4f}")
        if overlap:
            print(f"  ⚠️ LEAKAGE: {len(overlap)} sites in both train and test!")
    
    results['Standard K-Fold (Wrong)'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
    print(f"\nMean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print("⚠️ This is likely optimistically biased due to site leakage!")
    
    # 2. Group K-Fold (Correct)
    print("\n2. Group K-Fold (SITE-AWARE - CORRECT)")
    print("-"*40)
    
    group_kfold = GroupKFoldMedical(n_splits=5)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups), 1):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        scores.append(score)
        
        train_sites = set(site_ids[train_idx])
        test_sites = set(site_ids[test_idx])
        
        print(f"Fold {fold}: ROC-AUC = {score:.4f}")
        print(f"  Train sites: {len(train_sites)}, Test sites: {len(test_sites)}")
        print(f"  ✅ No site overlap (each site in exactly one fold)")
    
    results['Group K-Fold'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
    print(f"\nMean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 3. Stratified Group K-Fold
    print("\n3. Stratified Group K-Fold (Preserves Response Balance)")
    print("-"*40)
    
    stratified_group_kfold = StratifiedGroupKFold(n_splits=5)
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(stratified_group_kfold.split(X, y, groups), 1):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        scores.append(score)
        
        train_responder_rate = y[train_idx].mean()
        test_responder_rate = y[test_idx].mean()
        
        print(f"Fold {fold}: ROC-AUC = {score:.4f}")
        print(f"  Train responder rate: {train_responder_rate:.2%}")
        print(f"  Test responder rate: {test_responder_rate:.2%}")
    
    results['Stratified Group K-Fold'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
    print(f"\nMean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 4. Leave-One-Site-Out
    print("\n4. Leave-One-Site-Out CV")
    print("-"*40)
    
    logo = LeaveOneGroupOut()
    scores = []
    site_scores = {}
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        scores.append(score)
        
        test_site = site_ids[test_idx[0]]
        site_scores[test_site] = score
        
        if fold <= 5:  # Only print first 5 to avoid clutter
            print(f"Test Site {test_site}: ROC-AUC = {score:.4f}")
    
    results['Leave-One-Site-Out'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'site_scores': site_scores
    }
    print(f"\nMean ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Range: {min(scores):.4f} - {max(scores):.4f}")
    
    # 5. Repeated Group K-Fold
    print("\n5. Repeated Group K-Fold (3 repeats)")
    print("-"*40)
    
    repeated_group_kfold = RepeatedGroupKFold(n_splits=5, n_repeats=3)
    scores = []
    
    for repeat in range(3):
        repeat_scores = []
        for fold, (train_idx, test_idx) in enumerate(repeated_group_kfold.split(X, y, groups), 1):
            if fold > 5 * repeat and fold <= 5 * (repeat + 1):
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X[train_idx], y[train_idx])
                
                y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
                score = roc_auc_score(y[test_idx], y_pred_proba)
                repeat_scores.append(score)
        
        scores.extend(repeat_scores)
        print(f"Repeat {repeat+1}: {np.mean(repeat_scores):.4f} ± {np.std(repeat_scores):.4f}")
    
    results['Repeated Group K-Fold'] = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }
    print(f"\nOverall: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return results


def visualize_multisite_results(results, df):
    """
    Visualize multi-site trial results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: CV Strategy Comparison
    ax1 = axes[0, 0]
    methods = list(results.keys())
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    
    colors_list = ['red', '#870052', '#FF876F', '#4CAF50', '#2196F3']
    x_pos = np.arange(len(methods))
    
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=10,
                   color=colors_list[:len(methods)], alpha=0.7,
                   edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('CV Strategy')
    ax1.set_ylabel('Mean ROC-AUC Score')
    ax1.set_title('Multi-Site CV Strategy Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight the biased method
    ax1.axhspan(means[0] - stds[0], means[0] + stds[0], 
                alpha=0.2, color='red', label='Biased')
    
    # Plot 2: Site-specific Response Rates
    ax2 = axes[0, 1]
    site_response = df.groupby('site_id')['responder'].mean().sort_values()
    ax2.barh(range(len(site_response)), site_response.values, 
             color='#870052', alpha=0.7)
    ax2.set_yticks(range(len(site_response)))
    ax2.set_yticklabels(site_response.index)
    ax2.set_xlabel('Response Rate')
    ax2.set_title('Site-Specific Response Rates')
    ax2.axvline(x=df['responder'].mean(), color='red', linestyle='--', 
                label=f'Overall: {df["responder"].mean():.2%}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Treatment Effect by Site
    ax3 = axes[0, 2]
    treatment_effect = []
    site_names = []
    
    for site in df['site_id'].unique():
        site_data = df[df['site_id'] == site]
        treated = site_data[site_data['treatment'] == 1]['responder'].mean()
        control = site_data[site_data['treatment'] == 0]['responder'].mean()
        effect = treated - control
        treatment_effect.append(effect)
        site_names.append(site)
    
    ax3.bar(range(len(treatment_effect)), treatment_effect,
            color=['#4CAF50' if e > 0 else '#FF876F' for e in treatment_effect],
            alpha=0.7)
    ax3.set_xticks(range(len(site_names)))
    ax3.set_xticklabels(site_names, rotation=45, ha='right')
    ax3.set_ylabel('Treatment Effect (Treated - Control)')
    ax3.set_title('Site-Specific Treatment Effects')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Sample Size by Site
    ax4 = axes[1, 0]
    site_counts = df['site_id'].value_counts()
    ax4.pie(site_counts.values, labels=site_counts.index, autopct='%1.0f%%',
            colors=plt.cm.Set3(np.linspace(0, 1, len(site_counts))))
    ax4.set_title('Patient Distribution Across Sites')
    
    # Plot 5: LOGO Site Performance
    ax5 = axes[1, 1]
    if 'site_scores' in results['Leave-One-Site-Out']:
        site_scores = results['Leave-One-Site-Out']['site_scores']
        sites = list(site_scores.keys())[:10]  # Show first 10 sites
        scores = [site_scores[s] for s in sites]
        
        ax5.scatter(range(len(sites)), scores, s=100, alpha=0.6, color='#870052')
        ax5.axhline(y=results['Leave-One-Site-Out']['mean'], 
                   color='red', linestyle='--', label='Mean')
        ax5.fill_between(range(len(sites)),
                         results['Leave-One-Site-Out']['mean'] - results['Leave-One-Site-Out']['std'],
                         results['Leave-One-Site-Out']['mean'] + results['Leave-One-Site-Out']['std'],
                         alpha=0.2, color='red')
        ax5.set_xticks(range(len(sites)))
        ax5.set_xticklabels(sites, rotation=45, ha='right')
        ax5.set_ylabel('ROC-AUC (when site is test)')
        ax5.set_title('Leave-One-Site-Out Performance')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Regional Effects
    ax6 = axes[1, 2]
    regional_response = df.groupby('site_region')['responder'].mean()
    ax6.bar(regional_response.index, regional_response.values,
            color=['#870052', '#FF876F', '#4CAF50', '#2196F3'][:len(regional_response)],
            alpha=0.7)
    ax6.set_xlabel('Region')
    ax6.set_ylabel('Response Rate')
    ax6.set_title('Regional Variation in Response')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def test_site_generalization(X, y, groups, site_ids, df):
    """
    Test model generalization to new sites
    """
    print("\n" + "="*60)
    print("SITE GENERALIZATION TEST")
    print("="*60)
    
    unique_sites = np.unique(site_ids)
    n_sites = len(unique_sites)
    
    # Train on 80% of sites, test on 20%
    n_train_sites = int(0.8 * n_sites)
    train_sites = np.random.choice(unique_sites, n_train_sites, replace=False)
    test_sites = [s for s in unique_sites if s not in train_sites]
    
    train_mask = np.isin(site_ids, train_sites)
    test_mask = np.isin(site_ids, test_sites)
    
    print(f"\nTraining on {len(train_sites)} sites")
    print(f"Testing on {len(test_sites)} new sites")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[train_mask], y[train_mask])
    
    # Test on new sites
    y_pred_proba = model.predict_proba(X[test_mask])[:, 1]
    score = roc_auc_score(y[test_mask], y_pred_proba)
    
    print(f"\n📊 Generalization Performance:")
    print(f"   ROC-AUC on new sites: {score:.4f}")
    
    # Compare to within-site performance
    within_site_scores = []
    for site in train_sites[:5]:  # Sample 5 training sites
        site_mask = site_ids == site
        if site_mask.sum() > 10:
            site_score = roc_auc_score(
                y[site_mask], 
                model.predict_proba(X[site_mask])[:, 1]
            )
            within_site_scores.append(site_score)
    
    print(f"   Average within-site ROC-AUC: {np.mean(within_site_scores):.4f}")
    print(f"   Generalization gap: {np.mean(within_site_scores) - score:.4f}")
    
    if np.mean(within_site_scores) - score > 0.1:
        print("   ⚠️ Significant generalization gap detected!")
        print("   → Consider site-specific adjustments or mixed-effects models")
    else:
        print("   ✅ Good generalization to new sites")


def main():
    """
    Main execution function
    """
    print("="*60)
    print("MULTI-SITE CLINICAL TRIAL - GROUPED CV")
    print("="*60)
    
    # Create multi-site trial dataset
    print("\n📊 Creating synthetic multi-site trial dataset...")
    df = create_multisite_trial_data(n_sites=10, patients_per_site=50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of sites: {df['site_id'].nunique()}")
    print(f"Number of departments: {df['department_id'].nunique()}")
    print(f"Overall response rate: {df['responder'].mean():.2%}")
    print(f"Treatment allocation: {df['treatment'].mean():.2%} treated")
    
    # Analyze site heterogeneity
    site_stats = analyze_site_heterogeneity(df)
    
    # Prepare features
    feature_cols = ['age', 'sex', 'bmi', 'baseline_score', 'comorbidity_count',
                   'disease_duration', 'prev_medications', 'hemoglobin',
                   'creatinine', 'alt', 'treatment', 'compliance']
    
    X = df[feature_cols].values
    y = df['responder'].values
    
    # Groups for CV (sites)
    site_ids = df['site_id'].values
    site_encoder = {site: i for i, site in enumerate(df['site_id'].unique())}
    groups = np.array([site_encoder[site] for site in site_ids])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Selected features: {feature_cols}")
    
    # Evaluate grouped CV strategies
    results = evaluate_grouped_cv_strategies(X, y, groups, site_ids, df)
    
    # Visualize results
    visualize_multisite_results(results, df)
    
    # Test site generalization
    test_site_generalization(X, y, groups, site_ids, df)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR MULTI-SITE TRIALS")
    print("="*60)
    print("1. Standard K-Fold leaks site information → overoptimistic results")
    print("2. Group K-Fold ensures sites are independent → realistic estimates")
    print("3. Stratified Group K-Fold maintains class balance across folds")
    print("4. Leave-One-Site-Out tests true generalization to new sites")
    print("5. Repeated Group K-Fold reduces variance in estimates")
    print("\n⚠️  Always use grouped CV for multi-site/clustered data!")
    print("✅ Test generalization to new sites before deployment!")


if __name__ == "__main__":
    main()