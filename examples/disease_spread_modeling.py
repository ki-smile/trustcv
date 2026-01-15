#!/usr/bin/env python3
"""
Disease Spread Modeling - Spatial Cross-Validation
===================================================
This example demonstrates proper validation of epidemiological models
for disease spread prediction using spatial cross-validation methods.

Dataset: Simulated infectious disease outbreak data
Task: Predict disease spread to new geographic regions
Challenge: Spatial autocorrelation, population mobility, environmental factors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from trustcv import KFoldMedical
import warnings
warnings.filterwarnings('ignore')

# Import our spatial CV methods
import sys
sys.path.append('..')
from trustcv.splitters.spatial import (
    SpatialBlockCV, BufferedSpatialCV, 
    SpatiotemporalBlockCV, EnvironmentalHealthCV
)

# Set random seed for reproducibility
np.random.seed(42)

def create_disease_spread_data(n_locations=200, n_time_points=30):
    """
    Create synthetic disease spread dataset with realistic epidemiological patterns
    
    Features include:
    - Geographic coordinates (lat/lon)
    - Population demographics
    - Environmental factors
    - Mobility patterns
    - Previous disease incidence
    - Socioeconomic indicators
    """
    
    print("🦠 Generating realistic disease outbreak scenario...")
    
    # Generate geographic locations (focused on a region like Scandinavia)
    lat_center, lon_center = 60.0, 15.0  # Approximate center of Scandinavia
    lat_range, lon_range = 8.0, 12.0     # Degree range
    
    # Create clustered locations (representing cities/municipalities)
    n_clusters = 8
    cluster_centers = []
    for _ in range(n_clusters):
        cluster_lat = lat_center + np.random.uniform(-lat_range/2, lat_range/2)
        cluster_lon = lon_center + np.random.uniform(-lon_range/2, lon_range/2)
        cluster_centers.append([cluster_lat, cluster_lon])
    
    all_data = []
    
    for location_id in range(n_locations):
        # Assign to nearest cluster
        cluster_id = location_id % n_clusters
        cluster_center = cluster_centers[cluster_id]
        
        # Add noise around cluster center
        lat = cluster_center[0] + np.random.normal(0, 0.5)
        lon = cluster_center[1] + np.random.normal(0, 0.8)
        
        # Location characteristics
        population = np.random.lognormal(8, 1.5)  # Log-normal distribution for city sizes
        population = max(1000, min(1000000, int(population)))  # Reasonable bounds
        
        population_density = population / (np.random.uniform(10, 1000))  # per km²
        
        # Demographics
        median_age = np.random.normal(42, 8)
        elderly_proportion = np.random.beta(2, 6)  # Skewed towards lower proportions
        
        # Socioeconomic factors
        income_index = np.random.beta(2, 3) * 100  # Index 0-100
        education_level = np.random.beta(3, 2) * 100  # Index 0-100
        
        # Healthcare capacity
        hospitals_per_capita = np.random.exponential(0.5) / 1000  # per 1000 people
        icu_beds_per_capita = np.random.exponential(0.1) / 1000
        
        # Environmental factors
        temperature = np.random.normal(8, 12)  # Average temperature
        humidity = np.random.uniform(40, 90)   # Relative humidity %
        air_quality_index = np.random.exponential(50)  # Lower is better
        
        # Transportation connectivity
        airport_proximity = np.random.exponential(100)  # km to nearest major airport
        highway_density = np.random.exponential(0.5)    # km of highway per km²
        
        # Generate time series data for this location
        for time_point in range(n_time_points):
            date = datetime(2024, 1, 1) + timedelta(days=time_point)
            
            # Seasonal effects
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Mobility patterns (higher on weekends, holidays)
            weekday = date.weekday()
            weekend_factor = 1.3 if weekday >= 5 else 1.0
            
            # Previous disease history (lagged effect)
            if time_point == 0:
                previous_cases = np.random.poisson(max(1, population / 100000))
            else:
                # Cases influenced by previous days and spatial neighbors
                base_cases = max(1, int(population / 100000))
                previous_influence = 0.3  # 30% influence from previous day
                if time_point > 0:
                    prev_cases = all_data[-1]['daily_cases'] if all_data else base_cases
                    previous_cases = max(1, int(base_cases + prev_cases * previous_influence))
                else:
                    previous_cases = base_cases
            
            # Calculate transmission risk factors
            transmission_risk = (
                1.0 +  # Base risk
                0.2 * (population_density / 1000) +  # Density effect
                0.15 * (elderly_proportion * 100) +  # Age vulnerability
                -0.1 * (income_index / 100) +        # Socioeconomic protection
                -0.2 * (hospitals_per_capita * 1000) +  # Healthcare capacity
                0.1 * (air_quality_index / 100) +    # Environmental factors
                0.3 * seasonal_factor +              # Seasonal variation
                0.2 * weekend_factor +               # Mobility patterns
                -0.05 * (education_level / 100) +   # Education effect
                np.random.normal(0, 0.2)             # Random noise
            )
            
            transmission_risk = max(0.1, transmission_risk)  # Ensure positive
            
            # Generate daily new cases (target variable)
            expected_cases = previous_cases * transmission_risk
            daily_cases = max(0, int(np.random.poisson(expected_cases)))
            
            # Calculate reproduction number (R)
            reproduction_number = max(0.5, transmission_risk + np.random.normal(0, 0.1))
            
            # Case fatality rate (influenced by healthcare capacity and demographics)
            cfr = max(0.001, min(0.1, 
                0.02 + 
                0.01 * elderly_proportion - 
                0.005 * (hospitals_per_capita * 1000) +
                np.random.normal(0, 0.002)
            ))
            
            all_data.append({
                'location_id': location_id,
                'date': date,
                'time_point': time_point,
                'latitude': lat,
                'longitude': lon,
                'cluster_id': cluster_id,
                
                # Population characteristics
                'population': population,
                'population_density': population_density,
                'median_age': median_age,
                'elderly_proportion': elderly_proportion,
                
                # Socioeconomic
                'income_index': income_index,
                'education_level': education_level,
                
                # Healthcare
                'hospitals_per_capita': hospitals_per_capita,
                'icu_beds_per_capita': icu_beds_per_capita,
                
                # Environmental
                'temperature': temperature + np.random.normal(0, 2),  # Daily variation
                'humidity': humidity + np.random.normal(0, 5),
                'air_quality_index': air_quality_index + np.random.normal(0, 10),
                
                # Transportation
                'airport_proximity': airport_proximity,
                'highway_density': highway_density,
                
                # Time-varying factors
                'seasonal_factor': seasonal_factor,
                'weekend_factor': weekend_factor,
                
                # Previous epidemiological data
                'previous_cases': previous_cases,
                
                # Target variables
                'daily_cases': daily_cases,
                'reproduction_number': reproduction_number,
                'case_fatality_rate': cfr
            })
    
    df = pd.DataFrame(all_data)
    
    print(f"📊 Generated dataset: {df.shape}")
    print(f"   Locations: {df['location_id'].nunique()}")
    print(f"   Time points: {df['time_point'].nunique()}")
    print(f"   Total cases: {df['daily_cases'].sum():,}")
    print(f"   Mean daily cases per location: {df['daily_cases'].mean():.1f}")
    
    return df


def analyze_spatial_autocorrelation(df):
    """
    Analyze and visualize spatial autocorrelation in disease data
    """
    print("\n" + "="*60)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate spatial correlation matrix
    latest_time = df['time_point'].max()
    latest_data = df[df['time_point'] == latest_time].copy()
    
    # Calculate distances between all pairs of locations
    coords = latest_data[['longitude', 'latitude']].values
    distances = squareform(pdist(coords, metric='euclidean'))
    
    # Calculate correlation of daily cases with distance
    cases = latest_data['daily_cases'].values
    
    # Spatial autocorrelation analysis
    correlations = []
    distance_bins = np.linspace(0, distances.max(), 20)
    
    for i in range(len(distance_bins)-1):
        min_dist, max_dist = distance_bins[i], distance_bins[i+1]
        
        # Find pairs within this distance range
        mask = (distances >= min_dist) & (distances < max_dist)
        
        if mask.sum() > 10:  # Ensure sufficient pairs
            pairs_i, pairs_j = np.where(mask)
            case_pairs = [(cases[i], cases[j]) for i, j in zip(pairs_i, pairs_j) if i != j]
            
            if len(case_pairs) > 5:
                cases_i, cases_j = zip(*case_pairs)
                correlation = np.corrcoef(cases_i, cases_j)[0, 1]
                correlations.append({
                    'distance_bin': (min_dist + max_dist) / 2,
                    'correlation': correlation,
                    'n_pairs': len(case_pairs)
                })
    
    # Visualize spatial patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Geographic distribution of cases
    ax1 = axes[0, 0]
    scatter = ax1.scatter(latest_data['longitude'], latest_data['latitude'], 
                         c=latest_data['daily_cases'], 
                         s=latest_data['population']/5000,  # Size by population
                         cmap='Reds', alpha=0.7)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Disease Cases by Location\n(size = population)')
    plt.colorbar(scatter, ax=ax1, label='Daily Cases')
    
    # Plot 2: Spatial autocorrelation
    ax2 = axes[0, 1]
    if correlations:
        corr_df = pd.DataFrame(correlations)
        ax2.plot(corr_df['distance_bin'], corr_df['correlation'], 'o-', 
                color='#870052', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Distance (degrees)')
        ax2.set_ylabel('Spatial Correlation')
        ax2.set_title('Spatial Autocorrelation of Disease Cases')
        ax2.grid(True, alpha=0.3)
        
        # Highlight significant autocorrelation
        significant_range = corr_df[corr_df['correlation'] > 0.1]['distance_bin'].max()
        if not pd.isna(significant_range):
            ax2.axvline(x=significant_range, color='red', linestyle=':', alpha=0.7, 
                       label=f'Autocorr. range ≈ {significant_range:.1f}°')
            ax2.legend()
    
    # Plot 3: Cases vs Population Density
    ax3 = axes[1, 0]
    ax3.scatter(latest_data['population_density'], latest_data['daily_cases'], 
               alpha=0.6, color='#FF876F')
    ax3.set_xlabel('Population Density (per km²)')
    ax3.set_ylabel('Daily Cases')
    ax3.set_title('Cases vs Population Density')
    ax3.set_xscale('log')
    
    # Add trend line
    log_density = np.log10(latest_data['population_density'])
    z = np.polyfit(log_density, latest_data['daily_cases'], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log10(latest_data['population_density'].min()), 
                         np.log10(latest_data['population_density'].max()), 100)
    ax3.plot(x_trend, p(np.log10(x_trend)), "--", color='red', alpha=0.8)
    
    # Plot 4: Cluster-based analysis
    ax4 = axes[1, 1]
    cluster_stats = latest_data.groupby('cluster_id')['daily_cases'].agg(['mean', 'std']).reset_index()
    bars = ax4.bar(cluster_stats['cluster_id'], cluster_stats['mean'], 
                   yerr=cluster_stats['std'], capsize=5,
                   color=['#870052', '#FF876F', '#4CAF50', '#2196F3', '#FFA500', 
                         '#9C27B0', '#FF5722', '#795548'][:len(cluster_stats)])
    ax4.set_xlabel('Geographic Cluster')
    ax4.set_ylabel('Mean Daily Cases')
    ax4.set_title('Disease Burden by Geographic Cluster')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print(f"\n📊 Spatial Analysis Summary:")
    if correlations:
        max_corr = max(correlations, key=lambda x: x['correlation'])
        print(f"   Maximum spatial correlation: {max_corr['correlation']:.3f}")
        print(f"   at distance: {max_corr['distance_bin']:.2f} degrees")
        
        # Estimate range of spatial autocorrelation
        significant_corrs = [c for c in correlations if c['correlation'] > 0.1]
        if significant_corrs:
            autocorr_range = max(c['distance_bin'] for c in significant_corrs)
            print(f"   Spatial autocorrelation range: ~{autocorr_range:.1f} degrees")
            print(f"   Recommended buffer size: {autocorr_range * 1.2:.2f} degrees")
        else:
            print(f"   No significant spatial autocorrelation detected")
    
    return correlations


def evaluate_spatial_cv_strategies(X, y, coordinates, timestamps, location_ids):
    """
    Compare different spatial CV strategies for disease spread modeling
    """
    results = {}
    
    print("\n" + "="*60)
    print("SPATIAL CROSS-VALIDATION FOR DISEASE SPREAD MODELING")
    print("="*60)
    
    # 1. Standard K-Fold (WRONG - ignores spatial structure)
    print("\n1. Standard K-Fold (Ignoring Spatial Structure - WRONG!)")
    print("-"*50)
    
    kfold = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
    standard_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred = model.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred)
        
        standard_scores.append({'rmse': rmse, 'r2': r2})
        print(f"Fold {fold}: RMSE = {rmse:.3f}, R² = {r2:.3f}")
    
    results['Standard K-Fold (BIASED)'] = {
        'rmse': [s['rmse'] for s in standard_scores],
        'r2': [s['r2'] for s in standard_scores]
    }
    
    avg_rmse = np.mean([s['rmse'] for s in standard_scores])
    avg_r2 = np.mean([s['r2'] for s in standard_scores])
    print(f"Mean: RMSE = {avg_rmse:.3f}, R² = {avg_r2:.3f}")
    print("⚠️ This is likely OPTIMISTICALLY BIASED due to spatial autocorrelation!")
    
    # 2. Spatial Block Cross-Validation
    print("\n2. Spatial Block Cross-Validation")
    print("-"*50)
    
    spatial_cv = SpatialBlockCV(n_splits=5, block_size=1.5)  # 1.5 degree blocks
    spatial_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(spatial_cv.split(X, coordinates=coordinates), 1):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred = model.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred)
        
        spatial_scores.append({'rmse': rmse, 'r2': r2})
        
        # Geographic coverage
        train_coords = coordinates[train_idx]
        test_coords = coordinates[test_idx]
        
        print(f"Fold {fold}: RMSE = {rmse:.3f}, R² = {r2:.3f}")
        print(f"  Train region: Lat {train_coords[:, 1].min():.1f}-{train_coords[:, 1].max():.1f}, "
              f"Lon {train_coords[:, 0].min():.1f}-{train_coords[:, 0].max():.1f}")
        print(f"  Test region:  Lat {test_coords[:, 1].min():.1f}-{test_coords[:, 1].max():.1f}, "
              f"Lon {test_coords[:, 0].min():.1f}-{test_coords[:, 0].max():.1f}")
    
    results['Spatial Block CV'] = {
        'rmse': [s['rmse'] for s in spatial_scores],
        'r2': [s['r2'] for s in spatial_scores]
    }
    
    avg_rmse = np.mean([s['rmse'] for s in spatial_scores])
    avg_r2 = np.mean([s['r2'] for s in spatial_scores])
    print(f"Mean: RMSE = {avg_rmse:.3f}, R² = {avg_r2:.3f}")
    print("✅ More realistic estimate accounting for spatial structure")
    
    # 3. Buffered Spatial Cross-Validation
    print("\n3. Buffered Spatial Cross-Validation")
    print("-"*50)
    
    buffered_cv = BufferedSpatialCV(n_splits=5, buffer_size=0.5)  # 0.5 degree buffer
    buffered_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(buffered_cv.split(X, coordinates=coordinates), 1):
        if len(test_idx) < 10:  # Skip folds with insufficient data
            print(f"Fold {fold}: Skipped (insufficient test samples: {len(test_idx)})")
            continue
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred = model.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred)
        
        buffered_scores.append({'rmse': rmse, 'r2': r2})
        print(f"Fold {fold}: RMSE = {rmse:.3f}, R² = {r2:.3f} "
              f"(train: {len(train_idx)}, test: {len(test_idx)})")
    
    if buffered_scores:
        results['Buffered Spatial CV'] = {
            'rmse': [s['rmse'] for s in buffered_scores],
            'r2': [s['r2'] for s in buffered_scores]
        }
        avg_rmse = np.mean([s['rmse'] for s in buffered_scores])
        avg_r2 = np.mean([s['r2'] for s in buffered_scores])
        print(f"Mean: RMSE = {avg_rmse:.3f}, R² = {avg_r2:.3f}")
        print("✅ Most conservative estimate with buffer zones")
    
    # 4. Spatiotemporal Cross-Validation
    print("\n4. Spatiotemporal Block Cross-Validation")
    print("-"*50)

    st_cv = SpatiotemporalBlockCV(n_spatial_blocks=3, n_temporal_blocks=3)
    st_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(st_cv.split(X, coordinates=coordinates, timestamps=timestamps), 1):
        if len(test_idx) < 20:
            print(f"Fold {fold}: Skipped (insufficient test samples: {len(test_idx)})")
            continue
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        
        y_pred = model.predict(X[test_idx])
        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred)
        
        st_scores.append({'rmse': rmse, 'r2': r2})
        
        # Temporal coverage
        train_times = set(timestamps[train_idx])
        test_times = set(timestamps[test_idx])
        
        print(f"Fold {fold}: RMSE = {rmse:.3f}, R² = {r2:.3f}")
        print(f"  Train time range: {min(train_times)}-{max(train_times)}")
        print(f"  Test time range: {min(test_times)}-{max(test_times)}")
    
    if st_scores:
        results['Spatiotemporal CV'] = {
            'rmse': [s['rmse'] for s in st_scores],
            'r2': [s['r2'] for s in st_scores]
        }
        avg_rmse = np.mean([s['rmse'] for s in st_scores])
        avg_r2 = np.mean([s['r2'] for s in st_scores])
        print(f"Mean: RMSE = {avg_rmse:.3f}, R² = {avg_r2:.3f}")
        print("✅ Accounts for both spatial and temporal patterns")
    
    return results


def visualize_disease_modeling_results(results, df):
    """
    Visualize disease spread modeling results and spatial patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: RMSE Comparison
    ax1 = axes[0, 0]
    methods = list(results.keys())
    rmse_data = [results[m]['rmse'] for m in methods if results[m]['rmse']]
    rmse_labels = [m for m in methods if results[m]['rmse']]
    
    if rmse_data:
        bp1 = ax1.boxplot(rmse_data, labels=rmse_labels, patch_artist=True)
        colors = ['#FF4444', '#870052', '#FF876F', '#4CAF50', '#2196F3']
        for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax1.set_ylabel('RMSE (Daily Cases)')
    ax1.set_title('Model Performance: RMSE Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² Comparison
    ax2 = axes[0, 1]
    r2_data = [results[m]['r2'] for m in methods if results[m]['r2']]
    r2_labels = [m for m in methods if results[m]['r2']]
    
    if r2_data:
        bp2 = ax2.boxplot(r2_data, labels=r2_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Performance: R² Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Disease spread over time by region
    ax3 = axes[1, 0]
    time_series = df.groupby(['time_point', 'cluster_id'])['daily_cases'].sum().unstack(fill_value=0)
    
    for cluster in time_series.columns[:5]:  # Show top 5 clusters
        ax3.plot(time_series.index, time_series[cluster], 
                linewidth=2, label=f'Cluster {cluster}', alpha=0.8)
    
    ax3.set_xlabel('Time Point (days)')
    ax3.set_ylabel('Daily Cases')
    ax3.set_title('Disease Spread Over Time by Geographic Cluster')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature importance heatmap
    ax4 = axes[1, 1]
    
    # Calculate correlation matrix for key predictive features
    feature_cols = ['population_density', 'median_age', 'elderly_proportion', 
                   'income_index', 'hospitals_per_capita', 'temperature', 
                   'humidity', 'previous_cases']
    
    available_features = [col for col in feature_cols if col in df.columns]
    correlation_data = df[available_features + ['daily_cases']].corr()
    
    # Focus on correlations with daily_cases
    target_corr = correlation_data['daily_cases'].drop('daily_cases')
    
    # Create a simple bar plot instead of heatmap
    ax4.barh(range(len(target_corr)), target_corr.values,
             color=['#4CAF50' if x > 0 else '#FF876F' for x in target_corr.values])
    ax4.set_yticks(range(len(target_corr)))
    ax4.set_yticklabels([f.replace('_', ' ').title() for f in target_corr.index])
    ax4.set_xlabel('Correlation with Daily Cases')
    ax4.set_title('Feature Correlation with Disease Spread')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


def test_model_generalization(X, y, coordinates, location_ids):
    """
    Test model generalization to entirely new geographic regions
    """
    print("\n" + "="*60)
    print("GEOGRAPHIC GENERALIZATION TEST")
    print("="*60)
    
    # Create geographic train/test split
    unique_locations = np.unique(location_ids)
    n_locations = len(unique_locations)
    
    # Use 70% of locations for training, 30% for testing
    n_train_locations = int(0.7 * n_locations)
    train_locations = np.random.choice(unique_locations, n_train_locations, replace=False)
    test_locations = [loc for loc in unique_locations if loc not in train_locations]
    
    train_mask = np.isin(location_ids, train_locations)
    test_mask = np.isin(location_ids, test_locations)
    
    print(f"Training on {len(train_locations)} locations")
    print(f"Testing on {len(test_locations)} new locations")
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X[train_mask], y[train_mask])
    
    # Test on new locations
    y_pred = model.predict(X[test_mask])
    rmse = np.sqrt(mean_squared_error(y[test_mask], y_pred))
    r2 = r2_score(y[test_mask], y_pred)
    mae = mean_absolute_error(y[test_mask], y_pred)
    
    print(f"\n📊 Generalization Performance:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   R²: {r2:.3f}")
    print(f"   MAE: {mae:.3f}")
    
    # Analyze by geographic distance
    train_coords = coordinates[train_mask]
    test_coords = coordinates[test_mask]
    
    # Calculate distances from test locations to nearest training location
    distances_to_training = []
    for test_coord in test_coords:
        min_dist = min(np.linalg.norm(test_coord - train_coord) 
                      for train_coord in train_coords)
        distances_to_training.append(min_dist)
    
    distances_to_training = np.array(distances_to_training)
    
    print(f"\n🗺️ Geographic Analysis:")
    print(f"   Mean distance to training data: {distances_to_training.mean():.2f} degrees")
    print(f"   Max distance to training data: {distances_to_training.max():.2f} degrees")
    
    # Test if performance degrades with distance
    if len(distances_to_training) > 10:
        distance_correlation = np.corrcoef(distances_to_training, 
                                         np.abs(y[test_mask] - y_pred))[0, 1]
        print(f"   Error-distance correlation: {distance_correlation:.3f}")
        
        if distance_correlation > 0.3:
            print("   ⚠️ Performance degrades significantly with distance!")
            print("   → Consider regional models or location-specific features")
        else:
            print("   ✅ Good generalization across geographic distances")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mean_distance': distances_to_training.mean(),
        'max_distance': distances_to_training.max()
    }


def main():
    """
    Main execution function
    """
    print("="*60)
    print("DISEASE SPREAD MODELING - SPATIAL CV")
    print("="*60)
    
    # Create disease spread dataset
    # Reduced dataset size for faster execution
    df = create_disease_spread_data(n_locations=50, n_time_points=20)
    
    # Analyze spatial patterns
    correlations = analyze_spatial_autocorrelation(df)
    
    # Prepare features for modeling
    feature_cols = [
        'population', 'population_density', 'median_age', 'elderly_proportion',
        'income_index', 'education_level', 'hospitals_per_capita', 'icu_beds_per_capita',
        'temperature', 'humidity', 'air_quality_index', 'airport_proximity',
        'highway_density', 'seasonal_factor', 'weekend_factor', 'previous_cases'
    ]
    
    X = df[feature_cols].fillna(0).values
    y = df['daily_cases'].values
    coordinates = df[['longitude', 'latitude']].values
    timestamps = df['time_point'].values
    location_ids = df['location_id'].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n🔧 Model Features:")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target variable: Daily new cases")
    print(f"   Geographic range: {df['latitude'].min():.1f}° to {df['latitude'].max():.1f}° lat")
    print(f"   Time range: {df['date'].min()} to {df['date'].max()}")
    
    # Compare spatial CV strategies
    results = evaluate_spatial_cv_strategies(X, y, coordinates, timestamps, location_ids)
    
    # Visualize results
    visualize_disease_modeling_results(results, df)
    
    # Test geographic generalization
    generalization_results = test_model_generalization(X, y, coordinates, location_ids)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR DISEASE SPREAD MODELING")
    print("="*60)
    print("1. Standard CV overestimates performance due to spatial autocorrelation")
    print("2. Spatial Block CV provides realistic estimates for new regions")
    print("3. Buffered CV adds safety margins against spatial leakage")
    print("4. Spatiotemporal CV handles both geographic and temporal patterns")
    print("5. Geographic distance affects model generalization ability")
    print("\n⚠️  Never use random splits for spatially correlated epidemic data!")
    print("✅ Always validate disease models with geographic holdout regions!")
    print("\n💡 Applications:")
    print("   • COVID-19 spread prediction")
    print("   • Influenza outbreak modeling") 
    print("   • Vector-borne disease risk assessment")
    print("   • Environmental health impact studies")


if __name__ == "__main__":
    main()