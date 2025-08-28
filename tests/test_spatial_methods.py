"""
Unit tests for spatial cross-validation methods
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustcv.splitters.spatial import (
    SpatialBlockCV, BufferedSpatialCV,
    SpatiotemporalBlockCV, EnvironmentalHealthCV
)


class TestSpatialMethods:
    """Test suite for spatial cross-validation methods"""
    
    @classmethod
    def setup_class(cls):
        """Set up spatial test data"""
        np.random.seed(42)
        
        # Create spatial data (disease surveillance)
        n_locations = 200
        
        # Generate spatial coordinates (latitude, longitude)
        cls.lat = np.random.uniform(30, 45, n_locations)
        cls.lon = np.random.uniform(-120, -70, n_locations)
        cls.coordinates = np.column_stack([cls.lat, cls.lon])
        
        # Create spatial features with autocorrelation
        n_features = 8
        cls.X = np.zeros((n_locations, n_features))
        
        # Define disease hotspots
        n_hotspots = 3
        hotspot_indices = np.random.choice(n_locations, n_hotspots, replace=False)
        cls.hotspot_coords = cls.coordinates[hotspot_indices]
        
        # Generate spatially correlated features
        for i in range(n_locations):
            # Base environmental features
            cls.X[i, :4] = np.random.randn(4)
            
            # Distance to nearest hotspot affects other features
            distances_to_hotspots = [
                np.sqrt((cls.lat[i] - h[0])**2 + (cls.lon[i] - h[1])**2)
                for h in cls.hotspot_coords
            ]
            min_dist = min(distances_to_hotspots)
            
            # Spatial effect based on distance
            spatial_effect = np.exp(-min_dist / 5)
            cls.X[i, 4:] = np.random.randn(4) * (1 + spatial_effect)
        
        # Generate spatially correlated outcomes
        cls.y = np.zeros(n_locations, dtype=int)
        for i in range(n_locations):
            # Higher probability near hotspots
            min_dist = min([
                np.sqrt((cls.lat[i] - h[0])**2 + (cls.lon[i] - h[1])**2)
                for h in cls.hotspot_coords
            ])
            
            prob = 0.8 * np.exp(-min_dist / 3) + 0.1  # Base rate + spatial effect
            cls.y[i] = np.random.binomial(1, prob)
        
        # Create temporal dimension for spatiotemporal tests
        cls.timestamps = pd.date_range('2023-01-01', periods=n_locations, freq='D')
        
        # Small dataset for detailed tests
        cls.X_small = cls.X[:50]
        cls.y_small = cls.y[:50]
        cls.coords_small = cls.coordinates[:50]
    
    def test_spatial_block_cv(self):
        """Test basic spatial block cross-validation"""
        cv = SpatialBlockCV(n_splits=4)
        splits = list(cv.split(self.X, self.y, coordinates=self.coordinates))
        
        assert len(splits) == 4, "Should produce 4 spatial blocks"
        
        # Check that blocks partition the space
        all_test_indices = []
        for train_idx, test_idx in splits:
            all_test_indices.extend(test_idx)
            
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0, \
                "No overlap between train and test"
            
            # Test block should be spatially contiguous (roughly)
            test_coords = self.coordinates[test_idx]
            if len(test_coords) > 1:
                # Calculate spatial spread
                lat_range = test_coords[:, 0].max() - test_coords[:, 0].min()
                lon_range = test_coords[:, 1].max() - test_coords[:, 1].min()
                
                # Test block should be somewhat compact
                assert lat_range < 20 or lon_range < 60, \
                    "Test block should be spatially compact"
        
        # Each location should be tested exactly once
        assert len(set(all_test_indices)) == len(self.X), \
            "Each location should be tested exactly once"
    
    def test_buffered_spatial_cv(self):
        """Test buffered spatial cross-validation"""
        cv = BufferedSpatialCV(n_splits=3, buffer_size=2.0)
        splits = list(cv.split(self.X, self.y, coordinates=self.coordinates))
        
        assert len(splits) <= 3, "Should produce at most 3 splits"
        
        for train_idx, test_idx in splits:
            # Check buffer zone
            test_coords = self.coordinates[test_idx]
            train_coords = self.coordinates[train_idx]
            
            # Calculate minimum distance between test and train
            for test_coord in test_coords:
                distances_to_train = [
                    np.sqrt((test_coord[0] - train[0])**2 + 
                           (test_coord[1] - train[1])**2)
                    for train in train_coords
                ]
                
                if len(distances_to_train) > 0:
                    min_distance = min(distances_to_train)
                    # Buffer should be maintained (allowing some tolerance)
                    assert min_distance >= 1.5, \
                        f"Buffer zone violated: min distance {min_distance}"
    
    def test_spatiotemporal_block_cv(self):
        """Test spatiotemporal block cross-validation"""
        cv = SpatiotemporalBlockCV(n_splits=3, temporal_splits=2)
        
        # Need both spatial and temporal information
        splits = list(cv.split(
            self.X, 
            self.y,
            coordinates=self.coordinates,
            timestamps=self.timestamps
        ))
        
        # Should produce spatial_splits * temporal_splits combinations
        assert len(splits) <= 6, "Should produce at most 6 spatiotemporal blocks"
        
        for train_idx, test_idx in splits:
            # Check no overlap
            assert len(set(train_idx) & set(test_idx)) == 0, \
                "No overlap between train and test"
            
            # Test should be a spatiotemporal block
            test_coords = self.coordinates[test_idx]
            test_times = self.timestamps[test_idx]
            
            if len(test_idx) > 1:
                # Check spatial compactness
                lat_range = test_coords[:, 0].max() - test_coords[:, 0].min()
                lon_range = test_coords[:, 1].max() - test_coords[:, 1].min()
                
                # Check temporal compactness
                time_range = (test_times.max() - test_times.min()).days
                
                # Should be compact in at least one dimension
                assert lat_range < 20 or lon_range < 60 or time_range < 100, \
                    "Test block should be compact in space or time"
    
    def test_environmental_health_cv(self):
        """Test environmental health cross-validation"""
        # Add environmental covariates
        env_covariates = np.random.randn(len(self.X), 3)  # Temperature, pollution, humidity
        
        cv = EnvironmentalHealthCV(n_splits=4, buffer_size=1.0)
        splits = list(cv.split(
            self.X,
            self.y,
            coordinates=self.coordinates,
            environmental_covariates=env_covariates
        ))
        
        assert len(splits) > 0, "Should produce at least one split"
        
        for train_idx, test_idx in splits:
            # Check no overlap
            assert len(set(train_idx) & set(test_idx)) == 0, \
                "No overlap between train and test"
            
            # Check that environmental conditions are considered
            train_env = env_covariates[train_idx]
            test_env = env_covariates[test_idx]
            
            # Test set should have somewhat different environmental conditions
            train_mean = np.mean(train_env, axis=0)
            test_mean = np.mean(test_env, axis=0)
            
            # At least one environmental variable should differ
            env_differences = np.abs(train_mean - test_mean)
            assert np.max(env_differences) > 0.1, \
                "Test set should have different environmental conditions"
    
    def test_spatial_autocorrelation_detection(self):
        """Test that spatial methods handle autocorrelation"""
        # Create highly autocorrelated spatial data
        n = 100
        coords_auto = np.random.uniform(0, 10, (n, 2))
        
        # Create spatially smooth field
        X_auto = np.zeros((n, 1))
        y_auto = np.zeros(n)
        
        for i in range(n):
            # Value depends on location (spatial trend)
            X_auto[i] = np.sin(coords_auto[i, 0] / 2) + np.cos(coords_auto[i, 1] / 2)
            y_auto[i] = int(X_auto[i] > 0)
        
        # Test with spatial block CV
        cv = SpatialBlockCV(n_splits=4)
        splits = list(cv.split(X_auto, y_auto, coordinates=coords_auto))
        
        # Each block should have different characteristics
        block_means = []
        for _, test_idx in splits:
            block_mean = np.mean(X_auto[test_idx])
            block_means.append(block_mean)
        
        # Blocks should capture different parts of the spatial trend
        assert np.std(block_means) > 0.1, \
            "Spatial blocks should capture different regions"
    
    def test_grid_based_splitting(self):
        """Test grid-based spatial splitting"""
        # Create regular grid data
        grid_size = 10
        x_grid = np.repeat(np.arange(grid_size), grid_size)
        y_grid = np.tile(np.arange(grid_size), grid_size)
        coords_grid = np.column_stack([x_grid, y_grid])
        
        X_grid = np.random.randn(len(coords_grid), 5)
        y_grid_target = np.random.randint(0, 2, len(coords_grid))
        
        cv = SpatialBlockCV(n_splits=4)
        splits = list(cv.split(X_grid, y_grid_target, coordinates=coords_grid))
        
        assert len(splits) == 4, "Should produce 4 grid blocks"
        
        # Check that blocks are roughly equal size
        block_sizes = [len(test_idx) for _, test_idx in splits]
        assert max(block_sizes) - min(block_sizes) < 30, \
            "Grid blocks should be roughly equal size"
    
    def test_spatial_cv_with_clusters(self):
        """Test spatial CV with clustered data"""
        # Create clustered spatial data
        n_clusters = 5
        n_points_per_cluster = 20
        
        coords_clustered = []
        X_clustered = []
        y_clustered = []
        
        for cluster_id in range(n_clusters):
            # Cluster center
            center = np.random.uniform(0, 100, 2)
            
            for _ in range(n_points_per_cluster):
                # Points around center
                point = center + np.random.randn(2) * 2
                coords_clustered.append(point)
                
                # Features depend on cluster
                features = np.random.randn(5) + cluster_id
                X_clustered.append(features)
                
                # Outcome depends on cluster
                y_clustered.append(cluster_id % 2)
        
        coords_clustered = np.array(coords_clustered)
        X_clustered = np.array(X_clustered)
        y_clustered = np.array(y_clustered)
        
        cv = BufferedSpatialCV(n_splits=3, buffer_size=5.0)
        splits = list(cv.split(X_clustered, y_clustered, coordinates=coords_clustered))
        
        assert len(splits) > 0, "Should handle clustered data"
        
        for train_idx, test_idx in splits:
            # Ideally, entire clusters should be in same split
            # Check spatial separation
            if len(test_idx) > 0 and len(train_idx) > 0:
                test_coords = coords_clustered[test_idx]
                train_coords = coords_clustered[train_idx]
                
                # Calculate centroid distance
                test_centroid = np.mean(test_coords, axis=0)
                train_centroid = np.mean(train_coords, axis=0)
                centroid_distance = np.linalg.norm(test_centroid - train_centroid)
                
                assert centroid_distance > 5, \
                    "Clusters should be spatially separated"
    
    def test_spatial_cv_edge_cases(self):
        """Test spatial CV with edge cases"""
        # Very small dataset
        X_tiny = np.random.randn(10, 3)
        y_tiny = np.random.randint(0, 2, 10)
        coords_tiny = np.random.uniform(0, 10, (10, 2))
        
        cv = SpatialBlockCV(n_splits=2)
        splits = list(cv.split(X_tiny, y_tiny, coordinates=coords_tiny))
        
        assert len(splits) > 0, "Should handle small datasets"
        
        # All same location (no spatial variation)
        coords_same = np.ones((20, 2))
        X_same = np.random.randn(20, 3)
        y_same = np.random.randint(0, 2, 20)
        
        cv2 = SpatialBlockCV(n_splits=4)
        splits2 = list(cv2.split(X_same, y_same, coordinates=coords_same))
        
        # Should fall back to random splitting
        assert len(splits2) == 4, "Should handle data with no spatial variation"
    
    def test_spatial_cv_with_model(self):
        """Test spatial CV with actual model training"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        cv = SpatialBlockCV(n_splits=3)
        
        scores = []
        for train_idx, test_idx in cv.split(
            self.X_small, 
            self.y_small,
            coordinates=self.coords_small
        ):
            model.fit(self.X_small[train_idx], self.y_small[train_idx])
            score = model.score(self.X_small[test_idx], self.y_small[test_idx])
            scores.append(score)
        
        assert len(scores) == 3, "Should produce 3 scores"
        assert all(0 <= s <= 1 for s in scores), "Scores should be valid"
        
        # Spatial CV should give more conservative estimates
        # Compare with random split
        from sklearn.model_selection import KFold
        random_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        random_scores = []
        
        for train_idx, test_idx in random_cv.split(self.X_small):
            model.fit(self.X_small[train_idx], self.y_small[train_idx])
            score = model.score(self.X_small[test_idx], self.y_small[test_idx])
            random_scores.append(score)
        
        # Spatial CV should typically give lower scores due to spatial autocorrelation
        assert np.mean(scores) <= np.mean(random_scores) + 0.1, \
            "Spatial CV should give conservative estimates"
    
    def test_coordinate_validation(self):
        """Test that methods validate coordinate inputs"""
        cv = SpatialBlockCV(n_splits=3)
        
        # Should raise error without coordinates
        with pytest.raises(ValueError):
            list(cv.split(self.X, self.y))
        
        # Should raise error with wrong coordinate shape
        wrong_coords = np.random.randn(len(self.X), 3)  # 3D instead of 2D
        with pytest.raises(ValueError):
            list(cv.split(self.X, self.y, coordinates=wrong_coords))
        
        # Should raise error with wrong number of coordinates
        wrong_n_coords = self.coordinates[:10]
        with pytest.raises(ValueError):
            list(cv.split(self.X, self.y, coordinates=wrong_n_coords))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])