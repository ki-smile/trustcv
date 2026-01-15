"""
Spatial and Spatiotemporal Cross-Validation Methods

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team

For medical data with geographic/spatial components:
- Disease spread modeling
- Environmental health studies
- Healthcare accessibility analysis
- Epidemiological research
"""

import warnings
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state, indexable


class SpatialBlockCV(BaseCrossValidator):
    """
    Spatial Block Cross-Validation

    Divides spatial data into blocks to handle spatial autocorrelation.
    Essential for geographic medical data analysis.

    Parameters
    ----------
    n_splits : int, default=5
        Number of spatial blocks
    block_shape : str or tuple, default='grid'
        Shape of blocks: 'grid', 'random', or custom shape
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_shape: Literal["grid", "kmeans"] = "grid",
        block_size: Optional[float] = None,
        random_state: Optional[int] = None,
        coordinates: Optional[ArrayLike] = None,
        **kwargs,
    ):
        super().__init__()
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.block_shape = block_shape
        self.block_size = block_size
        self.random_state = random_state
        self.coordinates = coordinates  # store default coordinates
        self.kwargs = kwargs

    def _create_spatial_blocks(self, coordinates, n_blocks):
        """
        Create spatial blocks from coordinates

        Parameters
        ----------
        coordinates : array-like of shape (n_samples, 2)
            Spatial coordinates (x, y) or (lat, lon)
        n_blocks : int
            Number of blocks to create

        Returns
        -------
        block_ids : array-like of shape (n_samples,)
            Block assignment for each sample
        """
        n_samples = len(coordinates)

        if self.block_shape == "grid":
            # Create regular grid blocks
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]

            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(n_blocks)))

            # Create bins for x and y
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)

            # Assign samples to grid cells
            x_indices = np.digitize(x_coords, x_bins) - 1
            y_indices = np.digitize(y_coords, y_bins) - 1

            # Ensure indices are within bounds
            x_indices = np.clip(x_indices, 0, grid_size - 1)
            y_indices = np.clip(y_indices, 0, grid_size - 1)

            # Convert 2D grid indices to 1D block IDs
            block_ids = y_indices * grid_size + x_indices

            # Map to n_blocks if necessary
            if grid_size * grid_size > n_blocks:
                # Merge some blocks
                block_ids = block_ids % n_blocks

        elif self.block_shape == "random":
            # Random spatial clustering using k-means
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_blocks, random_state=self.random_state)
            block_ids = kmeans.fit_predict(coordinates)

        elif self.block_shape == "hexagonal":
            # Hexagonal binning for better spatial coverage
            raise NotImplementedError("Hexagonal blocks coming soon")

        else:
            raise ValueError(f"Unknown block_shape: {self.block_shape}")

        return block_ids

    def _grid_blocks(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Create grid-based spatial blocks.

        Parameters
        ----------
        coordinates : np.ndarray of shape (n_samples, 2)
            Spatial coordinates (x, y) or (lon, lat)

        Returns
        -------
        block_ids : np.ndarray of shape (n_samples,)
            Block assignment for each sample
        """
        n_samples = len(coordinates)
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]

        # Calculate grid dimensions to get approximately n_splits blocks
        grid_size = int(np.ceil(np.sqrt(self.n_splits)))

        # Create bins for x and y coordinates
        x_bins = np.linspace(x_coords.min(), x_coords.max() + 1e-10, grid_size + 1)
        y_bins = np.linspace(y_coords.min(), y_coords.max() + 1e-10, grid_size + 1)

        # Assign samples to grid cells
        x_indices = np.digitize(x_coords, x_bins) - 1
        y_indices = np.digitize(y_coords, y_bins) - 1

        # Ensure indices are within bounds
        x_indices = np.clip(x_indices, 0, grid_size - 1)
        y_indices = np.clip(y_indices, 0, grid_size - 1)

        # Convert 2D grid indices to 1D block IDs
        block_ids = y_indices * grid_size + x_indices

        return block_ids

    def _kmeans_blocks(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Create k-means clustering-based spatial blocks.

        Parameters
        ----------
        coordinates : np.ndarray of shape (n_samples, 2)
            Spatial coordinates (x, y) or (lon, lat)

        Returns
        -------
        block_ids : np.ndarray of shape (n_samples,)
            Block assignment for each sample based on k-means clustering
        """
        from sklearn.cluster import KMeans

        # Use k-means to create spatially coherent blocks
        kmeans = KMeans(n_clusters=self.n_splits, random_state=self.random_state, n_init=10)
        block_ids = kmeans.fit_predict(coordinates)

        return block_ids

    def split(self, X, y=None, groups=None, coordinates=None):
        """
        Generate spatial block splits

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like, optional
            Group labels
        coordinates : array-like of shape (n_samples, 2)
            Spatial coordinates for each sample

        Yields
        ------
        train, test indices
        """

        if coordinates is None:
            coordinates = self.coordinates
        if coordinates is None:
            raise ValueError("Spatial coordinates required for spatial block CV")

        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Coordinates must be an array of shape (n_samples, 2)")

        X, y, groups = indexable(X, y, groups)
        if coordinates.shape[0] != len(X):
            raise ValueError("Coordinate array size must match X")

        if self.block_shape == "grid":
            blocks = self._grid_blocks(coordinates)
        elif self.block_shape == "kmeans":
            blocks = self._kmeans_blocks(coordinates)
        else:
            raise ValueError(f"Unknown block_shape '{self.block_shape}'. Use 'grid' or 'kmeans'.")

        unique_blocks = np.unique(blocks)
        if len(unique_blocks) < self.n_splits:
            # Fall back to random assignment when spatial variation is insufficient
            warnings.warn(
                f"Insufficient spatial variation: got {len(unique_blocks)} unique blocks, "
                f"need {self.n_splits}. Falling back to random assignment."
            )
            rng = np.random.default_rng(self.random_state)
            blocks = rng.integers(0, self.n_splits, size=len(X))
            unique_blocks = np.unique(blocks)

        rng = np.random.default_rng(self.random_state)

        # Select exactly n_splits blocks if we have more
        if len(unique_blocks) > self.n_splits:
            selected_blocks = rng.choice(
                unique_blocks,
                size=self.n_splits,
                replace=False,
            )
        else:
            selected_blocks = unique_blocks

        for block in selected_blocks:
            test_mask = blocks == block
            if test_mask.sum() == 0:
                continue
            yield np.where(~test_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class BufferedSpatialCV(BaseCrossValidator):
    """
    Buffered Spatial Cross-Validation

    Creates buffer zones around test blocks to reduce spatial autocorrelation.
    Samples in buffer zones are excluded from training.

    Parameters
    ----------
    n_splits : int, default=5
        Number of spatial blocks
    buffer_size : float
        Size of buffer zone around test blocks
    distance_metric : str, default='euclidean'
        Distance metric for buffer calculation
    block_shape : str, default='grid'
        Shape of spatial blocks
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self,
        n_splits: int = 5,
        buffer_size: float = 0.1,
        distance_metric: str = "euclidean",
        block_shape: str = "grid",
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.buffer_size = buffer_size
        self.distance_metric = distance_metric
        self.block_shape = block_shape
        self.random_state = random_state

    def _calculate_distances(self, coordinates, test_indices):
        """
        Calculate distances from all points to test set

        Parameters
        ----------
        coordinates : array-like of shape (n_samples, 2)
            Spatial coordinates
        test_indices : array-like
            Indices of test samples

        Returns
        -------
        distances : array-like of shape (n_samples,)
            Minimum distance to test set for each sample
        """
        from scipy.spatial.distance import cdist

        test_coords = coordinates[test_indices]

        if self.distance_metric == "euclidean":
            # Calculate Euclidean distances
            distances = cdist(coordinates, test_coords, metric="euclidean")
        elif self.distance_metric == "haversine":
            # For geographic coordinates (lat, lon)
            from sklearn.metrics.pairwise import haversine_distances

            # Convert to radians
            coords_rad = np.radians(coordinates)
            test_coords_rad = np.radians(test_coords)

            # Calculate haversine distances
            distances = haversine_distances(coords_rad, test_coords_rad)
            # Convert to kilometers (Earth radius ≈ 6371 km)
            distances *= 6371
        else:
            distances = cdist(coordinates, test_coords, metric=self.distance_metric)

        # Return minimum distance to any test point
        return distances.min(axis=1)

    def split(self, X, y=None, groups=None, coordinates=None):
        """
        Generate buffered spatial splits

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like, optional
            Group labels
        coordinates : array-like of shape (n_samples, 2)
            Spatial coordinates

        Yields
        ------
        train, test indices
        """
        if coordinates is None:
            raise ValueError("Spatial coordinates required for buffered spatial CV")

        X, y, groups = indexable(X, y, groups)
        coordinates = np.array(coordinates)
        n_samples = len(X)
        indices = np.arange(n_samples)

        # First create spatial blocks
        spatial_cv = SpatialBlockCV(
            n_splits=self.n_splits, block_shape=self.block_shape, random_state=self.random_state
        )

        for train_idx, test_idx in spatial_cv.split(X, y, groups, coordinates):
            # Calculate distances from all points to test set
            distances = self._calculate_distances(coordinates, test_idx)

            # Determine buffer threshold
            if self.buffer_size < 1.0:
                # Buffer size as fraction of maximum distance
                max_dist = distances.max()
                buffer_threshold = max_dist * self.buffer_size
            else:
                # Absolute buffer size
                buffer_threshold = self.buffer_size

            # Exclude buffer zone from training
            buffer_mask = distances <= buffer_threshold
            train_mask = np.zeros(n_samples, dtype=bool)
            train_mask[train_idx] = True
            train_mask[buffer_mask] = False

            final_train_idx = indices[train_mask]

            if len(final_train_idx) > 0:
                yield final_train_idx, test_idx
            else:
                warnings.warn(
                    f"Buffer zone too large, no training samples remaining. "
                    f"Consider reducing buffer_size."
                )

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class SpatiotemporalBlockCV(BaseCrossValidator):
    """
    Spatiotemporal Block Cross-Validation

    Handles data with both spatial and temporal dimensions.
    Essential for disease spread modeling and environmental health studies.

    Parameters
    ----------
    n_spatial_blocks : int, default=3
        Number of spatial blocks
    n_temporal_blocks : int, default=3
        Number of temporal blocks
    buffer_space : float, default=0
        Spatial buffer size
    buffer_time : int, default=0
        Temporal buffer size (in time units)
    block_shape : str, default='grid'
        Shape of spatial blocks
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self,
        n_spatial_blocks: int = 3,
        n_temporal_blocks: int = 3,
        buffer_space: float = 0,
        buffer_time: int = 0,
        block_shape: str = "grid",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        # Backward-compatibility aliases
        if "n_splits" in kwargs and n_spatial_blocks == 3:
            n_spatial_blocks = kwargs.pop("n_splits")
        if "temporal_splits" in kwargs and n_temporal_blocks == 3:
            n_temporal_blocks = kwargs.pop("temporal_splits")
        if kwargs:
            pass

        self.n_spatial_blocks = n_spatial_blocks
        self.n_temporal_blocks = n_temporal_blocks
        self.buffer_space = buffer_space
        self.buffer_time = buffer_time
        self.block_shape = block_shape
        self.random_state = random_state

    def _create_spatiotemporal_blocks(self, coordinates, timestamps):
        """
        Create spatiotemporal blocks

        Parameters
        ----------
        coordinates : array-like of shape (n_samples, 2)
            Spatial coordinates
        timestamps : array-like of shape (n_samples,)
            Temporal information

        Returns
        -------
        block_ids : array-like of shape (n_samples,)
            Spatiotemporal block assignment
        """
        import pandas as pd

        # Create spatial blocks
        spatial_cv = SpatialBlockCV(
            n_splits=self.n_spatial_blocks,
            block_shape=self.block_shape,
            random_state=self.random_state,
        )
        spatial_blocks = spatial_cv._create_spatial_blocks(coordinates, self.n_spatial_blocks)

        # Create temporal blocks
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)

        time_min = timestamps.min()
        time_max = timestamps.max()
        time_range = time_max - time_min
        block_duration = time_range / self.n_temporal_blocks

        temporal_blocks = np.zeros(len(timestamps), dtype=int)
        for i in range(self.n_temporal_blocks):
            block_start = time_min + i * block_duration
            block_end = time_min + (i + 1) * block_duration

            if i == self.n_temporal_blocks - 1:
                # Last block includes end point
                mask = (timestamps >= block_start) & (timestamps <= time_max)
            else:
                mask = (timestamps >= block_start) & (timestamps < block_end)

            temporal_blocks[mask] = i

        # Combine spatial and temporal blocks
        n_total_blocks = self.n_spatial_blocks * self.n_temporal_blocks
        block_ids = spatial_blocks * self.n_temporal_blocks + temporal_blocks

        return block_ids, spatial_blocks, temporal_blocks

    def split(self, X, y=None, groups=None, coordinates=None, timestamps=None):
        """
        Generate spatiotemporal block splits

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like, optional
            Group labels
        coordinates : array-like of shape (n_samples, 2)
            Spatial coordinates
        timestamps : array-like of shape (n_samples,)
            Temporal information

        Yields
        ------
        train, test indices
        """
        if coordinates is None or timestamps is None:
            raise ValueError(
                "Both spatial coordinates and timestamps required for " "spatiotemporal block CV"
            )

        X, y, groups = indexable(X, y, groups)
        coordinates = np.array(coordinates)
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Create spatiotemporal blocks
        block_ids, spatial_blocks, temporal_blocks = self._create_spatiotemporal_blocks(
            coordinates, timestamps
        )

        unique_blocks = np.unique(block_ids)
        n_blocks = len(unique_blocks)

        # Generate splits
        for test_block in unique_blocks:
            test_mask = block_ids == test_block
            test_idx = indices[test_mask]

            if self.buffer_space > 0 or self.buffer_time > 0:
                # Apply spatiotemporal buffer
                train_mask = ~test_mask

                # Get test block's spatial and temporal components
                test_spatial = spatial_blocks[test_mask][0]
                test_temporal = temporal_blocks[test_mask][0]

                # Apply spatial buffer
                if self.buffer_space > 0:
                    from scipy.spatial.distance import cdist

                    test_coords = coordinates[test_mask]
                    distances = cdist(coordinates, test_coords).min(axis=1)
                    spatial_buffer = distances <= self.buffer_space
                    train_mask = train_mask & ~spatial_buffer

                # Apply temporal buffer
                if self.buffer_time > 0:
                    temporal_buffer = np.abs(temporal_blocks - test_temporal) <= self.buffer_time
                    train_mask = train_mask & ~temporal_buffer

                train_idx = indices[train_mask]
            else:
                # No buffer
                train_idx = indices[~test_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_spatial_blocks * self.n_temporal_blocks


class EnvironmentalHealthCV:
    """
    Specialized CV for environmental health studies

    Combines spatial, temporal, and environmental factors.
    Handles pollution studies, climate health impacts, etc.

    Parameters
    ----------
    spatial_blocks : int, default=4
        Number of spatial regions
    temporal_strategy : str, default='seasonal'
        Temporal splitting strategy: 'seasonal', 'yearly', 'custom'
    environmental_vars : list, optional
        Environmental variables to consider
    buffer_config : dict, optional
        Buffer configuration for different factors
    """

    def __init__(
        self,
        spatial_blocks: int = 4,
        temporal_strategy: str = "seasonal",
        environmental_vars: Optional[List[str]] = None,
        buffer_config: Optional[dict] = None,
        **kwargs,
    ):
        # Backward-compatibility aliases
        if "n_splits" in kwargs and spatial_blocks == 4:
            spatial_blocks = kwargs.pop("n_splits")
        if "buffer_size" in kwargs and buffer_config is None:
            buffer_size = kwargs.pop("buffer_size")
            buffer_config = {"default": buffer_size}
        if kwargs:
            pass

        self.spatial_blocks = spatial_blocks
        self.temporal_strategy = temporal_strategy
        self.environmental_vars = environmental_vars or []
        self.buffer_config = buffer_config or {}

    def split(self, X, y=None, coordinates=None, timestamps=None, environmental_data=None, **kwargs):
        """
        Generate environmental health CV splits

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Health outcomes
        coordinates : array-like
            Spatial locations
        timestamps : array-like
            Time points
        environmental_data : dict
            Environmental covariates

        Yields
        ------
        train, test indices
        """
        import pandas as pd

        # Parameter alias
        if "environmental_covariates" in kwargs and environmental_data is None:
            environmental_data = kwargs.pop("environmental_covariates")

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Handle missing timestamps - use sequential order
        if timestamps is None:
            timestamps = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
        elif not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)

        # Define temporal blocks based on strategy
        if self.temporal_strategy == "seasonal":
            # Split by seasons (0=winter, 1=spring, 2=summer, 3=fall)
            seasons = (timestamps.month % 12 // 3).values
            temporal_blocks = np.asarray(seasons, dtype=int)
            n_temporal = 4
        elif self.temporal_strategy == "yearly":
            # Split by years
            years = timestamps.year.values
            unique_years = np.unique(years)
            year_to_block = {year: i for i, year in enumerate(unique_years)}
            temporal_blocks = np.array([year_to_block[y] for y in years], dtype=int)
            n_temporal = len(unique_years)
        else:
            # Custom temporal blocks
            temporal_blocks = np.zeros(n_samples, dtype=int)
            n_temporal = 1

        # Create spatial blocks
        spatial_cv = SpatialBlockCV(n_splits=self.spatial_blocks)
        spatial_blocks = spatial_cv._create_spatial_blocks(coordinates, self.spatial_blocks)

        # Combine into spatiotemporal blocks
        block_ids = spatial_blocks * n_temporal + temporal_blocks
        unique_blocks = np.unique(block_ids)

        # Generate splits with environmental considerations
        for test_block in unique_blocks:
            test_mask = block_ids == test_block
            test_idx = indices[test_mask]
            train_mask = ~test_mask

            # Apply environmental buffers if specified
            if environmental_data is not None and self.environmental_vars:
                for var in self.environmental_vars:
                    if var in environmental_data:
                        var_data = environmental_data[var]
                        test_values = var_data[test_mask]

                        # Create buffer based on environmental similarity
                        if var in self.buffer_config:
                            threshold = self.buffer_config[var]
                            for test_val in test_values:
                                env_buffer = np.abs(var_data - test_val) <= threshold
                                train_mask = train_mask & ~env_buffer

            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self):
        """Returns estimated number of splits"""
        return self.spatial_blocks * 4  # Approximate for seasonal
