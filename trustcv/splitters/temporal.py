"""
Temporal cross-validation splitters for medical time-series data

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
Handles ICU monitoring, disease progression, clinical trials
"""

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from typing import Optional, Iterator, Tuple, Union
import warnings


class TimeSeriesSplit(_BaseKFold):
    """
    Time-aware cross-validation for clinical data
    
    Ensures training data always precedes test data temporally.
    Critical for predictive models in clinical settings.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits
    gap : int, default=0
        Gap between train and test (e.g., for prediction horizon)
    test_size : int or float, optional
        Size of test set in each split
        
    Examples
    --------
    >>> from trustcv.splitters import TemporalClinical
    >>> tscv = TemporalClinical(n_splits=5, gap=7)  # 7-day gap
    >>> for train, test in tscv.split(X, timestamps=dates):
    ...     # Train on past, test on future
    """
    
    def __init__(self, n_splits=5, gap=0, test_size=None, max_train_size=None):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.gap = gap
        self.test_size = test_size
        self.max_train_size = max_train_size
        
    def split(self, X, y=None, groups=None, timestamps=None):
        """
        Generate indices for temporal splits
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target variable
        groups : array-like, shape (n_samples,), optional
            Group labels (e.g., patient IDs)
        timestamps : array-like, shape (n_samples,)
            Temporal information for each sample
            
        Yields
        ------
        train : ndarray
            Training set indices
        test : ndarray
            Test set indices
        """
        n_samples = len(X)
        
        if timestamps is None:
            # If no timestamps, assume sequential order
            timestamps = np.arange(n_samples)
        else:
            # Convert to pandas datetime if needed
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.to_datetime(timestamps)
        
        # Sort indices by time
        time_idx = np.argsort(timestamps)
        
        # Calculate split indices
        if self.test_size is None:
            # Equal-sized test sets
            n_test = n_samples // (self.n_splits + 1)
        elif isinstance(self.test_size, float):
            n_test = int(n_samples * self.test_size)
        else:
            n_test = self.test_size
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test start position
            test_start = n_samples - (self.n_splits - i) * n_test
            test_end = test_start + n_test
            
            # Calculate train end position (with gap)
            train_end = test_start - self.gap
            
            if train_end <= 0:
                raise ValueError(
                    f"Not enough data for split {i+1}. "
                    f"Reduce n_splits or gap size."
                )
            
            # Apply max_train_size if specified
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            train_idx = time_idx[train_start:train_end]
            test_idx = time_idx[test_start:test_end]
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class BlockedTimeSeries(_BaseKFold):
    """
    Blocked time series cross-validation
    
    Preserves temporal dependencies by keeping time blocks together.
    Useful for seasonal medical data or clustered events.
    
    Parameters
    ----------
    n_splits : int
        Number of splits
    block_size : int or str
        Size of temporal blocks ('day', 'week', 'month', or integer)
        
    Examples
    --------
    >>> btscv = BlockedTimeSeries(n_splits=5, block_size='week')
    >>> for train, test in btscv.split(X, timestamps=dates):
    ...     # Blocks of weeks stay together
    """
    
    def __init__(self, n_splits=5, block_size='day'):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.block_size = block_size
        
    def split(self, X, y=None, groups=None, timestamps=None):
        """
        Generate blocked time series splits
        
        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like, optional
            Group labels
        timestamps : array-like
            Temporal information
            
        Yields
        ------
        train, test indices
        """
        if timestamps is None:
            raise ValueError("Timestamps required for blocked time series CV")
        
        # Convert to datetime
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)
        
        # Create blocks based on block_size
        if self.block_size == 'day':
            blocks = timestamps.date
        elif self.block_size == 'week':
            blocks = timestamps.isocalendar().week
        elif self.block_size == 'month':
            blocks = timestamps.month
        elif isinstance(self.block_size, int):
            # Numeric block size (e.g., every 7 samples)
            blocks = np.arange(len(timestamps)) // self.block_size
        else:
            raise ValueError(f"Unknown block_size: {self.block_size}")
        
        # Get unique blocks
        unique_blocks = np.unique(blocks)
        n_blocks = len(unique_blocks)
        
        if n_blocks < self.n_splits:
            raise ValueError(
                f"Number of blocks ({n_blocks}) < n_splits ({self.n_splits})"
            )
        
        # Calculate blocks per fold
        blocks_per_fold = n_blocks // self.n_splits
        
        # Generate splits
        for i in range(self.n_splits):
            test_blocks = unique_blocks[i * blocks_per_fold:(i + 1) * blocks_per_fold]
            
            train_mask = ~np.isin(blocks, test_blocks)
            test_mask = np.isin(blocks, test_blocks)
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """
    Purged Group Time Series Split for medical panel data
    
    Combines:
    - Temporal ordering (time series)
    - Group preservation (patient data)
    - Purging (gap to prevent leakage)
    - Embargo (no trading period simulation)
    
    Essential for financial-medical hybrid applications
    (e.g., healthcare cost prediction, insurance claims)
    
    Parameters
    ----------
    n_splits : int
        Number of splits
    purge_gap : int
        Temporal gap between train and test
    embargo_size : float
        Fraction of data to embargo after test
        
    Examples
    --------
    >>> pgts = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=30)
    >>> for train, test in pgts.split(X, groups=patients, timestamps=dates):
    ...     # Patient-aware temporal splitting with purging
    """
    
    def __init__(self, n_splits=5, purge_gap=0, embargo_size=0.0):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.purge_gap = purge_gap
        self.embargo_size = embargo_size
        
    def split(self, X, y=None, groups=None, timestamps=None):
        """
        Generate purged group time series splits
        
        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels  
        groups : array-like
            Patient/group identifiers
        timestamps : array-like
            Temporal information
            
        Yields
        ------
        train, test indices
        """
        if groups is None:
            raise ValueError("Groups required for purged group time series")
        if timestamps is None:
            raise ValueError("Timestamps required for purged group time series")
        
        # Convert to appropriate types
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)
        
        # Sort by time
        time_order = np.argsort(timestamps)
        sorted_times = timestamps[time_order]
        sorted_groups = np.array(groups)[time_order]
        
        # Calculate split points
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        embargo_samples = int(test_size * self.embargo_size)
        
        for i in range(self.n_splits):
            # Define test period
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)
            
            # Get test period times
            test_start_time = sorted_times[test_start]
            test_end_time = sorted_times[test_end - 1]
            
            # Apply purge gap
            if self.purge_gap > 0:
                train_cutoff_time = test_start_time - pd.Timedelta(days=self.purge_gap)
            else:
                train_cutoff_time = test_start_time
            
            # Get groups in test set
            test_mask = (timestamps >= test_start_time) & (timestamps <= test_end_time)
            test_groups = set(groups[test_mask])
            
            # Create train mask
            # Exclude: future data, test groups, purge period
            train_mask = (
                (timestamps < train_cutoff_time) &  # Before test (with purge)
                ~np.isin(groups, list(test_groups))  # Not in test groups
            )
            
            # Apply embargo if specified
            if embargo_samples > 0 and i < self.n_splits - 1:
                embargo_end = test_end + embargo_samples
                embargo_mask = (time_order >= test_end) & (time_order < embargo_end)
                train_mask = train_mask & ~embargo_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                warnings.warn(
                    f"Empty train or test set in split {i+1}. "
                    "Consider adjusting parameters."
                )
                continue
            
            yield train_idx, test_idx


class RollingWindowCV:
    """
    Rolling Window Cross-Validation (Walk-Forward Validation)
    
    Fixed-size training window that slides through time.
    Maintains constant training set size, important for stability.
    
    Parameters
    ----------
    window_size : int
        Size of training window
    step_size : int, default=1
        Step size for sliding window
    forecast_horizon : int, default=1
        Number of periods ahead to predict
    gap : int, default=0
        Gap between training and test sets
    """
    
    def __init__(self, window_size, step_size=1, forecast_horizon=1, gap=0):
        self.window_size = window_size
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
        self.gap = gap
        
    def split(self, X, y=None, groups=None):
        """
        Generate rolling window splits
        
        Yields
        ------
        train, test indices
        """
        n_samples = len(X)
        
        for start in range(0, n_samples - self.window_size - self.gap - self.forecast_horizon + 1, 
                          self.step_size):
            train_end = start + self.window_size
            test_start = train_end + self.gap
            test_end = test_start + self.forecast_horizon
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits"""
        if X is None:
            return None
        n_samples = len(X)
        n_splits = (n_samples - self.window_size - self.gap - self.forecast_horizon) // self.step_size + 1
        return max(0, n_splits)


class ExpandingWindowCV:
    """
    Expanding Window Cross-Validation
    
    Training set grows over time, always starting from the beginning.
    Useful when all historical data is relevant.
    
    Parameters
    ----------
    min_train_size : int
        Minimum training set size
    step_size : int, default=1
        Step size for expanding window
    forecast_horizon : int, default=1
        Number of periods ahead to predict
    gap : int, default=0
        Gap between training and test sets
    """
    
    def __init__(self, min_train_size, step_size=1, forecast_horizon=1, gap=0):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
        self.gap = gap
        
    def split(self, X, y=None, groups=None):
        """
        Generate expanding window splits
        
        Yields
        ------
        train, test indices
        """
        n_samples = len(X)
        
        for train_end in range(self.min_train_size, 
                               n_samples - self.gap - self.forecast_horizon + 1,
                               self.step_size):
            test_start = train_end + self.gap
            test_end = test_start + self.forecast_horizon
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits"""
        if X is None:
            return None
        n_samples = len(X)
        n_splits = (n_samples - self.min_train_size - self.gap - self.forecast_horizon) // self.step_size + 1
        return max(0, n_splits)


class PurgedKFoldCV:
    """
    Purged K-Fold with Embargo for Financial Time Series
    
    Implements purging and embargo to prevent data leakage in
    financial/medical cost prediction scenarios.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    purge_gap : int, default=0
        Number of samples to purge between train and test
    embargo_pct : float, default=0.0
        Percentage of test set size to embargo after test set
    """
    
    def __init__(self, n_splits=5, purge_gap=0, embargo_pct=0.0):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
    def split(self, X, y=None, groups=None, timestamps=None):
        """
        Generate purged k-fold splits
        
        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like, optional
            Group labels
        timestamps : array-like, optional
            Timestamps for ordering
            
        Yields
        ------
        train, test indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # If timestamps provided, sort by time
        if timestamps is not None:
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.to_datetime(timestamps)
            indices = indices[np.argsort(timestamps)]
        
        # Calculate fold sizes
        fold_size = n_samples // self.n_splits
        embargo_size = int(fold_size * self.embargo_pct)
        
        for i in range(self.n_splits):
            # Define test fold
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n_samples
            
            # Create test indices
            test_idx = indices[test_start:test_end]
            
            # Create train indices with purging and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Remove test indices
            train_mask[test_start:test_end] = False
            
            # Apply purge before test
            if self.purge_gap > 0 and test_start > 0:
                purge_start = max(0, test_start - self.purge_gap)
                train_mask[purge_start:test_start] = False
            
            # Apply purge after test
            if self.purge_gap > 0 and test_end < n_samples:
                purge_end = min(n_samples, test_end + self.purge_gap)
                train_mask[test_end:purge_end] = False
            
            # Apply embargo after test
            if embargo_size > 0 and test_end < n_samples:
                embargo_end = min(n_samples, test_end + embargo_size)
                train_mask[test_end:embargo_end] = False
            
            train_idx = indices[train_mask]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)
    
    Advanced method for financial time series that generates
    multiple train/test combinations with purging.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of groups to split data into
    n_test_groups : int, default=2
        Number of groups to use as test set
    purge_gap : int, default=0
        Purge gap between train and test
    embargo_pct : float, default=0.0
        Embargo percentage
    """
    
    def __init__(self, n_splits=5, n_test_groups=2, purge_gap=0, embargo_pct=0.0):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
    def split(self, X, y=None, groups=None):
        """
        Generate combinatorial purged splits
        
        Yields
        ------
        train, test indices
        """
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Split data into groups
        group_size = n_samples // self.n_splits
        groups_idx = []
        
        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups_idx.append(indices[start:end])
        
        # Generate combinations
        for test_groups in combinations(range(self.n_splits), self.n_test_groups):
            # Combine test groups
            test_idx = np.concatenate([groups_idx[g] for g in test_groups])
            
            # Create train indices with purging
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            
            # Apply purging around each test group
            if self.purge_gap > 0:
                for g in test_groups:
                    group_start = groups_idx[g][0]
                    group_end = groups_idx[g][-1] + 1
                    
                    # Purge before
                    purge_start = max(0, group_start - self.purge_gap)
                    train_mask[purge_start:group_start] = False
                    
                    # Purge after
                    purge_end = min(n_samples, group_end + self.purge_gap)
                    train_mask[group_end:purge_end] = False
            
            train_idx = indices[train_mask]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)


class NestedTemporalCV:
    """
    Nested Cross-Validation for Temporal Data
    
    Performs nested CV while preserving temporal order.
    Outer loop for evaluation, inner loop for hyperparameter tuning.
    
    Parameters
    ----------
    outer_cv : temporal CV object
        Outer CV for model evaluation
    inner_cv : temporal CV object
        Inner CV for hyperparameter tuning
    """
    
    def __init__(self, outer_cv=None, inner_cv=None):
        if outer_cv is None:
            outer_cv = ExpandingWindowCV(min_train_size=100)
        if inner_cv is None:
            inner_cv = RollingWindowCV(window_size=50)
            
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        
    def fit_predict(self, estimator, X, y, param_grid, timestamps=None):
        """
        Perform nested temporal cross-validation
        
        Parameters
        ----------
        estimator : estimator object
            The model to evaluate
        X : array-like
            Feature matrix
        y : array-like
            Target values
        param_grid : dict
            Hyperparameter grid for tuning
        timestamps : array-like
            Temporal information
            
        Returns
        -------
        scores : list
            Outer CV scores
        best_params : list
            Best parameters for each outer fold
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_squared_error
        
        outer_scores = []
        best_params = []
        
        for train_idx, test_idx in self.outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Custom CV iterator for inner loop that respects time
            def inner_cv_generator():
                for inner_train, inner_test in self.inner_cv.split(X_train, y_train):
                    yield inner_train, inner_test
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator, param_grid, 
                cv=inner_cv_generator(),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Evaluate on outer test set
            y_pred = grid_search.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            
            outer_scores.append(score)
            best_params.append(grid_search.best_params_)
            
        return outer_scores, best_params