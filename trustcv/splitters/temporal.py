"""
Temporal cross-validation splitters for medical time-series data

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
Handles ICU monitoring, disease progression, clinical trials
"""

import warnings
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


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
                    f"Not enough data for split {i+1}. " f"Reduce n_splits or gap size."
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

    def __init__(self, n_splits=5, block_size="day"):
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
        n_samples = len(X)
        use_datetime_blocks = timestamps is not None

        if timestamps is None:
            # Fallback to sequential index when timestamps are not provided
            timestamps = np.arange(n_samples)

        # Create blocks based on block_size
        if isinstance(self.block_size, int):
            # Numeric block size (e.g., every 7 samples)
            blocks = np.arange(n_samples) // self.block_size
        elif use_datetime_blocks and isinstance(self.block_size, str):
            # Convert to datetime only when we have real timestamps
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.to_datetime(timestamps)

            if self.block_size == "day":
                blocks = timestamps.date
            elif self.block_size == "week":
                blocks = timestamps.isocalendar().week
            elif self.block_size == "month":
                blocks = timestamps.month
            else:
                # Unknown string block_size with timestamps - use equal blocks
                blocks = np.arange(n_samples) // max(1, n_samples // max(1, self.n_splits))
        else:
            # No real timestamps or unknown block_size - create equal-sized blocks
            # Need n_splits + 1 blocks for forward-chaining time series CV
            n_blocks_needed = self.n_splits + 1
            blocks = np.arange(n_samples) // max(1, n_samples // n_blocks_needed)

        # Get unique blocks (sorted to maintain temporal order)
        unique_blocks = np.unique(blocks)
        n_blocks = len(unique_blocks)

        if n_blocks < self.n_splits + 1:
            raise ValueError(
                f"Number of blocks ({n_blocks}) must be >= n_splits + 1 ({self.n_splits + 1})"
            )

        # Calculate test size in blocks (similar to sklearn TimeSeriesSplit)
        test_size = n_blocks // (self.n_splits + 1)

        # Generate splits (forward-chaining: train on past, test on future)
        for i in range(self.n_splits):
            # Training blocks: from start to current position
            train_end = (i + 1) * test_size
            # Test blocks: next test_size blocks
            test_start = train_end
            test_end = min(test_start + test_size, n_blocks)

            train_blocks = unique_blocks[:train_end]
            test_blocks = unique_blocks[test_start:test_end]

            train_mask = np.isin(blocks, train_blocks)
            test_mask = np.isin(blocks, test_blocks)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
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

    def __init__(self, n_splits=5, purge_gap=0, embargo_size=0.0, group_exclusive=False, **kwargs):
        # Backward-compatibility alias
        if "exclude_test_groups" in kwargs and not group_exclusive:
            import warnings

            warnings.warn(
                "'exclude_test_groups' is deprecated; use 'group_exclusive' instead.",
                DeprecationWarning,
            )
            group_exclusive = kwargs.pop("exclude_test_groups")
        if kwargs:
            pass

        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.purge_gap = purge_gap
        self.embargo_size = embargo_size
        self.group_exclusive = bool(group_exclusive)

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
        # Groups are optional; if timestamps missing, use sequential order
        if timestamps is None:
            timestamps = np.arange(len(X))

        # Convert to appropriate types
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps)

        # Sort by time (keep helper arrays for time-based thresholds)
        time_order = np.argsort(timestamps)
        sorted_times = timestamps[time_order]
        if groups is not None:
            sorted_groups = np.array(groups)[time_order]

        # Calculate split points
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        # Apply embargo as a fraction of total dataset length to align with external checks
        embargo_samples = int(n_samples * self.embargo_size)

        for i in range(self.n_splits):
            # Define test period (contiguous block in time order)
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)

            # Position-based masks in SORTED space
            test_mask_pos = np.zeros(n_samples, dtype=bool)
            test_mask_pos[test_start:test_end] = True

            # Purge-left in position space: keep only positions before (test_start - purge_gap)
            purge_left_cutoff = max(0, test_start - int(self.purge_gap))
            train_pos_mask = np.zeros(n_samples, dtype=bool)
            train_pos_mask[:purge_left_cutoff] = True

            # Embargo-right in position space: drop positions immediately after test
            if embargo_samples > 0 and i < self.n_splits - 1:
                embargo_end_pos = min(test_end + embargo_samples, n_samples)
                emb_mask_pos = np.zeros(n_samples, dtype=bool)
                emb_mask_pos[test_end:embargo_end_pos] = True
                train_pos_mask = train_pos_mask & (~emb_mask_pos)

            # Map position masks to ORIGINAL indices
            test_idx = time_order[test_mask_pos]
            train_idx = time_order[train_pos_mask]

            # Optional exclusive grouping: remove train samples whose group appears in test
            if self.group_exclusive and groups is not None:
                test_groups = set(np.array(groups)[test_idx])
                keep = ~np.isin(groups[train_idx], list(test_groups))
                train_idx = train_idx[keep]

            if len(train_idx) == 0 or len(test_idx) == 0:
                warnings.warn(
                    f"Empty train or test set in split {i+1}. " "Consider adjusting parameters."
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

        for start in range(
            0, n_samples - self.window_size - self.gap - self.forecast_horizon + 1, self.step_size
        ):
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
        n_splits = (
            n_samples - self.window_size - self.gap - self.forecast_horizon
        ) // self.step_size + 1
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

    def __init__(self, initial_train_size=None, step_size=1, forecast_horizon=1, gap=0, **kwargs):
        # Backward-compatibility: alias old names to 'initial_train_size'
        if "min_train_size" in kwargs and initial_train_size is None:
            import warnings

            warnings.warn(
                "'min_train_size' is deprecated; use 'initial_train_size' instead.",
                DeprecationWarning,
            )
            initial_train_size = kwargs.pop("min_train_size")
        if "initial_size" in kwargs and initial_train_size is None:
            initial_train_size = kwargs.pop("initial_size")
        if kwargs:
            # Ignore unknown legacy kwargs silently to be resilient
            pass

        # Default to 10 if not specified
        if initial_train_size is None:
            initial_train_size = 10

        self.min_train_size = initial_train_size
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

        for train_end in range(
            self.min_train_size, n_samples - self.gap - self.forecast_horizon + 1, self.step_size
        ):
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
        n_splits = (
            n_samples - self.min_train_size - self.gap - self.forecast_horizon
        ) // self.step_size + 1
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
    embargo_size : float, default=0.0
        Percentage of test set size to embargo after test set (renamed from embargo_pct)
    """

    def __init__(self, n_splits=5, purge_gap=0, embargo_size=0.0, **kwargs):
        # Backward-compatibility aliases
        if "embargo_pct" in kwargs and embargo_size == 0.0:
            import warnings

            warnings.warn(
                "'embargo_pct' is deprecated; use 'embargo_size' instead.",
                DeprecationWarning,
            )
            embargo_size = kwargs.pop("embargo_pct")
        if "purge_size" in kwargs and purge_gap == 0:
            purge_gap = kwargs.pop("purge_size")
        if kwargs:
            pass

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = float(embargo_size)

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
        # Apply embargo as a fraction of the entire dataset length to align with
        # external validators that define embargo in absolute sample space
        embargo_size = int(n_samples * self.embargo_pct)

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


class BlockedTimeSeriesCV(BlockedTimeSeries):
    """Backward-compatible alias maintained for older imports."""

    pass


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)

    Advanced method for financial time series that generates
    multiple train/test combinations with purging.

    Parameters
    ----------
    n_splits : int, default=5
        Number of groups to split data into
    n_test_splits : int, default=2
        Number of groups to use as test set
    purge_gap : int, default=0
        Purge gap between train and test
    embargo_size : float, default=0.0
        Embargo percentage
    """

    def __init__(
        self,
        n_splits=5,
        n_test_splits=None,
        purge_gap=0,
        embargo_size=0.0,
        strict_order=True,
        **kwargs,
    ):
        # Backward-compatibility aliases
        import warnings

        if "n_test_groups" in kwargs and n_test_splits is None:
            warnings.warn(
                "'n_test_groups' is deprecated; use 'n_test_splits' instead.",
                DeprecationWarning,
            )
            n_test_splits = kwargs.pop("n_test_groups")
        if "embargo_pct" in kwargs and embargo_size == 0.0:
            warnings.warn(
                "'embargo_pct' is deprecated; use 'embargo_size' instead.",
                DeprecationWarning,
            )
            embargo_size = kwargs.pop("embargo_pct")
        # Alias purge_size -> purge_gap
        if "purge_size" in kwargs and purge_gap == 0:
            purge_gap = kwargs.pop("purge_size")
        if kwargs:
            pass

        if n_test_splits is None:
            n_test_splits = 2
        self.n_splits = n_splits
        self.n_test_groups = int(n_test_splits)
        self.purge_gap = purge_gap
        self.embargo_pct = float(embargo_size)
        self.strict_order = bool(strict_order)

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

        # Generate combinations and yield in chronological order of earliest test start
        combos = list(combinations(range(self.n_splits), self.n_test_groups))

        # sort by earliest index among the chosen test groups
        def combo_key(c):
            return min(groups_idx[g][0] for g in c)

        combos.sort(key=combo_key)
        # If strict order required, keep only one combo per earliest start
        if self.strict_order:
            used_earliest = set()
            combos_unique = []
            for c in combos:
                earliest = combo_key(c)
                if earliest in used_earliest:
                    continue
                used_earliest.add(earliest)
                combos_unique.append(c)
            combos = combos_unique
        for test_groups in combos:
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
            # Apply embargo after each test group as a fraction of total length
            emb = int(n_samples * self.embargo_pct)
            if emb > 0:
                for g in test_groups:
                    g_end = groups_idx[g][-1] + 1
                    emb_end = min(n_samples, g_end + emb)
                    train_mask[g_end:emb_end] = False

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

    def split(self, X, y=None, groups=None):
        """
        Generate outer CV splits (delegates to outer_cv)

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like, optional
            Target values
        groups : array-like, optional
            Group labels

        Yields
        ------
        train, test indices from outer CV
        """
        yield from self.outer_cv.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of outer splitting iterations"""
        return self.outer_cv.get_n_splits(X, y, groups)

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
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV

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
                estimator,
                param_grid,
                cv=inner_cv_generator(),
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            # Evaluate on outer test set
            y_pred = grid_search.predict(X_test)
            score = mean_squared_error(y_test, y_pred)

            outer_scores.append(score)
            best_params.append(grid_search.best_params_)

        return outer_scores, best_params
