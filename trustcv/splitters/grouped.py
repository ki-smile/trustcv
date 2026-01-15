"""
Grouped cross-validation splitters for medical data

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
Handles patient-level and hierarchical medical data structures
"""

import warnings
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state


class GroupKFoldMedical(_BaseKFold):
    """
    Patient-aware K-Fold cross-validator

    Ensures all records from a patient stay in the same fold.
    Critical for preventing data leakage in ML.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle patients before splitting
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> from trustcv.splitters import GroupKFoldMedical
    >>> cv = GroupKFoldMedical(n_splits=5)
    >>> for train, test in cv.split(X, y, patient_ids):
    ...     # All records from same patient in same fold
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target variable
        groups : array-like, shape (n_samples,)
            Patient identifiers

        Yields
        ------
        train : ndarray
            Training set indices for fold
        test : ndarray
            Test set indices for fold
        """
        if groups is None:
            raise ValueError("Patient IDs (groups) must be provided")

        # Convert to numpy array if needed
        if isinstance(groups, pd.Series):
            groups = groups.values

        # Get unique patients
        unique_patients = np.unique(groups)
        n_patients = len(unique_patients)

        if n_patients < self.n_splits:
            raise ValueError(
                f"Cannot have number of splits ({self.n_splits}) > "
                f"number of patients ({n_patients})"
            )

        # Shuffle patients if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            unique_patients = rng.permutation(unique_patients)

        # Create patient to fold mapping
        fold_sizes = np.full(self.n_splits, n_patients // self.n_splits, dtype=int)
        fold_sizes[: n_patients % self.n_splits] += 1

        current = 0
        patient_to_fold = {}
        for fold_idx, fold_size in enumerate(fold_sizes):
            fold_patients = unique_patients[current : current + fold_size]
            for patient in fold_patients:
                patient_to_fold[patient] = fold_idx
            current += fold_size

        # Generate train/test splits
        for test_fold in range(self.n_splits):
            train_idx = []
            test_idx = []

            for idx, patient in enumerate(groups):
                if patient_to_fold[patient] == test_fold:
                    test_idx.append(idx)
                else:
                    train_idx.append(idx)

            yield np.array(train_idx), np.array(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class StratifiedGroupKFold(_BaseKFold):
    """
    Stratified Group K-Fold cross-validator

    Combines stratification (preserving class balance) with grouping
    (keeping patient records together). Essential for imbalanced
    medical datasets with multiple records per patient.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    random_state : int, optional
        Random seed

    Examples
    --------
    >>> cv = StratifiedGroupKFold(n_splits=5)
    >>> # Preserves both class distribution AND patient grouping
    >>> for train, test in cv.split(X, y, patient_ids):
    ...     train_model(X[train], y[train])
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups):
        """
        Generate stratified grouped splits

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target variable (required for stratification)
        groups : array-like, shape (n_samples,)
            Patient identifiers

        Yields
        ------
        train : ndarray
            Training indices
        test : ndarray
            Test indices
        """
        if y is None:
            raise ValueError("Target variable y is required for stratification")
        if groups is None:
            raise ValueError("Patient IDs (groups) are required")

        # Convert to arrays
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(groups, pd.Series):
            groups = groups.values

        # Get patient-level labels (using majority vote)
        patient_labels = {}
        unique_patients = np.unique(groups)

        for patient in unique_patients:
            patient_mask = groups == patient
            patient_y = y[patient_mask]
            # Use majority class for patient
            unique_classes, counts = np.unique(patient_y, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            patient_labels[patient] = majority_class

        # Create patient-level arrays for stratification
        patient_array = np.array(list(patient_labels.keys()))
        label_array = np.array(list(patient_labels.values()))

        # Use StratifiedKFold on patients
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        # Generate splits
        for train_patients_idx, test_patients_idx in skf.split(patient_array, label_array):
            train_patients = patient_array[train_patients_idx]
            test_patients = patient_array[test_patients_idx]

            # Convert patient splits to sample splits
            train_idx = np.where(np.isin(groups, train_patients))[0]
            test_idx = np.where(np.isin(groups, test_patients))[0]

            # Verify stratification (robust to non-integer labels)
            # np.bincount requires integer labels; encode to 0..n_classes-1 first
            try:
                # Use pandas factorize for robust encoding of any dtype
                y_enc, _ = pd.factorize(y)
            except Exception:
                # Fallback to numpy unique-based encoding
                _, y_enc = np.unique(y, return_inverse=True)

            n_classes = int(y_enc.max()) + 1 if len(y_enc) else 0
            if n_classes > 0:
                train_dist = np.bincount(y_enc[train_idx], minlength=n_classes) / max(
                    len(train_idx), 1
                )
                test_dist = np.bincount(y_enc[test_idx], minlength=n_classes) / max(
                    len(test_idx), 1
                )
            else:
                train_dist = np.array([])
                test_dist = np.array([])

            if np.abs(train_dist - test_dist).max() > 0.1:
                warnings.warn(
                    "Class distribution difference > 10% between train and test. "
                    "Consider adjusting number of splits or checking data distribution."
                )

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class LeavePGroupsOut:
    """
    Leave-P-Groups-Out cross-validator

    Provides train/test indices where p groups are left out for testing.

    Parameters
    ----------
    n_groups : int
        Number of groups to leave out in each iteration
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def split(self, X, y=None, groups=None):
        """Generate indices to split data where p groups are left out"""
        if groups is None:
            raise ValueError("groups parameter is required")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if self.n_groups >= n_groups:
            raise ValueError(f"n_groups={self.n_groups} must be less than total groups={n_groups}")

        # Generate all combinations of p groups
        from itertools import combinations

        for test_groups in combinations(unique_groups, self.n_groups):
            test_groups_set = set(test_groups)

            train_idx = []
            test_idx = []

            for idx, group in enumerate(groups):
                if group in test_groups_set:
                    test_idx.append(idx)
                else:
                    train_idx.append(idx)

            yield np.array(train_idx), np.array(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        if groups is None:
            raise ValueError("groups parameter is required")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        from math import comb

        return comb(n_groups, self.n_groups)


class LeaveOneGroupOut:
    """
    Leave-One-Group-Out Cross-Validation (LOGOCV)

    Each group (e.g., patient, hospital) is used once as test set.
    Provides unbiased estimate for new group generalization.

    Parameters
    ----------
    None

    Examples
    --------
    >>> logo = LeaveOneGroupOut()
    >>> for train, test in logo.split(X, y, groups=patient_ids):
    ...     # Each patient used once as test set
    """

    def __init__(self):
        pass

    def split(self, X, y=None, groups=None):
        """
        Generate leave-one-group-out splits

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like
            Group identifiers

        Yields
        ------
        train, test indices
        """
        if groups is None:
            raise ValueError("Groups must be provided for LOGO CV")

        groups = np.array(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups, got {n_groups}")

        indices = np.arange(len(X))

        for test_group in unique_groups:
            test_mask = groups == test_group
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        if groups is None:
            return None
        return len(np.unique(groups))


class RepeatedGroupKFold:
    """
    Repeated Group K-Fold Cross-Validation

    Performs group k-fold CV multiple times with different randomization.
    Reduces variance in performance estimates for grouped data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    n_repeats : int, default=10
        Number of times to repeat the CV
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate repeated group k-fold splits

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        groups : array-like
            Group identifiers

        Yields
        ------
        train, test indices
        """
        if groups is None:
            raise ValueError("Groups must be provided")

        rng = np.random.RandomState(self.random_state)

        for repeat_idx in range(self.n_repeats):
            # Use different random state for each repeat
            repeat_seed = None if self.random_state is None else rng.randint(0, 2**32 - 1)

            gkf = GroupKFoldMedical(n_splits=self.n_splits, shuffle=True, random_state=repeat_seed)

            for train_idx, test_idx in gkf.split(X, y, groups):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits * self.n_repeats


class NestedGroupedCV:
    """
    Nested Cross-Validation for Grouped Data

    Performs nested CV while preserving group structure.
    Outer loop for evaluation, inner loop for hyperparameter tuning.

    Parameters
    ----------
    outer_cv : grouped CV object
        Outer CV for model evaluation
    inner_cv : grouped CV object
        Inner CV for hyperparameter tuning
    """

    def __init__(self, outer_cv=None, inner_cv=None):
        if outer_cv is None:
            outer_cv = GroupKFoldMedical(n_splits=5)
        if inner_cv is None:
            inner_cv = GroupKFoldMedical(n_splits=3)

        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

    def fit_predict(self, estimator, X, y, groups, param_grid):
        """
        Perform nested grouped cross-validation

        Parameters
        ----------
        estimator : estimator object
            The model to evaluate
        X : array-like
            Feature matrix
        y : array-like
            Target values
        groups : array-like
            Group identifiers
        param_grid : dict
            Hyperparameter grid for tuning

        Returns
        -------
        scores : list
            Outer CV scores
        best_params : list
            Best parameters for each outer fold
        """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV

        outer_scores = []
        best_params = []

        for train_idx, test_idx in self.outer_cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            # Custom CV iterator for inner loop that respects groups
            def inner_cv_generator():
                for inner_train, inner_test in self.inner_cv.split(X_train, y_train, groups_train):
                    yield inner_train, inner_test

            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator, param_grid, cv=inner_cv_generator(), scoring="accuracy", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Evaluate on outer test set
            y_pred = grid_search.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            outer_scores.append(score)
            best_params.append(grid_search.best_params_)

        return outer_scores, best_params

    def get_n_splits(self):
        """Returns the number of outer CV splits"""
        return self.outer_cv.get_n_splits()


class MultilevelCV:
    """
    Multi-level Cross-Validation for hierarchical data

    Handles data with multiple hierarchical levels (e.g., Hospital → Department → Patient).
    Can perform validation at any level of the hierarchy.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    validation_level : str
        Which hierarchy level to use for splitting
    """

    def __init__(self, n_splits=5, validation_level="level_2"):
        self.n_splits = n_splits
        self.validation_level = validation_level

    def split(self, X, y=None, groups=None):
        """Generate indices for multi-level validation"""
        if groups is None or not isinstance(groups, dict):
            raise ValueError("groups must be a dictionary with hierarchy levels")

        if self.validation_level not in groups:
            raise ValueError(f"validation_level {self.validation_level} not in groups")

        # Use the specified level for splitting
        level_groups = groups[self.validation_level]

        # Create k-fold split at the specified level using native implementation
        gkf = GroupKFoldMedical(n_splits=self.n_splits, shuffle=False)

        for train_idx, test_idx in gkf.split(X, y, level_groups):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class HierarchicalGroupKFold(_BaseKFold):
    """
    Hierarchical Group K-Fold for multi-level medical data

    Handles nested grouping structures like:
    - Hospital → Department → Patient
    - Study Site → Patient → Visit

    When splitting at a higher level (e.g., hospital), all nested entities
    (departments, patients) within that group stay together in train or test.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    hierarchy_level : str, default='patient'
        Level to split on. Must be a key in the hierarchy dict.
    shuffle : bool, default=True
        Whether to shuffle groups before splitting
    random_state : int or None, default=None
        Random state for reproducibility

    Examples
    --------
    >>> from trustcv.splitters import HierarchicalGroupKFold
    >>> hierarchy = {
    ...     'hospital': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'department': [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7],
    ...     'patient': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ... }
    >>> cv = HierarchicalGroupKFold(n_splits=2, hierarchy_level='hospital')
    >>> for train, test in cv.split(X, y, hierarchy=hierarchy):
    ...     # All patients from same hospital stay together
    """

    def __init__(self, n_splits=5, hierarchy_level="patient", shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.hierarchy_level = hierarchy_level

    def split(self, X, y=None, groups=None, hierarchy=None):
        """
        Generate hierarchical splits

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target values
        groups : array-like of shape (n_samples,), optional
            Group labels. If provided without hierarchy, used directly.
        hierarchy : dict, optional
            Dictionary mapping level names to group arrays.
            Each array has shape (n_samples,) with group assignments.
            Example: {'hospital': [...], 'patient': [...]}

        Yields
        ------
        train : ndarray
            Training set indices for this split
        test : ndarray
            Test set indices for this split
        """
        from sklearn.utils import indexable

        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)

        if hierarchy is None:
            if groups is None:
                raise ValueError("Either 'groups' or 'hierarchy' must be provided")
            # Fall back to regular grouped splitting using native implementation
            split_groups = np.asarray(groups)
        else:
            if not isinstance(hierarchy, dict):
                raise ValueError(
                    "hierarchy must be a dictionary mapping level names to group arrays"
                )
            if self.hierarchy_level not in hierarchy:
                available = list(hierarchy.keys())
                raise ValueError(
                    f"hierarchy_level '{self.hierarchy_level}' not found in hierarchy. "
                    f"Available levels: {available}"
                )
            # Use the specified hierarchy level for splitting
            split_groups = np.asarray(hierarchy[self.hierarchy_level])

        if len(split_groups) != n_samples:
            raise ValueError(
                f"Group array length ({len(split_groups)}) must match "
                f"number of samples ({n_samples})"
            )

        # Get unique groups
        unique_groups = np.unique(split_groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"the number of groups={n_groups}"
            )

        # Shuffle groups if requested
        rng = check_random_state(self.random_state)
        if self.shuffle:
            group_order = rng.permutation(unique_groups)
        else:
            group_order = unique_groups

        # Assign groups to folds
        # Distribute groups as evenly as possible across folds
        fold_assignments = np.zeros(n_groups, dtype=int)
        for i, group in enumerate(group_order):
            group_idx = np.where(unique_groups == group)[0][0]
            fold_assignments[group_idx] = i % self.n_splits

        # Generate train/test splits
        for fold in range(self.n_splits):
            # Find which groups are in this fold's test set
            test_group_mask = fold_assignments == fold
            test_groups = unique_groups[test_group_mask]

            # Get sample indices
            test_mask = np.isin(split_groups, test_groups)
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits
