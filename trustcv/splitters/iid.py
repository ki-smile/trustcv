"""
I.I.D. (Independent and Identically Distributed) Cross-Validation Methods

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility
Website: https://smile.ki.se

Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

import warnings
from typing import Iterator, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state, indexable
from sklearn.utils.validation import check_array


class HoldOut:
    """
    Hold-Out (Train-Test Split) Validation

    Splits data into single training and test sets.
    Computational complexity: O(T(n))

    Parameters
    ----------
    test_size : float or int, default=0.2
        If float, represents proportion of dataset for test set (0.0, 1.0)
        If int, represents absolute number of test samples
    random_state : int or None, default=None
        Random state for reproducibility
    stratify : array-like or None, default=None
        Data to use for stratified splitting
    """

    def __init__(
        self,
        test_size: Union[float, int] = 0.2,
        random_state: Optional[int] = None,
        stratify: Optional[np.ndarray] = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y=None, groups=None):
        """Generate train/test indices"""
        X, y = indexable(X, y)
        n_samples = len(X)

        if isinstance(self.test_size, float):
            n_test = int(n_samples * self.test_size)
        else:
            n_test = self.test_size

        rng = check_random_state(self.random_state)

        if self.stratify is not None and y is not None:
            # Stratified split
            from sklearn.model_selection import train_test_split

            indices = np.arange(n_samples)
            train_idx, test_idx = train_test_split(
                indices, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        else:
            # Random split
            indices = rng.permutation(n_samples)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

        yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns number of splits (always 1 for hold-out)"""
        return 1


class KFoldMedical(BaseCrossValidator):
    """
    Standard k-Fold Cross-Validation with medical considerations

    Computational complexity: O(k·T(n(k-1)/k))

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=False
        Whether to shuffle data before splitting
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate test indices for each fold"""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for each fold"""
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)

        for test_indices in self._iter_test_indices(X, y, groups):
            train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class StratifiedKFoldMedical(BaseCrossValidator):
    """
    Stratified k-Fold Cross-Validation

    Maintains class distribution in each fold.
    Particularly important for imbalanced medical datasets.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=False
        Whether to shuffle data before splitting
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """Generate stratified train/test indices"""
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        for train_idx, test_idx in skf.split(X, y):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits


class RepeatedKFold(BaseCrossValidator):
    """
    Repeated k-Fold Cross-Validation

    Performs k-fold CV multiple times with different randomization.
    Reduces variance in performance estimates.

    Computational complexity: O(r·k·T(n(k-1)/k))

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    n_repeats : int, default=10
        Number of times to repeat the k-fold CV
    random_state : int or None, default=None
        Random state for reproducibility
    stratify : bool, default=False
        If True, uses RepeatedStratifiedKFold (requires labels)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        stratify: bool = False,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for repeated k-fold"""
        if self.stratify:
            if y is None:
                raise ValueError("Stratified repeated k-fold requires y labels.")
            from sklearn.model_selection import RepeatedStratifiedKFold

            rkf = RepeatedStratifiedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
            )
            splitter = rkf.split(X, y)
        else:
            from sklearn.model_selection import RepeatedKFold as _RepeatedKFold

            rkf = _RepeatedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
            )
            splitter = rkf.split(X, y)

        for train_idx, test_idx in splitter:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_splits * self.n_repeats


class LOOCV(BaseCrossValidator):
    """
    Leave-One-Out Cross-Validation

    Each sample is used once as test set.
    Provides nearly unbiased estimate but computationally expensive.

    Computational complexity: O(n·T(n-1))

    Warning: Not recommended for large datasets (n > 1000)
    """

    def __init__(self):
        pass

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for LOOCV"""
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)

        if n_samples > 1000:
            warnings.warn(
                f"LOOCV with {n_samples} samples is computationally expensive. "
                "Consider using k-fold CV instead.",
                UserWarning,
            )

        for i in range(n_samples):
            test_idx = np.array([i])
            train_idx = np.concatenate([np.arange(i), np.arange(i + 1, n_samples)])
            yield train_idx, test_idx

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return len(X)


class LPOCV(BaseCrossValidator):
    """
    Leave-p-Out Cross-Validation

    Generalization of LOOCV where p samples are left out.
    Number of iterations: C(n, p) = n!/(p!(n-p)!)

    Parameters
    ----------
    p : int
        Number of samples to leave out in each iteration

    Warning: Combinatorial explosion for large p
    """

    def __init__(self, p: int):
        if p < 1:
            raise ValueError("p must be at least 1")
        self.p = p

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for LPOCV"""
        from itertools import combinations

        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)

        if self.p >= n_samples:
            raise ValueError(f"p ({self.p}) must be less than n_samples ({n_samples})")

        # Check for combinatorial explosion
        from math import comb

        n_iterations = comb(n_samples, self.p)
        if n_iterations > 1000:
            warnings.warn(
                f"LPOCV will generate {n_iterations} iterations. "
                "This may be computationally expensive.",
                UserWarning,
            )

        indices = np.arange(n_samples)
        for test_idx in combinations(indices, self.p):
            test_idx = np.array(test_idx)
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations"""
        from math import comb

        n_samples = len(X)
        return comb(n_samples, self.p)


class BootstrapValidation:
    """
    Bootstrap Validation with .632 and .632+ estimators

    Samples with replacement for training, uses out-of-bag samples for testing.

    Parameters
    ----------
    n_iterations : int, default=100
        Number of bootstrap iterations
    estimator : str, default='standard'
        Type of bootstrap estimator: 'standard', '.632', or '.632+'
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self,
        n_iterations: int = 100,
        estimator: str = "standard",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        # Backward-compatibility: n_splits -> n_iterations
        if "n_splits" in kwargs and n_iterations == 100:
            n_iterations = kwargs.pop("n_splits")
        if kwargs:
            pass

        self.n_iterations = n_iterations
        self.estimator = estimator
        self.random_state = random_state

        if estimator not in ["standard", ".632", ".632+"]:
            raise ValueError(f"Invalid estimator: {estimator}")

    def split(self, X, y=None, groups=None):
        """Generate bootstrap train/test indices"""
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        rng = check_random_state(self.random_state)

        for _ in range(self.n_iterations):
            # Sample with replacement
            train_idx = rng.choice(n_samples, size=n_samples, replace=True)
            # Out-of-bag samples
            test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

            if len(test_idx) == 0:
                # No out-of-bag samples, skip this iteration
                continue

            yield train_idx, test_idx

    def compute_error(self, train_errors, test_errors, y_true=None, y_pred_train=None):
        """
        Compute bootstrap error estimate

        Parameters
        ----------
        train_errors : array-like
            Training errors for each iteration
        test_errors : array-like
            Test (out-of-bag) errors for each iteration
        y_true : array-like, optional
            True labels (needed for .632+ estimator)
        y_pred_train : array-like, optional
            Training predictions (needed for .632+ estimator)
        """
        if self.estimator == "standard":
            return np.mean(test_errors)

        elif self.estimator == ".632":
            # .632 estimator: 0.632 * test_error + 0.368 * train_error
            return 0.632 * np.mean(test_errors) + 0.368 * np.mean(train_errors)

        elif self.estimator == ".632+":
            if y_true is None or y_pred_train is None:
                raise ValueError(".632+ estimator requires y_true and y_pred_train")

            # Calculate no-information error rate
            from sklearn.metrics import accuracy_score

            # Permute predictions to break relationship with true labels
            rng = check_random_state(self.random_state)
            y_pred_permuted = rng.permutation(y_pred_train)
            gamma = 1 - accuracy_score(y_true, y_pred_permuted)

            # Calculate relative overfitting rate
            err_test = np.mean(test_errors)
            err_train = np.mean(train_errors)
            R = (err_test - err_train) / (gamma - err_train) if gamma > err_train else 0
            R = min(max(R, 0), 1)  # Clip to [0, 1]

            # .632+ weight
            w = 0.632 / (1 - 0.368 * R)

            # .632+ estimator
            return w * err_test + (1 - w) * err_train

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_iterations


class MonteCarloCV(BaseCrossValidator):
    """
    Monte Carlo Cross-Validation (Random Sub-Sampling)

    Randomly splits data multiple times into train/test sets.
    Unlike k-fold, test sets may overlap between iterations.

    Parameters
    ----------
    n_iterations : int, default=100
        Number of random splits
    test_size : float or int, default=0.2
        If float, proportion of dataset for test set (0.0, 1.0)
        If int, absolute number of test samples
    random_state : int or None, default=None
        Random state for reproducibility
    """

    def __init__(
        self,
        n_iterations: int = 100,
        test_size: Union[float, int] = 0.2,
        random_state: Optional[int] = None,
    ):
        self.n_iterations = n_iterations
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate random train/test splits"""
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)

        if isinstance(self.test_size, float):
            n_test = int(n_samples * self.test_size)
        else:
            n_test = self.test_size

        if n_test >= n_samples:
            raise ValueError(f"test_size ({n_test}) must be less than n_samples ({n_samples})")

        rng = check_random_state(self.random_state)

        for _ in range(self.n_iterations):
            indices = rng.permutation(n_samples)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations"""
        return self.n_iterations


class NestedCV:
    """
    Nested Cross-Validation for hyperparameter tuning

    Uses outer CV for model evaluation and inner CV for hyperparameter selection.
    Provides unbiased performance estimate when tuning hyperparameters.

    Parameters
    ----------
    outer_cv : cross-validator object
        Outer cross-validation for model evaluation
    inner_cv : cross-validator object
        Inner cross-validation for hyperparameter tuning
    """

    def __init__(self, outer_cv=None, inner_cv=None):
        if outer_cv is None:
            outer_cv = KFoldMedical(n_splits=5)
        if inner_cv is None:
            inner_cv = KFoldMedical(n_splits=3)

        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

    def fit_predict(self, estimator, X, y, param_grid):
        """
        Perform nested cross-validation

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

        Returns
        -------
        scores : list
            Outer CV scores for each fold
        best_params : list
            Best parameters found for each outer fold
        """
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV

        outer_scores = []
        best_params = []

        for train_idx, test_idx in self.outer_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator, param_grid, cv=self.inner_cv, scoring="accuracy", n_jobs=-1
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


# === Canonical sklearn-style aliases (module-local) ===
# These mirror the package-level aliases in trustcv.splitters.__init__
try:
    KFold
except NameError:
    KFold = KFoldMedical
try:
    StratifiedKFold
except NameError:
    StratifiedKFold = StratifiedKFoldMedical
try:
    LeaveOneOut
except NameError:
    LeaveOneOut = LOOCV
try:
    LeavePOut
except NameError:
    LeavePOut = LPOCV

try:
    __all__
except NameError:
    __all__ = []
for _n in ("KFold", "StratifiedKFold", "LeaveOneOut", "LeavePOut"):
    if _n not in __all__:
        __all__.append(_n)
