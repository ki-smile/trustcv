"""
Visualization utilities for cross-validation methods

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

import warnings
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_cv_splits(
    cv, X, y=None, groups=None, n_splits=5, figsize=(10, 6), title: Optional[str] = None
):
    """
    Plot cross-validation splits for visualization

    Parameters
    ----------
    cv : cross-validator object
        Cross-validation splitter
    X : array-like
        Feature data
    y : array-like, optional
        Target data
    groups : array-like, optional
        Group labels for grouped CV
    n_splits : int
        Number of splits to visualize
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate splits
    splits = list(cv.split(X, y, groups))[:n_splits]
    n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]

    # Create visualization
    for i, (train_idx, test_idx) in enumerate(splits):
        # Plot train samples
        train_mask = np.zeros(n_samples)
        train_mask[train_idx] = 1
        ax.scatter(
            range(n_samples),
            [i] * n_samples,
            c=train_mask,
            cmap="RdBu",
            vmin=0,
            vmax=1,
            marker="s",
            s=10,
            alpha=0.7,
        )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("CV Fold")
    ax.set_title(title or "Cross-Validation Splits")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Train (Blue) / Test (Red)")

    plt.tight_layout()
    return fig, ax


def plot_cv_indices(
    cv, X, y=None, groups=None, n_splits=5, figsize=(12, 8), title: Optional[str] = None
):
    """
    Plot detailed cross-validation indices

    Parameters
    ----------
    cv : cross-validator object
        Cross-validation splitter
    X : array-like
        Feature data
    y : array-like, optional
        Target data
    groups : array-like, optional
        Group labels for grouped CV
    n_splits : int
        Number of splits to visualize
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib figure and axes array
    """
    fig, axes = plt.subplots(n_splits, 1, figsize=figsize, sharex=True)
    if n_splits == 1:
        axes = [axes]

    splits = list(cv.split(X, y, groups))[:n_splits]
    n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]

    for i, (train_idx, test_idx) in enumerate(splits):
        ax = axes[i]

        # Create binary masks
        indices = np.arange(n_samples)
        train_mask = np.isin(indices, train_idx)
        test_mask = np.isin(indices, test_idx)

        # Plot
        ax.bar(
            indices[train_mask], np.ones(train_mask.sum()), color="blue", alpha=0.5, label="Train"
        )
        ax.bar(indices[test_mask], np.ones(test_mask.sum()), color="red", alpha=0.5, label="Test")

        ax.set_ylabel(f"Fold {i+1}")
        ax.set_ylim([0, 1.2])
        ax.set_yticks([])

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(title or "Cross-Validation Split Indices", fontsize=14)
    plt.tight_layout()
    return fig, axes


def plot_temporal_cv(cv, n_samples=100, n_splits=5, figsize=(12, 6), title: Optional[str] = None):
    """
    Visualize temporal cross-validation strategy

    Parameters
    ----------
    cv : temporal cross-validator
        Temporal CV splitter
    n_samples : int
        Number of samples to simulate
    n_splits : int
        Number of splits
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate sample data
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)

    # Get splits
    splits = list(cv.split(X, y))[:n_splits]

    # Visualize
    for i, (train_idx, test_idx) in enumerate(splits):
        # Plot train period
        ax.barh(i, len(train_idx), left=train_idx[0], color="blue", alpha=0.6, height=0.6)
        # Plot test period
        ax.barh(i, len(test_idx), left=test_idx[0], color="red", alpha=0.6, height=0.6)

    ax.set_xlabel("Time Index")
    ax.set_ylabel("CV Fold")
    ax.set_title(title or "Temporal Cross-Validation Strategy")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])
    ax.legend(["Train", "Test"])

    plt.tight_layout()
    return fig, ax


def plot_grouped_cv(cv, groups, n_splits=5, figsize=(10, 6), title: Optional[str] = None):
    """
    Visualize grouped cross-validation strategy

    Parameters
    ----------
    cv : grouped cross-validator
        Grouped CV splitter
    groups : array-like
        Group labels
    n_splits : int
        Number of splits
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Generate sample data
    n_samples = len(groups)
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)

    # Get unique groups
    unique_groups = np.unique(groups)
    group_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
    group_color_map = dict(zip(unique_groups, group_colors))

    # Get splits
    splits = list(cv.split(X, y, groups))[:n_splits]

    # Visualize
    for i, (train_idx, test_idx) in enumerate(splits):
        for idx in train_idx:
            group = groups[idx]
            ax.scatter(idx, i, color=group_color_map[group], marker="o", s=30, alpha=0.6)
        for idx in test_idx:
            group = groups[idx]
            ax.scatter(idx, i, color=group_color_map[group], marker="x", s=50, alpha=0.8)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("CV Fold")
    ax.set_title(title or "Grouped Cross-Validation Strategy")
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])

    # Add legend for groups
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=f"Group {g}"
        )
        for g, color in group_color_map.items()
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2)

    plt.tight_layout()
    return fig, ax


def plot_spatial_cv(cv, coordinates, n_splits=5, figsize=(12, 8), title: Optional[str] = None):
    """
    Visualize spatial cross-validation strategy

    Parameters
    ----------
    cv : spatial cross-validator
        Spatial CV splitter
    coordinates : array-like of shape (n_samples, 2)
        Spatial coordinates
    n_splits : int
        Number of splits
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    fig, axes = plt.subplots(2, (n_splits + 1) // 2, figsize=figsize)
    axes = axes.flatten()

    # Generate sample data
    n_samples = len(coordinates)
    X = np.random.randn(n_samples, 5)
    y = np.random.randint(0, 2, n_samples)

    # Get splits
    splits = list(cv.split(X, y))[:n_splits]

    # Visualize each fold
    for i, (train_idx, test_idx) in enumerate(splits):
        ax = axes[i]

        # Plot train points
        ax.scatter(
            coordinates[train_idx, 0],
            coordinates[train_idx, 1],
            c="blue",
            alpha=0.5,
            s=20,
            label="Train",
        )
        # Plot test points
        ax.scatter(
            coordinates[test_idx, 0],
            coordinates[test_idx, 1],
            c="red",
            alpha=0.7,
            s=30,
            label="Test",
        )

        ax.set_title(f"Fold {i+1}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if i == 0:
            ax.legend()

    # Hide unused subplots
    for i in range(n_splits, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title or "Spatial Cross-Validation Strategy", fontsize=14)
    plt.tight_layout()
    return fig, axes


def plot_validation_curves(
    train_scores,
    val_scores,
    param_values,
    param_name="Parameter",
    figsize=(10, 6),
    title: Optional[str] = None,
):
    """
    Plot validation curves for hyperparameter tuning

    Parameters
    ----------
    train_scores : array-like
        Training scores
    val_scores : array-like
        Validation scores
    param_values : array-like
        Parameter values
    param_name : str
        Name of the parameter
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot
    ax.plot(param_values, train_mean, "b-", label="Training score")
    ax.fill_between(
        param_values, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
    )

    ax.plot(param_values, val_mean, "r-", label="Validation score")
    ax.fill_between(param_values, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")

    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")
    ax.set_title(title or "Validation Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_learning_curves(
    train_sizes, train_scores, val_scores, figsize=(10, 6), title: Optional[str] = None
):
    """
    Plot learning curves

    Parameters
    ----------
    train_sizes : array-like
        Training set sizes
    train_scores : array-like
        Training scores
    val_scores : array-like
        Validation scores
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot
    ax.plot(train_sizes, train_mean, "b-", marker="o", label="Training score")
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
    )

    ax.plot(train_sizes, val_mean, "r-", marker="s", label="Validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title or "Learning Curves")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


# ---- sklearn-style wrappers ----
def plot_learning_curve(
    estimator,
    X,
    y=None,
    *,
    cv=None,
    scoring=None,
    n_jobs=None,
    train_sizes=None,
    shuffle=True,
    random_state=None,
    groups=None,
    figsize=(10, 6),
    title: Optional[str] = None,
):
    """
    Wrapper to compute and plot learning curves via scikit-learn.

    Accepts trustcv or sklearn CV splitters in `cv`. Pass `groups` when using
    grouped CV.
    """
    import numpy as _np
    from sklearn.model_selection import learning_curve as _learning_curve

    if train_sizes is None:
        train_sizes = _np.linspace(0.1, 1.0, 5)

    # Call sklearn with widest signature available; fall back if needed
    try:
        sizes, train_scores, val_scores, fit_times, _ = _learning_curve(
            estimator,
            X,
            y,
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            shuffle=shuffle,
            random_state=random_state,
            return_times=True,
        )
    except TypeError:
        sizes, train_scores, val_scores, fit_times, _ = _learning_curve(
            estimator,
            X,
            y,
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )

    fig, ax = plot_learning_curves(sizes, train_scores, val_scores, figsize=figsize, title=title)
    return fig


def plot_validation_curve(
    estimator,
    X,
    y,
    *,
    param_name: str,
    param_range,
    cv=None,
    scoring=None,
    n_jobs=None,
    groups=None,
    logx: bool = False,
    figsize=(10, 6),
    title: Optional[str] = None,
):
    """
    Wrapper to compute and plot validation curve via scikit-learn.

    Accepts trustcv or sklearn CV splitters in `cv`. Pass `groups` when using
    grouped CV.
    """
    from sklearn.model_selection import validation_curve as _validation_curve

    train_scores, val_scores = _validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    fig, ax = plot_validation_curves(
        train_scores, val_scores, param_range, param_name=param_name, figsize=figsize, title=title
    )
    if logx:
        try:
            ax.set_xscale("log")
        except Exception:
            pass
    return fig
