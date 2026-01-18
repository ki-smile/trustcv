"""
JAX/Flax Cross-Validation with TrustCV

This example demonstrates how to use JAX/Flax neural networks
with TrustCV for medical/healthcare ML applications.

JAX uses a functional programming paradigm:
- Models are stateless pure functions
- Parameters are managed explicitly via pytrees
- JIT compilation provides significant speedups
- Flax provides a high-level neural network API

Requirements:
    pip install jax jaxlib flax optax

Usage:
    python jax_flax_cv.py
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Check if JAX is available
JAX_AVAILABLE = False
FLAX_AVAILABLE = False
OPTAX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError as e:
    print(f"JAX not available: {e}")

try:
    import flax
    import flax.linen as nn
    from flax.training import train_state

    FLAX_AVAILABLE = True
    print(f"Flax version: {flax.__version__}")
except ImportError as e:
    print(f"Flax not available: {e}")

try:
    import optax

    OPTAX_AVAILABLE = True
except ImportError as e:
    print(f"Optax not available: {e}")


def create_medical_dataset(n_patients=200, samples_per_patient=5, n_features=20):
    """
    Create a synthetic medical dataset with patient grouping.

    Returns:
        X: Feature matrix
        y: Labels
        patient_ids: Patient group identifiers
    """
    n_samples = n_patients * samples_per_patient

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=3,
        weights=[0.7, 0.3],
        random_state=42,
    )

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create patient IDs
    patient_ids = np.repeat(np.arange(n_patients), samples_per_patient)

    return X, y, patient_ids


def define_mlp_model():
    """
    Define a simple MLP using Flax.

    Flax uses a functional approach where the model is defined
    as a class with a __call__ method.
    """

    class MLP(nn.Module):
        """Multi-layer perceptron for binary classification."""

        hidden_dim: int = 64
        n_classes: int = 2
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(self, x, training: bool = True):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
            x = nn.Dense(self.hidden_dim // 2)(x)
            x = nn.relu(x)
            x = nn.Dense(self.n_classes)(x)
            return x

    return MLP


def run_jax_cv_low_level():
    """
    Example using JAXAdapter directly for fine-grained control.
    """
    print("\n" + "=" * 60)
    print("JAX/Flax Cross-Validation (Low-Level API)")
    print("=" * 60)

    from trustcv.frameworks.jax import JAXAdapter
    from trustcv.splitters import StratifiedKFold

    # Create dataset
    print("\n1. Creating synthetic medical dataset...")
    X, y, patient_ids = create_medical_dataset(n_patients=100, samples_per_patient=3)
    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Class distribution: {np.bincount(y)}")

    # Initialize adapter
    print("\n2. Initializing JAX adapter...")
    adapter = JAXAdapter(batch_size=32, seed=42, use_jit=True)

    # Define model
    print("\n3. Defining Flax MLP model...")
    MLP = define_mlp_model()
    model = MLP(hidden_dim=64, n_classes=2)

    # Cross-validation splitter
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Manual CV loop
    print("\n4. Running cross-validation...")
    all_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        print(f"\n   Fold {fold_idx + 1}/3")

        # Create data splits
        train_data, val_data = adapter.create_data_splits((X, y), train_idx, val_idx)

        # Training
        state = None
        optimizer = optax.adam(1e-3)

        for epoch in range(10):
            result = adapter.train_epoch(
                model, train_data, optimizer=optimizer, state=state
            )
            state = result["state"]

        # Evaluation
        val_metrics = adapter.evaluate(model, val_data, state=state)
        print(f"   Val accuracy: {val_metrics['val_acc']:.4f}")
        all_scores.append(val_metrics["val_acc"])

    print(f"\n   Mean accuracy: {np.mean(all_scores):.4f} (+/- {np.std(all_scores):.4f})")

    return all_scores


def run_jax_cv_high_level():
    """
    Example using JAXCVRunner for simplified workflow.
    """
    print("\n" + "=" * 60)
    print("JAX/Flax Cross-Validation (High-Level API)")
    print("=" * 60)

    from trustcv.frameworks.jax import JAXAdapter, JAXCVRunner
    from trustcv.splitters import StratifiedKFold

    # Create dataset
    print("\n1. Creating synthetic medical dataset...")
    X, y, patient_ids = create_medical_dataset(n_patients=100, samples_per_patient=3)
    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Class distribution: {np.bincount(y)}")

    # Define model factory
    MLP = define_mlp_model()

    def model_fn():
        return MLP(hidden_dim=64, n_classes=2)

    # Initialize CV runner
    print("\n2. Initializing JAXCVRunner...")
    runner = JAXCVRunner(
        model_fn=model_fn,
        cv_splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        adapter=JAXAdapter(batch_size=32, seed=42),
    )

    # Run cross-validation
    print("\n3. Running 5-fold cross-validation...")
    results = runner.run(X, y, epochs=15, optimizer=optax.adam(1e-3))

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results.summary())

    return results


def run_grouped_cv_example():
    """
    Example with patient-grouped cross-validation.
    """
    print("\n" + "=" * 60)
    print("Patient-Grouped Cross-Validation with JAX/Flax")
    print("=" * 60)

    from trustcv import TrustCVValidator
    from trustcv.frameworks.jax import JAXAdapter, JAXCVRunner
    from trustcv.splitters import StratifiedGroupKFold

    # Create dataset with patient grouping
    print("\n1. Creating dataset with patient grouping...")
    X, y, patient_ids = create_medical_dataset(n_patients=100, samples_per_patient=5)
    print(f"   - Samples: {len(X)}")
    print(f"   - Patients: {len(np.unique(patient_ids))}")
    print(f"   - Samples per patient: 5")

    # Define model
    MLP = define_mlp_model()

    def model_fn():
        return MLP(hidden_dim=64, n_classes=2)

    # Use grouped splitter to prevent patient leakage
    print("\n2. Using Stratified Group K-Fold (no patient leakage)...")
    runner = JAXCVRunner(
        model_fn=model_fn,
        cv_splitter=StratifiedGroupKFold(n_splits=5),
        adapter=JAXAdapter(batch_size=32, seed=42),
    )

    # Run with groups
    print("\n3. Running cross-validation with patient grouping...")
    results = runner.run(X, y, epochs=10, groups=patient_ids)

    print("\n" + "=" * 60)
    print("RESULTS (Patient-Grouped CV)")
    print("=" * 60)
    print(results.summary())

    print("\nKey benefit: No patient appears in both train and test sets!")

    return results


def compare_with_sklearn():
    """
    Compare JAX/Flax model with sklearn baseline.
    """
    print("\n" + "=" * 60)
    print("Comparison: JAX/Flax MLP vs sklearn RandomForest")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier

    from trustcv import TrustCVValidator
    from trustcv.frameworks.jax import JAXAdapter, JAXCVRunner
    from trustcv.splitters import StratifiedKFold

    # Create dataset
    X, y, _ = create_medical_dataset(n_patients=100, samples_per_patient=3)

    # sklearn baseline
    print("\n1. Evaluating sklearn RandomForest...")
    validator = TrustCVValidator(
        method="stratified_kfold",
        n_splits=5,
        random_state=42,
    )
    rf_results = validator.validate(
        model=RandomForestClassifier(n_estimators=100, random_state=42), X=X, y=y
    )

    # JAX/Flax MLP
    print("\n2. Evaluating JAX/Flax MLP...")
    MLP = define_mlp_model()
    runner = JAXCVRunner(
        model_fn=lambda: MLP(hidden_dim=64, n_classes=2),
        cv_splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        adapter=JAXAdapter(batch_size=32, seed=42),
    )
    jax_results = runner.run(X, y, epochs=20)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # TrustCVValidator returns ValidationResult with mean_scores/std_scores
    rf_acc = rf_results.mean_scores.get("accuracy", 0)
    rf_std = rf_results.std_scores.get("accuracy", 0)

    # JAXCVRunner returns CVResults with metrics property
    jax_acc = jax_results.metrics.get("val_acc", {}).get("mean", 0)
    jax_std = jax_results.metrics.get("val_acc", {}).get("std", 0)

    print(f"\n{'Model':<25} {'Accuracy':<20}")
    print("-" * 45)
    print(f"{'RandomForest (sklearn)':<25} {rf_acc:.4f} (+/- {rf_std:.4f})")
    print(f"{'MLP (JAX/Flax)':<25} {jax_acc:.4f} (+/- {jax_std:.4f})")


if __name__ == "__main__":
    if not all([JAX_AVAILABLE, FLAX_AVAILABLE, OPTAX_AVAILABLE]):
        print("\n" + "=" * 60)
        print("JAX/Flax/Optax not fully available!")
        print("=" * 60)
        print("\nInstall with: pip install jax jaxlib flax optax")
        print("\nFor GPU support, see: https://github.com/google/jax#installation")
    else:
        # Run examples
        run_jax_cv_low_level()
        run_jax_cv_high_level()
        run_grouped_cv_example()
        compare_with_sklearn()

        print("\n" + "=" * 60)
        print("All JAX/Flax examples completed successfully!")
        print("=" * 60)
