#!/usr/bin/env python3
"""
PyTorch Medical Cross-Validation Example

Demonstrates how to use trustcv with PyTorch for medical classification
with proper patient-level cross-validation.

Developed at SMAILE (Stockholm Medical AI and Learning Environments)
Karolinska Institutet - https://smile.ki.se
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Import trustcv components
from trustcv.splitters import GroupKFold, StratifiedGroupKFold, TimeSeriesSplit

np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("PyTorch Medical Cross-Validation with trustcv")
print("=" * 70)


# =============================================================================
# 1. Create Synthetic Medical Dataset
# =============================================================================
def create_medical_dataset(n_samples=500, n_features=30):
    """Create synthetic medical dataset with patient grouping."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=3,
        weights=[0.7, 0.3],  # Imbalanced (common in medical data)
        random_state=42
    )

    # Create patient IDs (multiple samples per patient)
    n_patients = n_samples // 4
    patient_ids = np.repeat(np.arange(n_patients), 4)[:n_samples]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, patient_ids


# =============================================================================
# 2. Define PyTorch Model
# =============================================================================
class MedicalClassifier(nn.Module):
    """Simple neural network for medical classification."""

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# 3. Training and Evaluation Functions
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model on validation data."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {'accuracy': accuracy, 'auc': auc}


# =============================================================================
# 4. Cross-Validation with trustcv Splitters
# =============================================================================
def run_cv_pytorch(X, y, patient_ids, cv_splitter, n_epochs=30, batch_size=32):
    """Run cross-validation with PyTorch model using trustcv splitter."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y, groups=patient_ids)):
        print(f"\n--- Fold {fold + 1} ---")

        # Verify no patient leakage
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        assert len(train_patients & val_patients) == 0, "Patient leakage detected!"

        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

        # Create data loaders
        X_train = torch.FloatTensor(X[train_idx])
        y_train = torch.LongTensor(y[train_idx])
        X_val = torch.FloatTensor(X[val_idx])
        y_val = torch.LongTensor(y[val_idx])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size
        )

        # Initialize model for each fold
        model = MedicalClassifier(input_dim=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_auc = 0
        patience_counter = 0
        patience = 5

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        print(f"  Best Val Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  Best Val AUC: {best_metrics['auc']:.4f}")

        fold_results.append(best_metrics)

        # Memory cleanup
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_results


# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Create dataset
    X, y, patient_ids = create_medical_dataset(n_samples=500)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Patients: {len(np.unique(patient_ids))}")
    print(f"Class distribution: {np.bincount(y)}")

    # Example 1: Patient-Grouped K-Fold
    print("\n" + "=" * 70)
    print("Example 1: Patient-Grouped K-Fold Cross-Validation")
    print("=" * 70)

    cv_grouped = GroupKFold(n_splits=5)
    results_grouped = run_cv_pytorch(X, y, patient_ids, cv_grouped)

    print("\n--- Summary: Patient-Grouped K-Fold ---")
    accuracies = [r['accuracy'] for r in results_grouped]
    aucs = [r['auc'] for r in results_grouped]
    print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Example 2: Stratified Group K-Fold
    print("\n" + "=" * 70)
    print("Example 2: Stratified Patient-Grouped K-Fold")
    print("=" * 70)

    cv_stratified = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    results_stratified = run_cv_pytorch(X, y, patient_ids, cv_stratified)

    print("\n--- Summary: Stratified Patient-Grouped K-Fold ---")
    accuracies = [r['accuracy'] for r in results_stratified]
    aucs = [r['auc'] for r in results_stratified]
    print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Example 3: Time Series Split (for longitudinal data)
    print("\n" + "=" * 70)
    print("Example 3: Time Series Split (Temporal Validation)")
    print("=" * 70)

    cv_temporal = TimeSeriesSplit(n_splits=5)

    # Note: TimeSeriesSplit doesn't use groups parameter
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(cv_temporal.split(X)):
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")

    print("\n" + "=" * 70)
    print("PyTorch Cross-Validation Demo Complete!")
    print("=" * 70)
    print("\nKey Points:")
    print("- Used trustcv splitters for proper patient-level validation")
    print("- No patient appears in both train and validation sets")
    print("- Early stopping prevents overfitting")
    print("- Memory cleanup between folds")
