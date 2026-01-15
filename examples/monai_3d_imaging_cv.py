#!/usr/bin/env python3
"""
MONAI 3D Medical Imaging Cross-Validation Example

Demonstrates how to use trustcv with MONAI for 3D medical image classification
with proper patient-level cross-validation.

Developed at SMAILE (Stockholm Medical AI and Learning Environments)
Karolinska Institutet - https://smile.ki.se

Requirements:
    pip install monai torch
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import trustcv components
from trustcv.splitters import GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut

np.random.seed(42)
torch.manual_seed(42)

print("=" * 70)
print("MONAI 3D Medical Imaging Cross-Validation with trustcv")
print("=" * 70)

# Check if MONAI is available
try:
    import monai
    from monai.networks.nets import DenseNet121, ResNet
    from monai.transforms import Compose, RandRotate90, RandFlip, ToTensor
    MONAI_AVAILABLE = True
    print(f"MONAI version: {monai.__version__}")
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI not installed. Install with: pip install monai")
    print("Running in simulation mode...")


# =============================================================================
# 1. Create Synthetic 3D Medical Image Dataset
# =============================================================================
class Synthetic3DMedicalDataset(Dataset):
    """
    Synthetic 3D medical imaging dataset.

    In a real scenario, this would load CT/MRI scans from disk.
    Each patient may have multiple scans (e.g., follow-up imaging).
    """

    def __init__(self, n_samples=100, image_size=(64, 64, 32), n_channels=1):
        self.n_samples = n_samples
        self.image_size = image_size
        self.n_channels = n_channels

        # Simulate patient structure (multiple scans per patient)
        n_patients = n_samples // 3
        self.patient_ids = np.repeat(np.arange(n_patients), 3)[:n_samples]

        # Generate synthetic labels (e.g., tumor present/absent)
        # Make it correlate with patient (patient-level disease)
        patient_labels = np.random.binomial(1, 0.3, n_patients)
        self.labels = patient_labels[self.patient_ids]

        # Generate synthetic image data
        np.random.seed(42)
        self.images = []
        for i in range(n_samples):
            # Create base noise image
            img = np.random.randn(n_channels, *image_size).astype(np.float32) * 0.1

            # Add class-specific patterns
            if self.labels[i] == 1:
                # Add "tumor-like" region for positive cases
                center = tuple(s // 2 for s in image_size)
                for c in range(n_channels):
                    img[c, center[0]-5:center[0]+5,
                          center[1]-5:center[1]+5,
                          center[2]-3:center[2]+3] += 0.5

            self.images.append(img)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx]), self.labels[idx]


# =============================================================================
# 2. Define 3D CNN Model (simplified if MONAI not available)
# =============================================================================
class Simple3DCNN(torch.nn.Module):
    """Simple 3D CNN for medical image classification."""

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.features = torch.nn.Sequential(
            # Conv block 1
            torch.nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),

            # Conv block 2
            torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),

            # Conv block 3
            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(use_monai=True, in_channels=1, num_classes=2):
    """Create 3D classification model."""
    if use_monai and MONAI_AVAILABLE:
        # Use MONAI's pre-built 3D DenseNet
        model = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes
        )
    else:
        model = Simple3DCNN(in_channels=in_channels, num_classes=num_classes)
    return model


# =============================================================================
# 3. Training and Evaluation Functions
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    # Calculate AUC if we have both classes
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # Single class in validation

    return {'accuracy': accuracy, 'auc': auc}


# =============================================================================
# 4. Cross-Validation with trustcv Splitters
# =============================================================================
def run_3d_cv(dataset, cv_splitter, n_epochs=10, batch_size=4, use_monai=True):
    """Run cross-validation for 3D medical imaging."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    X = np.arange(len(dataset))  # Indices
    y = np.array(dataset.labels)
    patient_ids = dataset.patient_ids

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y, groups=patient_ids)):
        print(f"\n--- Fold {fold + 1} ---")

        # Verify no patient leakage
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        overlap = train_patients & val_patients
        assert len(overlap) == 0, f"Patient leakage detected! Overlapping patients: {overlap}"

        print(f"Train: {len(train_idx)} samples, {len(train_patients)} patients")
        print(f"Val: {len(val_idx)} samples, {len(val_patients)} patients")

        # Create data loaders using indices
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Initialize model
        model = create_model(use_monai=use_monai).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0
        best_metrics = None

        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                      f"Val Acc={val_metrics['accuracy']:.4f}")

        if best_metrics:
            print(f"  Best Val Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"  Best Val AUC: {best_metrics['auc']:.4f}")
            fold_results.append(best_metrics)

        # Memory cleanup (critical for 3D models)
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()

    return fold_results


# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Create synthetic dataset
    print("\nCreating synthetic 3D medical imaging dataset...")
    dataset = Synthetic3DMedicalDataset(
        n_samples=60,  # Small for demo (real datasets would be larger)
        image_size=(32, 32, 16),  # Smaller for demo
        n_channels=1
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Image shape: {dataset[0][0].shape}")
    print(f"Patients: {len(np.unique(dataset.patient_ids))}")
    print(f"Class distribution: {np.bincount(dataset.labels)}")

    # Example 1: Patient-Grouped K-Fold
    print("\n" + "=" * 70)
    print("Example 1: Patient-Grouped K-Fold for 3D Medical Imaging")
    print("=" * 70)

    cv_grouped = GroupKFold(n_splits=3)
    results_grouped = run_3d_cv(
        dataset, cv_grouped,
        n_epochs=10, batch_size=4,
        use_monai=MONAI_AVAILABLE
    )

    if results_grouped:
        print("\n--- Summary: Patient-Grouped K-Fold ---")
        accuracies = [r['accuracy'] for r in results_grouped]
        aucs = [r['auc'] for r in results_grouped]
        print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Example 2: Leave-One-Patient-Out (for small datasets)
    print("\n" + "=" * 70)
    print("Example 2: Leave-One-Group-Out (Per-Patient Validation)")
    print("=" * 70)
    print("(Useful for external validation simulation)")

    # Only run 3 folds for demo
    cv_logo = LeaveOneGroupOut()
    fold_count = 0
    logo_results = []

    X = np.arange(len(dataset))
    y = np.array(dataset.labels)
    patient_ids = dataset.patient_ids

    for fold, (train_idx, val_idx) in enumerate(cv_logo.split(X, y, groups=patient_ids)):
        if fold_count >= 3:
            break

        print(f"\nFold {fold + 1}: Testing on patient {patient_ids[val_idx[0]]}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        fold_count += 1

    print("\n" + "=" * 70)
    print("MONAI 3D Medical Imaging CV Demo Complete!")
    print("=" * 70)
    print("\nKey Points for 3D Medical Imaging CV:")
    print("1. Patient-level grouping prevents data leakage")
    print("2. Same patient's scans never in both train and validation")
    print("3. Memory management is critical for 3D models")
    print("4. MONAI provides optimized 3D architectures")
    print("5. Smaller batch sizes needed for 3D data")
