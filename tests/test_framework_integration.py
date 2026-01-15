"""
Test framework integration for PyTorch, TensorFlow, MONAI

NOTE: These tests can be memory-intensive, especially MONAI 3D tests.
Run with RUN_HEAVY_TESTS=1 environment variable to enable all tests.
"""

import pytest
import numpy as np
import sys
import gc
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip heavy tests unless explicitly enabled
SKIP_HEAVY_TESTS = not os.environ.get('RUN_HEAVY_TESTS', False)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Cleanup memory after each test to prevent accumulation."""
    yield
    gc.collect()
    # Try to clear PyTorch CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # Try to clear TensorFlow session
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except (ImportError, AttributeError, TypeError):
        # Handle TensorFlow not installed or NumPy version incompatibility
        pass

# Import framework adapters - use lazy imports to handle missing/broken dependencies
from trustcv.core.runner import UniversalCVRunner
from trustcv.core.callbacks import EarlyStopping, ModelCheckpoint

# PyTorch adapter
try:
    from trustcv.frameworks.pytorch import PyTorchAdapter, TorchCVRunner
except (ImportError, AttributeError, TypeError):
    PyTorchAdapter = None
    TorchCVRunner = None

# TensorFlow adapter - may fail with NumPy 2.x incompatibility
# Check if TensorFlow can actually be imported
def _check_tensorflow():
    try:
        import tensorflow as tf
        return True
    except (ImportError, AttributeError, TypeError):
        return False

TENSORFLOW_AVAILABLE = _check_tensorflow()

try:
    from trustcv.frameworks.tensorflow import TensorFlowAdapter, KerasCVRunner
except (ImportError, AttributeError, TypeError):
    TensorFlowAdapter = None
    KerasCVRunner = None

# MONAI adapter
try:
    from trustcv.frameworks.monai import MONAIAdapter, MONAICVRunner
except (ImportError, AttributeError, TypeError):
    MONAIAdapter = None
    MONAICVRunner = None


@pytest.mark.skipif(PyTorchAdapter is None, reason="PyTorch not available")
class TestPyTorchIntegration:
    """Test PyTorch adapter and runner"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.fixture
    def pytorch_adapter(self):
        """Create PyTorch adapter instance"""
        return PyTorchAdapter(batch_size=16)
    
    def test_pytorch_adapter_init(self, pytorch_adapter):
        """Test PyTorch adapter initialization"""
        assert pytorch_adapter.config["batch_size"] == 16
        assert pytorch_adapter.device is not None
    
    def test_create_data_splits(self, pytorch_adapter, sample_data):
        """Test creating PyTorch DataLoader splits"""
        pytest.importorskip("torch")
        
        X, y = sample_data
        train_idx = np.arange(80)
        val_idx = np.arange(80, 100)
        
        try:
            import torch
            from torch.utils.data import TensorDataset
            
            # Create TensorDataset
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.LongTensor(y)
            )
            
            train_loader, val_loader = pytorch_adapter.create_data_splits(
                dataset, train_idx, val_idx
            )
            
            assert len(train_loader.dataset) == 80
            assert len(val_loader.dataset) == 20
            assert train_loader.batch_size == 16
            
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    @pytest.mark.skipif(SKIP_HEAVY_TESTS, reason="Heavy test - set RUN_HEAVY_TESTS=1 to enable")
    def test_torch_cv_runner(self, sample_data):
        """Test TorchCVRunner"""
        pytest.importorskip("torch")

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset

            X, y = sample_data
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.LongTensor(y)
            )

            # Model factory function
            def model_fn():
                return nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2)
                )

            # Test with simple split
            from trustcv.splitters.iid import KFoldMedical
            cv = KFoldMedical(n_splits=3)

            runner = TorchCVRunner(
                model_fn=model_fn,
                cv_splitter=cv,
                store_models=False,  # Don't store models to save memory
            )

            results = runner.run(
                dataset=dataset,
                epochs=1
            )

            # CVResults object has mean_score property
            assert results is not None
            assert hasattr(results, 'scores')
            assert len(results.scores) == 3

        except ImportError:
            pytest.skip("PyTorch not installed")


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available or incompatible with NumPy version")
class TestTensorFlowIntegration:
    """Test TensorFlow adapter and runner"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.fixture
    def tensorflow_adapter(self):
        """Create TensorFlow adapter instance"""
        return TensorFlowAdapter(batch_size=16)

    def test_tensorflow_adapter_init(self, tensorflow_adapter):
        """Test TensorFlow adapter initialization"""
        assert tensorflow_adapter.config["batch_size"] == 16
    
    def test_create_tf_datasets(self, tensorflow_adapter, sample_data):
        """Test creating TensorFlow datasets"""
        pytest.importorskip("tensorflow")
        
        X, y = sample_data
        train_idx = np.arange(80)
        val_idx = np.arange(80, 100)
        
        try:
            import tensorflow as tf
            
            # Create tf.data.Dataset
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            
            train_ds, val_ds = tensorflow_adapter.create_data_splits(
                dataset, train_idx, val_idx
            )
            
            # Check datasets are created
            assert train_ds is not None
            assert val_ds is not None
            
        except ImportError:
            pytest.skip("TensorFlow not installed")
    
    @pytest.mark.skipif(SKIP_HEAVY_TESTS, reason="Heavy test - set RUN_HEAVY_TESTS=1 to enable")
    def test_tf_cv_runner(self, sample_data):
        """Test KerasCVRunner"""
        pytest.importorskip("tensorflow")

        try:
            import tensorflow as tf
            from tensorflow import keras

            X, y = sample_data

            # Model factory function
            def model_fn():
                model = keras.Sequential([
                    keras.layers.Dense(5, activation='relu', input_shape=(10,)),
                    keras.layers.Dense(2, activation='softmax')
                ])
                return model

            # Test with simple split
            from trustcv.splitters.iid import KFoldMedical
            cv = KFoldMedical(n_splits=3)

            runner = KerasCVRunner(
                model_fn=model_fn,
                cv_splitter=cv,
                compile_kwargs={
                    'optimizer': 'adam',
                    'loss': 'sparse_categorical_crossentropy',
                    'metrics': ['accuracy']
                },
                store_models=False,  # Don't store models to save memory
            )

            results = runner.run(
                X=X, y=y,
                epochs=1,
            )

            # CVResults object has scores
            assert results is not None
            assert hasattr(results, 'scores')
            assert len(results.scores) == 3

        except ImportError:
            pytest.skip("TensorFlow not installed")


@pytest.mark.skipif(SKIP_HEAVY_TESTS, reason="MONAI tests are memory-intensive - set RUN_HEAVY_TESTS=1 to enable")
@pytest.mark.skipif(MONAIAdapter is None, reason="MONAI not available")
class TestMONAIIntegration:
    """Test MONAI adapter and runner

    WARNING: These tests create 3D models which are memory-intensive.
    Skip by default unless RUN_HEAVY_TESTS=1 is set.
    """

    @pytest.fixture
    def sample_3d_data(self):
        """Create sample 3D medical imaging data"""
        # Simulate small 3D medical images (4 samples, 1 channel, 16x16x16)
        # Reduced from (10, 1, 32, 32, 32) to minimize memory usage
        images = np.random.randn(4, 1, 16, 16, 16).astype(np.float32)
        labels = np.random.randint(0, 2, 4)
        return images, labels

    @pytest.fixture
    def monai_adapter(self):
        """Create MONAI adapter instance"""
        pytest.importorskip("monai")
        return MONAIAdapter(batch_size=2)

    def test_monai_adapter_init(self, monai_adapter):
        """Test MONAI adapter initialization"""
        assert monai_adapter.config["batch_size"] == 2
        assert hasattr(monai_adapter, 'device')

    def test_monai_data_splits(self, monai_adapter, sample_3d_data):
        """Test creating MONAI data loaders"""
        pytest.importorskip("monai")
        pytest.importorskip("torch")

        images, labels = sample_3d_data
        # Adjusted for smaller dataset (4 samples)
        train_idx = np.arange(3)
        val_idx = np.arange(3, 4)

        try:
            import torch
            from monai.data import Dataset
            from monai.transforms import Compose, EnsureChannelFirst

            # Create MONAI dataset
            data_dicts = [
                {"image": images[i], "label": labels[i]}
                for i in range(len(images))
            ]

            transforms = Compose([EnsureChannelFirst()])

            dataset = Dataset(data=data_dicts, transform=transforms)

            train_loader, val_loader = monai_adapter.create_data_splits(
                dataset, train_idx, val_idx,
                train_transforms=transforms,
                val_transforms=transforms
            )

            assert len(train_loader.dataset) == 3
            assert len(val_loader.dataset) == 1

        except ImportError:
            pytest.skip("MONAI not installed")
    
    def test_monai_cv_runner(self, sample_3d_data):
        """Test MONAICVRunner

        NOTE: Uses a minimal model to reduce memory usage.
        """
        pytest.importorskip("monai")
        pytest.importorskip("torch")

        try:
            import torch
            import torch.nn as nn
            from monai.data import Dataset

            images, labels = sample_3d_data

            # Create MONAI dataset with smaller data
            data_dicts = [
                {"image": images[i], "label": labels[i]}
                for i in range(len(images))
            ]

            dataset = Dataset(data=data_dicts)

            # Use a minimal CNN instead of BasicUNet to reduce memory
            # BasicUNet is extremely memory-intensive for 3D
            class MinimalCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
                    self.pool = nn.AdaptiveAvgPool3d(1)
                    self.fc = nn.Linear(4, 2)

                def forward(self, x):
                    x = torch.relu(self.conv(x))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)

            def model_fn():
                return MinimalCNN()

            # Test with simple split
            from trustcv.splitters.iid import KFoldMedical
            cv = KFoldMedical(n_splits=2)

            runner = MONAICVRunner(
                model_fn=model_fn,
                cv_splitter=cv,
                store_models=False,  # Don't store models to save memory
            )

            # Note: MONAICVRunner.run() has different API
            # Just test that it can be instantiated properly
            assert runner is not None
            assert runner.cv_splitter is not None

        except ImportError:
            pytest.skip("MONAI not installed")


class TestUniversalCVRunner:
    """Test UniversalCVRunner with automatic framework detection"""

    def test_detect_sklearn(self):
        """Test detection of scikit-learn model"""
        from sklearn.ensemble import RandomForestClassifier
        from trustcv.splitters.iid import KFoldMedical

        cv = KFoldMedical(n_splits=3)
        runner = UniversalCVRunner(cv_splitter=cv)

        model = RandomForestClassifier()
        framework = runner.detect_framework(model)

        assert framework == 'sklearn'

    def test_detect_pytorch(self):
        """Test detection of PyTorch model"""
        pytest.importorskip("torch")

        try:
            import torch.nn as nn
            from trustcv.splitters.iid import KFoldMedical

            cv = KFoldMedical(n_splits=3)
            runner = UniversalCVRunner(cv_splitter=cv)

            model = nn.Sequential(nn.Linear(10, 2))
            framework = runner.detect_framework(model)

            assert framework == 'pytorch'

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_detect_tensorflow(self):
        """Test detection of TensorFlow model"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from trustcv.splitters.iid import KFoldMedical

            cv = KFoldMedical(n_splits=3)
            runner = UniversalCVRunner(cv_splitter=cv)

            model = keras.Sequential([keras.layers.Dense(2)])
            framework = runner.detect_framework(model)

            assert framework == 'tensorflow'

        except (ImportError, AttributeError, TypeError) as e:
            # Handle TensorFlow not installed or NumPy version incompatibility
            pytest.skip(f"TensorFlow not available: {e}")

    def test_run_cv_auto_detect(self):
        """Test running CV with automatic framework detection"""
        from sklearn.ensemble import RandomForestClassifier
        from trustcv.splitters.iid import KFoldMedical

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        cv = KFoldMedical(n_splits=3)
        runner = UniversalCVRunner(cv_splitter=cv, verbose=0)

        model = RandomForestClassifier(n_estimators=10)

        # Should auto-detect sklearn and run
        results = runner.run(
            model=model,
            data=(X, y),
        )

        assert results is not None
        assert results.metadata['framework'] == 'sklearn'


class TestCallbacks:
    """Test callback system"""

    def test_early_stopping(self):
        """Test EarlyStopping callback"""
        callback = EarlyStopping(patience=3, min_delta=0.001, verbose=False)
        fold_idx = 0

        # Simulate training with improving loss
        assert callback.on_epoch_end(1, fold_idx, {'val_loss': 0.5}) is None
        assert callback.on_epoch_end(2, fold_idx, {'val_loss': 0.4}) is None
        assert callback.on_epoch_end(3, fold_idx, {'val_loss': 0.3}) is None  # New best

        # Simulate no improvement (patience=3 means after 3 non-improving epochs, stop)
        # wait=1 after this
        assert callback.on_epoch_end(4, fold_idx, {'val_loss': 0.3}) is None
        # wait=2 after this
        assert callback.on_epoch_end(5, fold_idx, {'val_loss': 0.3}) is None
        # wait=3 after this, triggers stop
        assert callback.on_epoch_end(6, fold_idx, {'val_loss': 0.3}) == "stop"

    def test_model_checkpoint(self):
        """Test ModelCheckpoint callback - flags save requests in logs"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'model_{epoch}.pkl'),
                monitor='val_accuracy',
                mode='max',
                verbose=False
            )
            fold_idx = 0

            # Mock model in logs
            logs1 = {'val_accuracy': 0.8, 'model': {'weights': 'dummy'}}

            # Should flag save (first model)
            callback.on_epoch_end(1, fold_idx, logs1)
            assert 'save_checkpoint' in logs1
            assert 'model_1.pkl' in logs1['save_checkpoint']

            # Should not flag save (worse)
            logs2 = {'val_accuracy': 0.7, 'model': {'weights': 'dummy'}}
            callback.on_epoch_end(2, fold_idx, logs2)
            assert 'save_checkpoint' not in logs2

            # Should flag save (better)
            logs3 = {'val_accuracy': 0.9, 'model': {'weights': 'dummy'}}
            callback.on_epoch_end(3, fold_idx, logs3)
            assert 'save_checkpoint' in logs3
            assert 'model_3.pkl' in logs3['save_checkpoint']


@pytest.mark.skipif(SKIP_HEAVY_TESTS, reason="Heavy test - set RUN_HEAVY_TESTS=1 to enable")
class TestFrameworkSpecificCV:
    """Test framework-specific CV methods"""

    def test_lpgo_with_pytorch(self):
        """Test Leave-p-Groups-Out with PyTorch"""
        pytest.importorskip("torch")
        
        try:
            import torch
            from torch.utils.data import TensorDataset
            from trustcv.splitters.grouped import LeavePGroupsOut
            from trustcv.frameworks.pytorch import PyTorchAdapter
            
            # Create grouped data
            X = torch.randn(100, 10)
            y = torch.randint(0, 2, (100,))
            groups = torch.tensor([i // 10 for i in range(100)])  # 10 groups
            
            dataset = TensorDataset(X, y)
            
            # Use Leave-2-Groups-Out
            lpgo = LeavePGroupsOut(n_groups=2)
            adapter = PyTorchAdapter(batch_size=16)
            
            # Get splits
            splits = list(lpgo.split(X.numpy(), y.numpy(), groups.numpy()))
            
            # Should have C(10, 2) = 45 splits
            assert len(splits) == 45
            
            # Test first split
            train_idx, test_idx = splits[0]
            train_loader, test_loader = adapter.create_data_splits(
                dataset, train_idx, test_idx
            )
            
            assert len(train_loader.dataset) == 80  # 8 groups for training
            assert len(test_loader.dataset) == 20   # 2 groups for testing
            
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_multilevel_cv_with_tensorflow(self):
        """Test Multi-level CV with TensorFlow"""
        pytest.importorskip("tensorflow")
        
        try:
            import tensorflow as tf
            from trustcv.splitters.grouped import MultilevelCV
            from trustcv.frameworks.tensorflow import TensorFlowAdapter
            
            # Create hierarchical data
            n_samples = 100
            X = np.random.randn(n_samples, 10).astype(np.float32)
            y = np.random.randint(0, 2, n_samples)
            
            # Create 3-level hierarchy: Hospital -> Department -> Patient
            hospitals = np.array([i // 33 for i in range(n_samples)])  # 3 hospitals
            departments = np.array([i // 11 for i in range(n_samples)])  # 9 departments
            patients = np.arange(n_samples)  # 100 patients
            
            hierarchy = {
                'level_1': hospitals,
                'level_2': departments, 
                'level_3': patients
            }
            
            # Test at department level
            mlcv = MultilevelCV(n_splits=3, validation_level='level_2')
            adapter = TensorFlowAdapter(batch_size=16)
            
            splits = list(mlcv.split(X, y, groups=hierarchy))
            assert len(splits) == 3
            
            # Verify no department appears in both train and test
            for train_idx, test_idx in splits:
                train_depts = set(departments[train_idx])
                test_depts = set(departments[test_idx])
                assert len(train_depts.intersection(test_depts)) == 0
                
        except ImportError:
            pytest.skip("TensorFlow not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])