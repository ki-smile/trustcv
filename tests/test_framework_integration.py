"""
Test framework integration for PyTorch, TensorFlow, MONAI
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import framework adapters
from trustcv.frameworks.pytorch import PyTorchAdapter, TorchCVRunner
from trustcv.frameworks.tensorflow import TensorFlowAdapter, KerasCVRunner  
from trustcv.frameworks.monai import MONAIAdapter, MONAICVRunner
from trustcv.core.runner import UniversalCVRunner
from trustcv.core.callbacks import EarlyStopping, ModelCheckpoint


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
        assert pytorch_adapter.batch_size == 16
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
            
            # Simple model
            model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 2)
            )
            
            runner = TorchCVRunner(
                model=model,
                optimizer_class=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss()
            )
            
            # Test with simple split
            from trustcv.splitters.iid import KFoldMedical
            cv = KFoldMedical(n_splits=3)
            
            results = runner.run_cv(
                data=dataset,
                cv_splitter=cv,
                epochs=1
            )
            
            assert 'mean_accuracy' in results
            assert 'std_accuracy' in results
            assert 'fold_scores' in results
            assert len(results['fold_scores']) == 3
            
        except ImportError:
            pytest.skip("PyTorch not installed")


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
        assert tensorflow_adapter.batch_size == 16
    
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
    
    def test_tf_cv_runner(self, sample_data):
        """Test KerasCVRunner"""
        pytest.importorskip("tensorflow")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            X, y = sample_data
            
            # Simple model
            model = keras.Sequential([
                keras.layers.Dense(5, activation='relu', input_shape=(10,)),
                keras.layers.Dense(2, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            runner = KerasCVRunner(model=model)
            
            # Test with simple split
            from trustcv.splitters.iid import KFoldMedical
            cv = KFoldMedical(n_splits=3)
            
            results = runner.run_cv(
                X=X, y=y,
                cv_splitter=cv,
                epochs=1,
                verbose=0
            )
            
            assert 'mean_accuracy' in results
            assert 'std_accuracy' in results
            assert 'fold_scores' in results
            assert len(results['fold_scores']) == 3
            
        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestMONAIIntegration:
    """Test MONAI adapter and runner"""
    
    @pytest.fixture
    def sample_3d_data(self):
        """Create sample 3D medical imaging data"""
        # Simulate 3D medical images (10 samples, 1 channel, 32x32x32)
        images = np.random.randn(10, 1, 32, 32, 32).astype(np.float32)
        labels = np.random.randint(0, 2, 10)
        return images, labels
    
    @pytest.fixture
    def monai_adapter(self):
        """Create MONAI adapter instance"""
        return MONAIAdapter(batch_size=2)
    
    def test_monai_adapter_init(self, monai_adapter):
        """Test MONAI adapter initialization"""
        assert monai_adapter.batch_size == 2
        assert hasattr(monai_adapter, 'device')
    
    def test_monai_data_splits(self, monai_adapter, sample_3d_data):
        """Test creating MONAI data loaders"""
        pytest.importorskip("monai")
        pytest.importorskip("torch")
        
        images, labels = sample_3d_data
        train_idx = np.arange(8)
        val_idx = np.arange(8, 10)
        
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
            
            assert len(train_loader.dataset) == 8
            assert len(val_loader.dataset) == 2
            
        except ImportError:
            pytest.skip("MONAI not installed")
    
    def test_monai_cv_runner(self, sample_3d_data):
        """Test MONAICVRunner"""
        pytest.importorskip("monai")
        pytest.importorskip("torch")
        
        try:
            import torch
            import torch.nn as nn
            from monai.networks.nets import BasicUNet
            from monai.data import Dataset
            
            images, labels = sample_3d_data
            
            # Create MONAI dataset
            data_dicts = [
                {"image": images[i], "label": labels[i]} 
                for i in range(len(images))
            ]
            
            dataset = Dataset(data=data_dicts)
            
            # Simple 3D model
            model = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                features=(8, 16, 32)
            )
            
            runner = MONAICVRunner(
                model=model,
                optimizer_class=torch.optim.Adam,
                loss_function=nn.CrossEntropyLoss()
            )
            
            # Test with simple split
            from trustcv.splitters.iid import KFoldMedical
            cv = KFoldMedical(n_splits=2)
            
            results = runner.run_cv(
                data=dataset,
                cv_splitter=cv,
                epochs=1
            )
            
            assert 'mean_dice' in results or 'mean_accuracy' in results
            assert 'fold_scores' in results
            assert len(results['fold_scores']) == 2
            
        except ImportError:
            pytest.skip("MONAI not installed")


class TestUniversalCVRunner:
    """Test UniversalCVRunner with automatic framework detection"""
    
    def test_detect_sklearn(self):
        """Test detection of scikit-learn data"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        runner = UniversalCVRunner()
        framework = runner.detect_framework(X)
        
        assert framework == 'sklearn'
    
    def test_detect_pytorch(self):
        """Test detection of PyTorch data"""
        pytest.importorskip("torch")
        
        try:
            import torch
            from torch.utils.data import TensorDataset
            
            X = torch.randn(100, 10)
            y = torch.randint(0, 2, (100,))
            dataset = TensorDataset(X, y)
            
            runner = UniversalCVRunner()
            framework = runner.detect_framework(dataset)
            
            assert framework == 'pytorch'
            
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_detect_tensorflow(self):
        """Test detection of TensorFlow data"""
        pytest.importorskip("tensorflow")
        
        try:
            import tensorflow as tf
            
            dataset = tf.data.Dataset.from_tensor_slices(
                (np.random.randn(100, 10), np.random.randint(0, 2, 100))
            )
            
            runner = UniversalCVRunner()
            framework = runner.detect_framework(dataset)
            
            assert framework == 'tensorflow'
            
        except ImportError:
            pytest.skip("TensorFlow not installed")
    
    def test_run_cv_auto_detect(self):
        """Test running CV with automatic framework detection"""
        from trustcv.splitters.iid import KFoldMedical
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        runner = UniversalCVRunner()
        cv = KFoldMedical(n_splits=3)
        
        # Should auto-detect sklearn and run
        results = runner.run_cv(
            data=X,
            labels=y,
            cv_splitter=cv
        )
        
        assert results is not None
        assert 'framework' in results
        assert results['framework'] == 'sklearn'


class TestCallbacks:
    """Test callback system"""
    
    def test_early_stopping(self):
        """Test EarlyStopping callback"""
        callback = EarlyStopping(patience=3, min_delta=0.001)
        
        # Simulate training with improving loss
        assert not callback.on_epoch_end(1, {'val_loss': 0.5})
        assert not callback.on_epoch_end(2, {'val_loss': 0.4})
        assert not callback.on_epoch_end(3, {'val_loss': 0.3})
        
        # Simulate no improvement
        assert not callback.on_epoch_end(4, {'val_loss': 0.3})
        assert not callback.on_epoch_end(5, {'val_loss': 0.3})
        assert not callback.on_epoch_end(6, {'val_loss': 0.3})
        assert callback.on_epoch_end(7, {'val_loss': 0.3})  # Should stop
    
    def test_model_checkpoint(self):
        """Test ModelCheckpoint callback"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = ModelCheckpoint(
                filepath=os.path.join(tmpdir, 'model_{epoch}.pkl'),
                monitor='val_accuracy',
                mode='max'
            )
            
            # Mock model
            model = {'weights': np.random.randn(10, 10)}
            
            # Should save (first model)
            callback.on_epoch_end(1, {'val_accuracy': 0.8}, model)
            assert os.path.exists(os.path.join(tmpdir, 'model_1.pkl'))
            
            # Should not save (worse)
            callback.on_epoch_end(2, {'val_accuracy': 0.7}, model)
            
            # Should save (better)
            callback.on_epoch_end(3, {'val_accuracy': 0.9}, model)
            assert os.path.exists(os.path.join(tmpdir, 'model_3.pkl'))


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