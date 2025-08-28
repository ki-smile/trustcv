"""
Base classes for framework-agnostic cross-validation

These abstractions allow trustcv to work with any ML framework while
promoting best practices in model evaluation and regulatory compliance.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union, Tuple, Dict, Any, List
import numpy as np
from dataclasses import dataclass, field
import warnings


@dataclass
class CVResults:
    """
    Standardized results container for cross-validation
    
    Attributes:
        scores: Performance scores for each fold
        models: Trained models from each fold (optional)
        predictions: Predictions on validation sets
        indices: Train/test indices for each fold
        metadata: Additional framework-specific information
    """
    scores: List[Dict[str, float]]
    models: Optional[List[Any]] = None
    predictions: Optional[List[np.ndarray]] = None
    indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mean_score(self) -> Dict[str, float]:
        """Calculate mean scores across folds"""
        if not self.scores:
            return {}
        
        keys = self.scores[0].keys()
        return {
            key: np.mean([fold[key] for fold in self.scores])
            for key in keys
        }
    
    @property
    def std_score(self) -> Dict[str, float]:
        """Calculate standard deviation of scores across folds"""
        if not self.scores:
            return {}
        
        keys = self.scores[0].keys()
        return {
            key: np.std([fold[key] for fold in self.scores])
            for key in keys
        }
    
    def summary(self) -> str:
        """Generate a summary of CV results"""
        mean = self.mean_score
        std = self.std_score
        
        summary_lines = ["Cross-Validation Results Summary:"]
        summary_lines.append("-" * 40)
        
        for metric in mean.keys():
            summary_lines.append(
                f"{metric}: {mean[metric]:.4f} (+/- {std[metric]:.4f})"
            )
        
        if self.metadata:
            summary_lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                summary_lines.append(f"  {key}: {value}")
        
        return "\n".join(summary_lines)


class CVSplitter(ABC):
    """
    Abstract base class for all CV splitters
    
    This class defines the interface that all cross-validation splitters
    must implement, ensuring compatibility across different frameworks.
    """
    
    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Get the number of splits/folds
        
        Parameters:
            X: Features array (optional for some splitters)
            y: Target array (optional for some splitters)
            groups: Group labels for grouped CV (optional)
            
        Returns:
            Number of cross-validation folds
        """
        pass
    
    @abstractmethod
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for cross-validation
        
        Parameters:
            X: Features array
            y: Target array (optional)
            groups: Group labels for grouped CV (optional)
            
        Yields:
            train_indices: Indices for training set
            test_indices: Indices for test/validation set
        """
        pass
    
    def validate_split(self, train_idx: np.ndarray, test_idx: np.ndarray, 
                      groups: Optional[np.ndarray] = None) -> Dict[str, bool]:
        """
        Validate that a split meets best practices
        
        Parameters:
            train_idx: Training indices
            test_idx: Test indices
            groups: Group labels (optional)
            
        Returns:
            Dictionary with validation results
        """
        validations = {
            'no_overlap': len(np.intersect1d(train_idx, test_idx)) == 0,
            'train_not_empty': len(train_idx) > 0,
            'test_not_empty': len(test_idx) > 0,
        }
        
        if groups is not None:
            train_groups = np.unique(groups[train_idx])
            test_groups = np.unique(groups[test_idx])
            validations['no_group_leakage'] = len(
                np.intersect1d(train_groups, test_groups)
            ) == 0
        
        return validations


class FrameworkAdapter(ABC):
    """
    Abstract adapter for framework-specific implementations
    
    This class provides the interface for adapting trustcv's splitting
    strategies to different ML frameworks' data handling and training loops.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize adapter with framework-specific parameters
        
        Parameters:
            **kwargs: Framework-specific configuration
        """
        self.config = kwargs
    
    @abstractmethod
    def create_data_splits(self, data: Any, train_idx: np.ndarray, 
                          val_idx: np.ndarray) -> Tuple[Any, Any]:
        """
        Create framework-specific data loaders/datasets from indices
        
        Parameters:
            data: Original dataset (format depends on framework)
            train_idx: Indices for training data
            val_idx: Indices for validation data
            
        Returns:
            train_data: Framework-specific training data structure
            val_data: Framework-specific validation data structure
        """
        pass
    
    @abstractmethod
    def train_epoch(self, model: Any, train_data: Any, 
                   optimizer: Optional[Any] = None, **kwargs) -> Dict[str, float]:
        """
        Train model for one epoch
        
        Parameters:
            model: Framework-specific model
            train_data: Training data in framework-specific format
            optimizer: Optimizer (optional, framework-specific)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, val_data: Any, 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model on validation data
        
        Parameters:
            model: Framework-specific model
            val_data: Validation data in framework-specific format
            metrics: List of metrics to compute (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def clone_model(self, model: Any) -> Any:
        """
        Create a fresh copy of the model
        
        Parameters:
            model: Original model
            
        Returns:
            Cloned model with reset weights
        """
        # Default implementation - frameworks can override
        warnings.warn(
            "Using default model cloning which may not work for all frameworks. "
            "Consider implementing framework-specific cloning.",
            UserWarning
        )
        return model
    
    def get_predictions(self, model: Any, data: Any) -> np.ndarray:
        """
        Get predictions from model
        
        Parameters:
            model: Trained model
            data: Data to predict on
            
        Returns:
            Predictions as numpy array
        """
        raise NotImplementedError(
            "Prediction extraction must be implemented by framework adapter"
        )
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save model to disk
        
        Parameters:
            model: Model to save
            path: Path to save model
        """
        raise NotImplementedError(
            "Model saving must be implemented by framework adapter"
        )
    
    def load_model(self, path: str) -> Any:
        """
        Load model from disk
        
        Parameters:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        raise NotImplementedError(
            "Model loading must be implemented by framework adapter"
        )


class SklearnAdapter(FrameworkAdapter):
    """
    Adapter for scikit-learn models (backward compatibility)
    """
    
    def create_data_splits(self, data: Tuple[np.ndarray, np.ndarray], 
                          train_idx: np.ndarray, 
                          val_idx: np.ndarray) -> Tuple[Any, Any]:
        """Create train/validation splits for sklearn"""
        X, y = data
        return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])
    
    def train_epoch(self, model: Any, train_data: Tuple[np.ndarray, np.ndarray], 
                   optimizer: None = None, **kwargs) -> Dict[str, float]:
        """Train sklearn model (single fit call)"""
        X_train, y_train = train_data
        model.fit(X_train, y_train, **kwargs)
        
        # Return training score if model supports it
        train_metrics = {}
        if hasattr(model, 'score'):
            train_metrics['train_score'] = model.score(X_train, y_train)
        
        return train_metrics
    
    def evaluate(self, model: Any, val_data: Tuple[np.ndarray, np.ndarray], 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate sklearn model"""
        X_val, y_val = val_data
        
        eval_metrics = {}
        
        # Default score
        if hasattr(model, 'score'):
            eval_metrics['score'] = model.score(X_val, y_val)
        
        # Get predictions for additional metrics
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
            eval_metrics['predictions'] = y_pred
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val)
            eval_metrics['probabilities'] = y_proba
        
        return eval_metrics
    
    def clone_model(self, model: Any) -> Any:
        """Clone sklearn model"""
        from sklearn.base import clone
        return clone(model)
    
    def get_predictions(self, model: Any, data: Any) -> np.ndarray:
        """Get predictions from sklearn model"""
        if isinstance(data, tuple):
            X, _ = data
        else:
            X = data
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        elif hasattr(model, 'predict'):
            return model.predict(X)
        else:
            raise ValueError("Model doesn't have predict or predict_proba method")
    
    def save_model(self, model: Any, path: str) -> None:
        """Save sklearn model using joblib"""
        import joblib
        joblib.dump(model, path)
    
    def load_model(self, path: str) -> Any:
        """Load sklearn model using joblib"""
        import joblib
        return joblib.load(path)