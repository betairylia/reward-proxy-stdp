"""General PyTorch dataset wrapper for spiking neural networks.

Provides a unified interface to load any PyTorch dataset and convert
it to JAX arrays compatible with spiking neural networks.
"""
import jax.numpy as jnp
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Callable
import numpy as np


class DatasetWrapper:
    """Wrapper for PyTorch datasets that converts to JAX arrays.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        PyTorch dataset instance
    flatten : bool
        If True, flatten image data (e.g., (28, 28) -> (784,))
    normalize : bool
        If True, ensure data is in [0, 1] range
    transform_fn : Callable, optional
        Custom transformation: fn(data, label) -> (data, label)
    
    Attributes
    ----------
    images : jnp.ndarray
        All data as JAX array, shape (N, ...)
    labels : jnp.ndarray
        All labels as JAX array, shape (N,)
    input_shape : tuple
        Shape of a single input sample
    num_classes : int or None
        Number of unique classes
    """
    
    def __init__(
        self,
        dataset: Dataset,
        flatten: bool = True,
        normalize: bool = True,
        transform_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.flatten = flatten
        self.normalize = normalize
        self.transform_fn = transform_fn
        self._load_data()
    
    def _load_data(self):
        """Load all data from PyTorch dataset into JAX arrays."""
        all_data = []
        all_labels = []
        
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            
            # Handle different dataset formats
            if isinstance(item, (tuple, list)):
                data, label = item[0], item[1]
            else:
                data, label = item, -1
            
            # Convert to numpy
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            elif not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Apply custom transform
            if self.transform_fn is not None:
                data, label = self.transform_fn(data, label)
            
            all_data.append(data)
            all_labels.append(label)
        
        # Stack into arrays
        all_data = np.stack(all_data)
        all_labels = np.array(all_labels)
        
        # Flatten if requested
        if self.flatten and len(all_data.shape) > 2:
            all_data = all_data.reshape(len(all_data), -1)
        
        # Normalize if requested
        if self.normalize:
            data_min, data_max = all_data.min(), all_data.max()
            if data_max > data_min:
                all_data = (all_data - data_min) / (data_max - data_min)
        
        # Convert to JAX arrays
        self.images = jnp.array(all_data, dtype=jnp.float32)
        self.labels = jnp.array(all_labels, dtype=jnp.int32)
        
        # Store metadata
        self.input_shape = self.images.shape[1:]
        self.num_classes = len(np.unique(all_labels)) if all_labels[0] != -1 else None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[jnp.ndarray, int]:
        return self.images[idx], int(self.labels[idx])
    
    def get_batch(self, indices) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get multiple samples by indices."""
        return self.images[indices], self.labels[indices]


def load_vision_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    train: bool = True,
    flatten: bool = True,
    normalize: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
) -> DatasetWrapper:
    """Load common vision datasets from torchvision.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name: 'mnist', 'fashion_mnist', 'cifar10', 'cifar100',
        'kmnist', 'emnist', 'svhn', 'stl10'
    data_dir : str
        Root directory for dataset storage
    train : bool
        If True, load training set; otherwise test set
    flatten : bool
        If True, flatten images to 1D vectors
    normalize : bool
        If True, normalize to [0, 1]
    download : bool
        If True, download dataset if not found
    transform : transforms.Compose, optional
        Custom PyTorch transforms
    
    Returns
    -------
    DatasetWrapper
        Wrapped dataset with JAX arrays
    """
    dataset_name = dataset_name.lower().replace('-', '_')
    data_path = Path(data_dir) / dataset_name
    data_path.mkdir(parents=True, exist_ok=True)
    
    if transform is None:
        transform = transforms.ToTensor()
    
    # Map dataset names to torchvision classes
    dataset_map = {
        'mnist': datasets.MNIST,
        'fashion_mnist': datasets.FashionMNIST,
        'fashionmnist': datasets.FashionMNIST,
        'kmnist': datasets.KMNIST,
        'emnist': lambda **kwargs: datasets.EMNIST(split='balanced', **kwargs),
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'svhn': lambda **kwargs: datasets.SVHN(split='train' if kwargs.pop('train') else 'test', **kwargs),
        'stl10': lambda **kwargs: datasets.STL10(split='train' if kwargs.pop('train') else 'test', **kwargs),
    }
    
    if dataset_name not in dataset_map:
        available = ', '.join(sorted(dataset_map.keys()))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    dataset_class = dataset_map[dataset_name]
    
    try:
        dataset = dataset_class(
            root=str(data_path),
            train=train,
            download=download,
            transform=transform
        )
    except TypeError:
        # Some datasets don't have 'train' parameter
        dataset = dataset_class(
            root=str(data_path),
            download=download,
            transform=transform
        )
    
    return DatasetWrapper(dataset, flatten=flatten, normalize=normalize)


def load_custom_dataset(
    dataset: Dataset,
    flatten: bool = True,
    normalize: bool = True,
    transform_fn: Optional[Callable] = None
) -> DatasetWrapper:
    """Load any custom PyTorch dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Any PyTorch dataset instance
    flatten : bool
        If True, flatten data to 1D
    normalize : bool
        If True, normalize to [0, 1]
    transform_fn : Callable, optional
        Custom transformation function
    
    Returns
    -------
    DatasetWrapper
        Wrapped dataset with JAX arrays
    """
    return DatasetWrapper(dataset, flatten=flatten, normalize=normalize, transform_fn=transform_fn)


# Convenience functions for common datasets
def load_mnist(data_dir: str = "./data", train: bool = True, **kwargs) -> DatasetWrapper:
    """Load MNIST dataset (28x28 grayscale digits, 10 classes)."""
    return load_vision_dataset('mnist', data_dir=data_dir, train=train, **kwargs)


def load_fashion_mnist(data_dir: str = "./data", train: bool = True, **kwargs) -> DatasetWrapper:
    """Load Fashion-MNIST dataset (28x28 grayscale clothing, 10 classes)."""
    return load_vision_dataset('fashion_mnist', data_dir=data_dir, train=train, **kwargs)


def load_cifar10(data_dir: str = "./data", train: bool = True, **kwargs) -> DatasetWrapper:
    """Load CIFAR-10 dataset (32x32 color images, 10 classes)."""
    return load_vision_dataset('cifar10', data_dir=data_dir, train=train, **kwargs)


def load_cifar100(data_dir: str = "./data", train: bool = True, **kwargs) -> DatasetWrapper:
    """Load CIFAR-100 dataset (32x32 color images, 100 classes)."""
    return load_vision_dataset('cifar100', data_dir=data_dir, train=train, **kwargs)
