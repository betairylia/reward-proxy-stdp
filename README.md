# Reward-Proxy STDP

A spiking neural network implementation using Leaky Integrate-and-Fire (LIF) neurons with small-world connectivity.

## Features

- **LIF Neurons**: Biologically-inspired spiking neurons
- **Small-World Networks**: Watts-Strogatz initialization
- **JAX Backend**: Fast, GPU-accelerated computation
- **Dataset Integration**: Easy loading of PyTorch datasets

## Quick Start

```bash
# Install dependencies
uv sync

# Run basic demo
python main.py

# Run examples
python example.py
```

## Usage

### Load a Dataset

```python
from datasets import load_mnist, load_vision_dataset

# Load MNIST
train_data = load_mnist(train=True)
test_data = load_mnist(train=False)

# Or any other dataset
cifar_data = load_vision_dataset('cifar10', train=True)
```

### Initialize Network

```python
from nets.initialization import init_watts_strogatz
import jax

key = jax.random.PRNGKey(42)
net = init_watts_strogatz(
    key=key,
    N_neurons=1000,      # Total neurons
    degree=20,           # Connections per neuron
    N_perceptors=784,    # Input neurons (28x28 for MNIST)
    N_effectors=10,      # Output neurons (10 classes)
    beta=0.3             # Rewiring probability
)
```

### Run Simulation

```python
from nets import step_LIF

# Get a sample
image, label = test_data[0]

# Run for 50 timesteps
for t in range(50):
    net = step_LIF(net, image * 2.0)

# Check output
import jax.numpy as jnp
predicted = jnp.argmax(net.v[net.effectors])
```

## Supported Datasets

- **MNIST**: Handwritten digits (784 inputs, 10 classes)
- **Fashion-MNIST**: Clothing items (784 inputs, 10 classes)
- **CIFAR-10**: Color images (3072 inputs, 10 classes)
- **CIFAR-100**: Color images (3072 inputs, 100 classes)
- **KMNIST**: Japanese characters (784 inputs, 10 classes)
- **Custom**: Any PyTorch dataset via `load_custom_dataset()`

## Project Structure

```
reward-proxy-stdp/
├── nets/
│   ├── net_data.py          # Network data structures
│   ├── net_exec.py          # LIF neuron dynamics
│   └── initialization/
│       └── small_world.py   # Network initialization
├── datasets/
│   └��─ pytorch_wrapper.py   # Dataset loading
├── main.py                  # Quick demo
└── example.py               # Detailed examples
```

## Network Parameters

Customize network behavior:

```python
from nets import NetworkParams

params = NetworkParams(
    v_threshold=1.0,    # Spike threshold
    v_rest=0.0,         # Resting potential
    LIF_factor=0.85     # Membrane decay factor
)

net = init_watts_strogatz(..., params=params)
```

## Dependencies

- JAX >= 0.4
- jaxtyping >= 0.3.9
- PyTorch >= 2.0
- torchvision >= 0.15

## License

MIT
