"""Example: Using datasets with the spiking neural network.

Demonstrates loading various datasets and running them through
the LIF spiking neural network.
"""
import jax
import jax.numpy as jnp

from nets import NetworkParams, step_LIF
from nets.initialization import init_watts_strogatz
from datasets import load_mnist, load_vision_dataset


def run_network(net, input_data, n_timesteps=50, scale=2.0):
    """Run the spiking network for multiple timesteps.
    
    Parameters
    ----------
    net : NetworkData
        Initialized network
    input_data : jnp.ndarray
        Input vector for perceptors
    n_timesteps : int
        Number of simulation steps
    scale : float
        Input scaling factor (higher = more spikes)
    
    Returns
    -------
    NetworkData
        Network state after simulation
    """
    perception_input = input_data * scale
    
    for t in range(n_timesteps):
        net = step_LIF(net, perception_input)
    
    return net


def example_mnist():
    """Basic example using MNIST dataset."""
    print("="*60)
    print("MNIST Example")
    print("="*60)
    
    # Load MNIST
    print("\nLoading MNIST dataset...")
    train_data = load_mnist(train=True)
    test_data = load_mnist(train=False)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Input shape: {train_data.input_shape}")
    print(f"Classes: {train_data.num_classes}")
    
    # Initialize network
    print("\nInitializing network...")
    key = jax.random.PRNGKey(42)
    net = init_watts_strogatz(
        key=key,
        N_neurons=1000,
        degree=20,
        N_perceptors=784,  # 28x28 flattened
        N_effectors=10,    # 10 digit classes
        beta=0.3
    )
    print(f"Network: {net.v.shape[0]} neurons, {net.weights.shape[1]} connections/neuron")
    
    # Process a test sample
    image, label = test_data[0]
    print(f"\nProcessing test sample: digit '{label}'")
    
    net = run_network(net, image, n_timesteps=50, scale=2.0)
    
    # Check output
    output_potentials = net.v[net.effectors]
    predicted = jnp.argmax(output_potentials)
    
    print(f"\nResults:")
    print(f"  Predicted: {predicted}")
    print(f"  Actual: {label}")
    print(f"  Output potentials: {output_potentials}")
    print(f"  Match: {'✓' if predicted == label else '✗'}")


def example_other_datasets():
    """Show how to load other datasets."""
    print("\n" + "="*60)
    print("Other Datasets")
    print("="*60)
    
    datasets_info = {
        'fashion_mnist': ('Fashion-MNIST', 784, 10),
        'cifar10': ('CIFAR-10', 3072, 10),
        'kmnist': ('KMNIST', 784, 10),
    }
    
    for name, (display_name, expected_inputs, expected_classes) in datasets_info.items():
        try:
            print(f"\n{display_name}:")
            data = load_vision_dataset(name, train=False, download=True)
            print(f"  ✓ Loaded {len(data)} samples")
            print(f"  Input shape: {data.input_shape} ({data.input_shape[0]} inputs)")
            print(f"  Classes: {data.num_classes}")
            
            # Verify expected dimensions
            assert data.input_shape[0] == expected_inputs
            assert data.num_classes == expected_classes
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def example_custom_params():
    """Example with custom network parameters."""
    print("\n" + "="*60)
    print("Custom Network Parameters")
    print("="*60)
    
    # Load data
    data = load_mnist(train=False)
    
    # Custom parameters
    custom_params = NetworkParams(
        v_threshold=1.5,      # Higher threshold
        v_rest=0.0,
        LIF_factor=0.85       # Different decay
    )
    
    print(f"\nCustom parameters:")
    print(f"  Threshold: {custom_params.v_threshold}")
    print(f"  Rest potential: {custom_params.v_rest}")
    print(f"  LIF factor: {custom_params.LIF_factor}")
    
    # Initialize with custom params
    key = jax.random.PRNGKey(123)
    net = init_watts_strogatz(
        key=key,
        N_neurons=1000,
        degree=20,
        N_perceptors=784,
        N_effectors=10,
        beta=0.3,
        params=custom_params
    )
    
    print(f"\nNetwork initialized with custom parameters")
    print(f"  Threshold: {net.params.v_threshold}")
    print(f"  LIF factor: {net.params.LIF_factor}")


def main():
    """Run all examples."""
    print("\nSpiking Neural Network - Dataset Examples\n")
    
    example_mnist()
    example_other_datasets()
    example_custom_params()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nQuick reference:")
    print("  from datasets import load_mnist, load_vision_dataset")
    print("  from nets import step_LIF")
    print("  from nets.initialization import init_watts_strogatz")


if __name__ == "__main__":
    main()
