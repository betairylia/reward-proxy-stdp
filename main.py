"""Main entry point for reward-proxy-stdp.

A spiking neural network implementation using LIF neurons with
small-world connectivity and STDP learning.
"""
import jax
import jax.numpy as jnp

from nets import step_LIF
from nets.initialization import init_watts_strogatz
from datasets import load_mnist


def main():
    """Quick demo of the spiking neural network."""
    print("Reward-Proxy STDP - Spiking Neural Network")
    print("="*50)
    
    # Load dataset
    print("\n1. Loading MNIST dataset...")
    test_data = load_mnist(train=False)
    print(f"   ✓ {len(test_data)} test samples")
    
    # Initialize network
    print("\n2. Initializing network...")
    key = jax.random.PRNGKey(42)
    net = init_watts_strogatz(
        key=key,
        N_neurons=1000,
        degree=20,
        N_perceptors=784,
        N_effectors=10,
        beta=0.3
    )
    print(f"   ✓ {net.v.shape[0]} neurons")
    
    # Run simulation
    print("\n3. Running simulation...")
    image, label = test_data[0]
    
    for t in range(50):
        net = step_LIF(net, image * 2.0)
    
    predicted = jnp.argmax(net.v[net.effectors])
    print(f"   ✓ Predicted: {predicted}, Actual: {label}")
    
    print("\n" + "="*50)
    print("Run 'python example.py' for more examples")


if __name__ == "__main__":
    main()
