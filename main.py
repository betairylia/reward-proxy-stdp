"""Main entry point for reward-proxy-stdp."""
import jax
import jax.numpy as jnp

from nets import step_LIF
from nets.factory import create_network
from datasets import load_mnist


def main():
    print("Reward-Proxy STDP - Spiking Neural Network")
    print("=" * 50)

    print("\n1. Loading MNIST dataset...")
    test_data = load_mnist(train=False)
    print(f"   ✓ {len(test_data)} test samples")

    print("\n2. Initializing network...")
    net = create_network("lif", jax.random.PRNGKey(42))
    print(f"   ✓ {net.data.v.shape[0]} neurons")

    print("\n3. Running simulation...")
    image, label = test_data[0]

    for _ in range(50):
        net = step_LIF(net, image * 2.0)

    predicted = jnp.argmax(net.data.v[net.data.effectors])
    print(f"   ✓ Predicted: {predicted}, Actual: {label}")

    print("\n" + "=" * 50)
    print("Run 'uv run python example.py' for more examples")


if __name__ == "__main__":
    main()
