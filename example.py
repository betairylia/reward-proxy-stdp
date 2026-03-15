"""Example: Using datasets with the spiking neural network."""
import jax
import jax.numpy as jnp
import equinox as eqx

from nets import NetworkData, NetworkParams, step_LIF, T_LIF
from nets.factory import create_network, register_component
from nets.initialization import init_watts_strogatz
from datasets import load_mnist, load_vision_dataset


class LIFNet(eqx.Module):
    data:   NetworkData
    params: NetworkParams


def run_network(net: T_LIF, input_data, n_timesteps=50, scale=2.0) -> T_LIF:
    """Run the spiking network for multiple timesteps."""
    perception_input = input_data * scale
    for _ in range(n_timesteps):
        net = step_LIF(net, perception_input)
    return net


def example_mnist():
    """Basic example using factory + MNIST."""
    print("=" * 60)
    print("MNIST Example")
    print("=" * 60)

    print("\nLoading MNIST dataset...")
    test_data = load_mnist(train=False)
    print(f"Test samples: {len(test_data)}")

    print("\nInitializing network...")
    net = create_network(LIFNet)
    net = init_watts_strogatz(jax.random.PRNGKey(42), net)
    print(f"Network: {net.data.v.shape[0]} neurons, "
          f"{net.data.weights.shape[1]} connections/neuron")

    image, label = test_data[0]
    print(f"\nProcessing test sample: digit '{label}'")
    net = run_network(net, image)

    output_potentials = net.data.v[net.data.effectors]
    predicted = jnp.argmax(output_potentials)
    print(f"\nResults:")
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {label}")
    print(f"  Match: {'✓' if predicted == label else '✗'}")


def example_other_datasets():
    """Show how to load other datasets."""
    print("\n" + "=" * 60)
    print("Other Datasets")
    print("=" * 60)

    datasets_info = {
        "fashion_mnist": ("Fashion-MNIST", 784, 10),
        "cifar10":       ("CIFAR-10",      3072, 10),
        "kmnist":        ("KMNIST",        784,  10),
    }

    for name, (display_name, expected_inputs, expected_classes) in datasets_info.items():
        try:
            print(f"\n{display_name}:")
            data = load_vision_dataset(name, train=False, download=True)
            print(f"  ✓ Loaded {len(data)} samples")
            print(f"  Input shape: {data.input_shape}")
            assert data.input_shape[0] == expected_inputs
            assert data.num_classes == expected_classes
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def example_custom_params():
    """Example with a custom entity type and overridden params."""
    print("\n" + "=" * 60)
    print("Custom Network Parameters")
    print("=" * 60)

    # Override just the params by passing a custom NetworkParams to the net
    net = create_network(LIFNet)
    net = eqx.tree_at(
        lambda n: n.params,
        net,
        NetworkParams(v_threshold=1.5, v_rest=0.0, LIF_factor=0.85),
    )
    net = init_watts_strogatz(jax.random.PRNGKey(123), net)

    print(f"  Threshold: {net.params.v_threshold}")
    print(f"  LIF factor: {net.params.LIF_factor}")
    print(f"  Network: {net.data.v.shape[0]} neurons")


def main():
    print("\nSpiking Neural Network - Examples\n")
    example_mnist()
    example_other_datasets()
    example_custom_params()
    print("\n" + "=" * 60)
    print("Examples completed!")


if __name__ == "__main__":
    main()
