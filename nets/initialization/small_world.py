from jaxtyping import Array, Float, Int
import jax
import jax.numpy as jnp

from ..net_data import NetworkData, NetworkParams


def init_essentials(
    N_neurons: int,
    N_perceptors: int,
    N_effectors: int,
    params: NetworkParams | None = None,
) -> tuple[NetworkParams, Float[Array, "N_neurons"], Int[Array, "N_perceptors"], Int[Array, "N_effectors"]]:
    """Common initialization shared across wiring strategies.

    Returns (params, v, perceptors, effectors).
    Perceptors are the first *N_perceptors* neurons,
    effectors are the last *N_effectors* neurons.
    """
    if params is None:
        params = NetworkParams()

    v = jnp.full(N_neurons, params.v_rest)
    perceptors = jnp.arange(N_perceptors, dtype=jnp.int32)
    effectors = jnp.arange(N_neurons - N_effectors, N_neurons, dtype=jnp.int32)

    return params, v, perceptors, effectors


def init_watts_strogatz(
    key: jax.Array,
    N_neurons: int,
    degree: int,
    N_perceptors: int,
    N_effectors: int,
    beta: float = 0.3,
    params: NetworkParams | None = None,
) -> NetworkData:
    """Stochastic small-world initialisation (Watts-Strogatz style).

    1. Build a ring-lattice where each neuron connects to its *degree*
       nearest neighbours (forward direction on the ring).
    2. For every edge, with probability *beta*, rewire the target to a
       uniformly random neuron (avoiding self-connections).
    3. Weights are drawn from U(0, 1).

    Parameters
    ----------
    key : jax.Array
        PRNG key for reproducibility.
    N_neurons : int
        Total number of neurons in the network.
    degree : int
        Number of outgoing connections per neuron.
    N_perceptors : int
        Number of input (perceptor) neurons.
    N_effectors : int
        Number of output (effector) neurons.
    beta : float
        Rewiring probability (0 = pure ring, 1 = fully random).
    params : NetworkParams | None
        Optional override for network hyper-parameters.
    """
    net_params, v, perceptors, effectors = init_essentials(
        N_neurons, N_perceptors, N_effectors, params
    )

    # --- Ring lattice ---
    # Each neuron i connects to i+1, i+2, ..., i+degree (mod N_neurons)
    offsets = jnp.arange(1, degree + 1)                          # (degree,)
    neuron_ids = jnp.arange(N_neurons)                            # (N_neurons,)
    ring = (neuron_ids[:, None] + offsets[None, :]) % N_neurons   # (N_neurons, degree)

    # --- Stochastic rewiring ---
    key_rewire, key_target, key_weights = jax.random.split(key, 3)

    # Decide which edges to rewire
    rewire_mask = jax.random.bernoulli(
        key_rewire, p=beta, shape=(N_neurons, degree)
    )

    # Draw random targets in [0, N_neurons - 1]
    random_targets = jax.random.randint(
        key_target, shape=(N_neurons, degree), minval=0, maxval=N_neurons
    )

    # Avoid self-connections: if random target == neuron id, shift by 1
    self_mask = random_targets == neuron_ids[:, None]
    random_targets = jnp.where(self_mask, (random_targets + 1) % N_neurons, random_targets)

    forward_connections = jnp.where(rewire_mask, random_targets, ring).astype(jnp.int32)

    # --- Random weights ---
    weights = jax.random.uniform(key_weights, shape=(N_neurons, degree))

    return NetworkData(
        params=net_params,
        v=v,
        forward_connections=forward_connections,
        weights=weights,
        perceptors=perceptors,
        effectors=effectors,
    )
