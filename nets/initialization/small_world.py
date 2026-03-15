import equinox as eqx
from jaxtyping import Array
import jax
import jax.numpy as jnp

from ..net_exec import T_LIF


def init_watts_strogatz(
    key:  jax.Array,
    net:  T_LIF,
    beta: float = 0.3,
) -> T_LIF:
    """Fill a network's topology using the Watts-Strogatz small-world model.

    1. Build a ring-lattice where each neuron connects to its *degree*
       nearest neighbours (forward direction on the ring).
    2. For every edge, with probability *beta*, rewire the target to a
       uniformly random neuron (avoiding self-connections).
    3. Weights are drawn from U(0, 1).

    Parameters
    ----------
    key : jax.Array
        PRNG key for reproducibility.
    net : T_LIF
        Network shell produced by ``create_network``.
        Dimensions (N_neurons, degree) are read from its existing arrays.
    beta : float
        Rewiring probability (0 = pure ring, 1 = fully random).

    Returns
    -------
    T_LIF
        Same entity type as input, with ``data.forward_connections`` and
        ``data.weights`` filled in.
    """
    N      = net.data.v.shape[0]
    degree = net.data.weights.shape[1]

    # --- Ring lattice ---
    offsets    = jnp.arange(1, degree + 1)
    neuron_ids = jnp.arange(N)
    ring = (neuron_ids[:, None] + offsets[None, :]) % N

    # --- Stochastic rewiring ---
    key_rewire, key_target, key_weights = jax.random.split(key, 3)

    rewire_mask    = jax.random.bernoulli(key_rewire, p=beta, shape=(N, degree))
    random_targets = jax.random.randint(key_target, shape=(N, degree), minval=0, maxval=N)

    # Avoid self-connections
    self_mask      = random_targets == neuron_ids[:, None]
    random_targets = jnp.where(self_mask, (random_targets + 1) % N, random_targets)

    forward_connections = jnp.where(rewire_mask, random_targets, ring).astype(jnp.int32)
    weights             = jax.random.uniform(key_weights, shape=(N, degree))

    return eqx.tree_at(
        lambda n: (n.data.forward_connections, n.data.weights),
        net,
        (forward_connections, weights),
    )
