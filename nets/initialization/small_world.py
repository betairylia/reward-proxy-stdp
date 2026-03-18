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

    Structural constraints enforced:
    - Perceptors (first ``N_p`` indices) receive no incoming connections —
      all targets are restricted to ``[N_p, N)``.
    - Effectors (last ``N_e`` indices) have no outgoing connections —
      their weight rows are zeroed.

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
    N         = net.data.v.shape[0]
    degree    = net.data.weights.shape[1]
    N_p       = net.data.perceptors.shape[0]   # perceptors occupy [0, N_p)
    N_e       = net.data.effectors.shape[0]    # effectors occupy [N-N_e, N)
    N_targets = N - N_p                        # valid targets: [N_p, N)

    # --- Ring lattice (targets restricted to [N_p, N)) ---
    offsets    = jnp.arange(1, degree + 1)
    neuron_ids = jnp.arange(N)
    ring = N_p + (neuron_ids[:, None] + offsets[None, :]) % N_targets

    # --- Stochastic rewiring ---
    key_rewire, key_target, key_weights = jax.random.split(key, 3)

    rewire_mask    = jax.random.bernoulli(key_rewire, p=beta, shape=(N, degree))
    random_targets = jax.random.randint(key_target, shape=(N, degree), minval=N_p, maxval=N)

    # Avoid self-connections
    self_mask      = random_targets == neuron_ids[:, None]
    random_targets = jnp.where(self_mask, N_p + (random_targets - N_p + 1) % N_targets, random_targets)

    forward_connections = jnp.where(rewire_mask, random_targets, ring).astype(jnp.int32)
    weights             = jax.random.uniform(key_weights, shape=(N, degree))

    # Effectors have no outgoing connections: zero their weights
    is_effector = (neuron_ids >= N - N_e)[:, None]
    weights = jnp.where(is_effector, 0.0, weights)

    return eqx.tree_at(
        lambda n: (n.data.forward_connections, n.data.weights),
        net,
        (forward_connections, weights),
    )
