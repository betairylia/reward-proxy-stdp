from typing import TypeVar

from jaxtyping import Array, Float, Int
import jax
import jax.numpy as jnp

from ..net_data import NetworkData, NetworkParams
from ..net_exec import T_LIF


T = TypeVar("T")


def _build_essentials(
    N_neurons: int,
    N_perceptors: int,
    N_effectors: int,
) -> tuple[Float[Array, "N_neurons"], Int[Array, "N_perceptors"], Int[Array, "N_effectors"]]:
    """Return (v, perceptors, effectors) initialised to rest."""
    v = jnp.zeros(N_neurons)
    perceptors = jnp.arange(N_perceptors, dtype=jnp.int32)
    effectors  = jnp.arange(N_neurons - N_effectors, N_neurons, dtype=jnp.int32)
    return v, perceptors, effectors


def init_watts_strogatz(
    key:          jax.Array,
    net_type:     type[T_LIF],
    N_neurons:    int,
    degree:       int,
    N_perceptors: int,
    N_effectors:  int,
    beta:         float = 0.3,
    params:       NetworkParams | None = None,
) -> T_LIF:
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
    net_type : type[T_LIF]
        Concrete entity type to instantiate (e.g. LIFNet).
        Must accept ``data=NetworkData(...)`` and ``params=NetworkParams(...)``.
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
    if params is None:
        params = NetworkParams()

    v, perceptors, effectors = _build_essentials(N_neurons, N_perceptors, N_effectors)

    # --- Ring lattice ---
    offsets    = jnp.arange(1, degree + 1)
    neuron_ids = jnp.arange(N_neurons)
    ring = (neuron_ids[:, None] + offsets[None, :]) % N_neurons

    # --- Stochastic rewiring ---
    key_rewire, key_target, key_weights = jax.random.split(key, 3)

    rewire_mask     = jax.random.bernoulli(key_rewire, p=beta, shape=(N_neurons, degree))
    random_targets  = jax.random.randint(key_target, shape=(N_neurons, degree), minval=0, maxval=N_neurons)

    # Avoid self-connections
    self_mask      = random_targets == neuron_ids[:, None]
    random_targets = jnp.where(self_mask, (random_targets + 1) % N_neurons, random_targets)

    forward_connections = jnp.where(rewire_mask, random_targets, ring).astype(jnp.int32)

    weights = jax.random.uniform(key_weights, shape=(N_neurons, degree))

    data = NetworkData(
        v=v,
        forward_connections=forward_connections,
        weights=weights,
        perceptors=perceptors,
        effectors=effectors,
    )

    return net_type(data=data, params=params)
