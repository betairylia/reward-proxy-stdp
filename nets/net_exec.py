import equinox as eqx
from jaxtyping import Array, Float, Bool
import jax.numpy as jnp

from .archetypes import T_LIF


def step_LIF(
    net: T_LIF,
    perceptions: Float[Array, "N_perceptors"]
) -> T_LIF:
    """Single timestep of LIF neuron dynamics.

    Parameters
    ----------
    net : T_LIF
        Current network state — any type satisfying LIFArchetype.
    perceptions : Float[Array, "N_perceptors"]
        Input values for perceptor neurons.

    Returns
    -------
    T_LIF
        Updated network state, same concrete type as input.
    """
    # Start with current membrane potentials
    v = net.v

    # LIF Decay
    v = net.params.v_rest + (v - net.params.v_rest) * net.params.LIF_factor

    # Apply inputs to perceptor neurons
    v = v.at[net.perceptors].add(perceptions)

    # Mark binary spikes
    spikes: Bool[Array, "N_neurons"] = v >= net.params.v_threshold

    # Reset spiked neurons to rest potential
    v = jnp.where(spikes, net.params.v_rest, v)

    # Propagate spikes through connections
    spike_contributions = net.weights * spikes[:, None]
    v = v.at[net.forward_connections].add(spike_contributions)

    return eqx.tree_at(lambda n: n.v, net, v)
