"""LIF system: archetype definition and step function.

The LIFArchetype Protocol declares exactly which components step_LIF needs.
Any concrete type that has `.data: NetworkData` and `.params: NetworkParams`
satisfies the constraint — no explicit inheritance required.
"""

from typing import Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float, Bool
import jax.numpy as jnp

from .net_data import NetworkData, NetworkParams


# ── Archetype ─────────────────────────────────────────────────────────────────

class LIFArchetype(Protocol):
    """Required components for LIF neuron dynamics."""
    data:   NetworkData
    params: NetworkParams


T_LIF = TypeVar("T_LIF", bound=LIFArchetype)


# ── System ────────────────────────────────────────────────────────────────────

def step_LIF(
    net: T_LIF,
    perceptions: Float[Array, "N_perceptors"],
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
    v = net.data.v

    # LIF decay
    v = net.params.v_rest + (v - net.params.v_rest) * net.params.LIF_factor

    # Inject perceptor inputs
    v = v.at[net.data.perceptors].add(perceptions)

    # Threshold → binary spikes
    spikes: Bool[Array, "N_neurons"] = v >= net.params.v_threshold

    # Reset spiked neurons
    v = jnp.where(spikes, net.params.v_rest, v)

    # Propagate through weighted connections
    v = v.at[net.data.forward_connections].add(net.data.weights * spikes[:, None])

    return eqx.tree_at(lambda n: n.data.v, net, v)
