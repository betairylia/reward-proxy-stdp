"""LIF system: archetype, spike generation, and leaky-integrate dynamics.

The LIF step is split into two stages:
  1. generate_spike  — threshold comparison, neuron reset, SpikeData update
  2. step_leaky_integrate — decay, perception injection, spike propagation

step_LIF composes both stages for convenience.

LIFArchetype uses structural subtyping (Protocol + TypeVar) — any concrete
type with the required fields satisfies the constraint, no inheritance needed.
"""

from typing import Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float, Bool
import jax.numpy as jnp

from .net_data import NetworkData, NetworkParams, SpikeData


# ── Archetype ─────────────────────────────────────────────────────────────────

class LIFArchetype(Protocol):
    """Required components for LIF neuron dynamics."""
    data:   NetworkData
    params: NetworkParams
    spikes: SpikeData


T_LIF = TypeVar("T_LIF", bound=LIFArchetype)


# ── System ────────────────────────────────────────────────────────────────────

def generate_spike(
    net: T_LIF
) -> T_LIF:
    """Threshold comparison, neuron reset, and SpikeData update.

    Fires neurons whose membrane potential meets or exceeds v_threshold,
    resets their voltage to v_rest, and stores the spike vector in net.spikes.

    Parameters
    ----------
    net : T_LIF
        Current network state — must satisfy LIFArchetype (includes spikes).

    Returns
    -------
    T_LIF
        Updated network state with reset voltages and fresh SpikeData.
    """
    v = net.data.v

    # Threshold → binary spikes
    spikes: Bool[Array, "N_neurons"] = v >= net.params.v_threshold

    # Reset spiked neurons
    v = jnp.where(spikes, net.params.v_rest, v)

    return eqx.tree_at(
        lambda n: (n.data.v, n.spikes.s),
        net,
        (v, spikes)
    )


def step_leaky_integrate(
    net: T_LIF,
    perceptions: Float[Array, "N_perceptors"],
) -> T_LIF:
    """Leaky-integrate half of the LIF step (no spike generation).

    Applies membrane decay, injects perceptor inputs, and propagates spikes
    from the current net.spikes through weighted forward connections.
    Call generate_spike first to ensure net.spikes reflects the current step.

    Parameters
    ----------
    net : T_LIF
        Current network state — any type satisfying LIFArchetype.
    perceptions : Float[Array, "N_perceptors"]
        Input values for perceptor neurons.

    Returns
    -------
    T_LIF
        Updated network state with new membrane potentials.
    """
    v = net.data.v

    # LIF decay
    v = net.params.v_rest + (v - net.params.v_rest) * net.params.LIF_factor

    # Inject perceptor inputs
    v = v.at[net.data.perceptors].add(perceptions)

    # Propagate through weighted connections
    v = v.at[net.data.forward_connections].add(net.data.weights * net.spikes.s[:, None])

    return eqx.tree_at(lambda n: n.data.v, net, v)

def step_LIF(
    net: T_LIF,
    perceptions: Float[Array, "N_perceptors"],
) -> T_LIF:
    net = generate_spike(net)
    return step_leaky_integrate(net, perceptions)
