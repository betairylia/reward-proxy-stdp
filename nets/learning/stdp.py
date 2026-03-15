"""STDP learning rule: components, archetype, and weight-update system.

Spike-Timing-Dependent Plasticity adjusts synaptic weights based on the
relative timing of pre- and postsynaptic spikes, approximated via a
per-neuron exponential trace:

  trace[t] = beta * trace[t-1] + spike[t]

Weight update at each step:
  LTP: post fires now, pre trace is high  →  +lr_base * lr_positive
  LTD: pre fires now, post trace is high  →  -lr_base * lr_negative

step_STDP expects spikes to already reflect the current timestep
(i.e., generate_spike must be called before step_STDP).
"""

from typing import Protocol, TypeVar

import equinox as eqx
from jaxtyping import Array, Float
from nets.net_data import NetworkData, NetworkParams, SpikeData


# ── Components ────────────────────────────────────────────────────────────────

class STDPParams(eqx.Module):
    """Hyperparameters for the STDP learning rule."""
    lr_base:          Float = 1e-3   # global learning rate scale
    lr_positive:      Float = 1.0    # LTP coefficient (potentiation)
    lr_negative:      Float = 1.0    # LTD coefficient (depression)
    spike_trace_beta: Float = 0.99   # trace decay factor per timestep

class STDPData(eqx.Module):
    """Mutable STDP state."""
    spike_trace: Float[Array, "N_neurons"]  # per-neuron exponential spike trace


# ── Archetype ─────────────────────────────────────────────────────────────────

class STDPArchetype(Protocol):
    """Required components for STDP weight updates."""
    data:        NetworkData
    stdp_data:   STDPData
    params:      NetworkParams
    stdp_params: STDPParams
    spikes:      SpikeData

T_STDP = TypeVar("T_STDP", bound=STDPArchetype)


# ── System ────────────────────────────────────────────────────────────────────

def step_STDP(
    net: T_STDP,
) -> T_STDP:
    """Update synaptic weights and spike trace for one timestep.

    Reads net.spikes (set by generate_spike) and applies the STDP rule:
    LTP when a postsynaptic neuron fires shortly after a presynaptic one,
    LTD when a presynaptic neuron fires shortly after a postsynaptic one.

    Parameters
    ----------
    net : T_STDP
        Current network state — must satisfy STDPArchetype.

    Returns
    -------
    T_STDP
        Updated network state with new weights and spike trace.
    """
    # Decay trace and accumulate current spikes
    new_trace = net.stdp_params.spike_trace_beta * net.stdp_data.spike_trace + net.spikes.s

    # LTD term: pre fires now, post had a recent trace  →  depression
    pre_sr = net.spikes.s[:, None] * new_trace[net.data.forward_connections]

    # LTP term: post fires now, pre had a recent trace  →  potentiation
    post_sr = net.spikes.s[net.data.forward_connections] * new_trace[:, None]

    # Weight update: potentiate LTP synapses, depress LTD synapses
    new_w = (
        net.data.weights
        + net.stdp_params.lr_base * net.stdp_params.lr_positive * post_sr
        - net.stdp_params.lr_base * net.stdp_params.lr_negative * pre_sr
    )

    return eqx.tree_at(
        lambda n: (n.data.weights, n.stdp_data.spike_trace),
        net,
        (new_w, new_trace)
    )
