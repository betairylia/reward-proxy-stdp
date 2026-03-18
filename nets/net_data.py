import math

import equinox as eqx
from jaxtyping import Array, Float, Int, Bool


# ── Components ────────────────────────────────────────────────────────────────

class NetworkParams(eqx.Module):
    """LIF neuron hyperparameters."""

    v_threshold: float = 1.0
    v_rest:      float = 0.0

    # RC characteristic time ≈ 5× spike width
    LIF_factor:  float = math.exp(-1 / 6.0)


class NetworkData(eqx.Module):
    """Dynamic network state (topology + membrane potentials)."""

    # Membrane potential
    v:                   Float[Array, "N_neurons"]

    # Connectivity
    forward_connections: Int[Array, "N_neurons degree"]
    weights:             Float[Array, "N_neurons degree"]

    # I/O neuron indices
    perceptors:          Int[Array, "N_perceptors"]
    effectors:           Int[Array, "N_effectors"]
    out_edge_mask:      Bool[Array, "N_neurons"]

class SpikeData(eqx.Module):
    s: Bool[Array, "N_neurons"]
