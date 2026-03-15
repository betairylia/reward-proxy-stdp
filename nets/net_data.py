import math

import equinox as eqx
from jaxtyping import Array, Float, Int


class NetworkParams(eqx.Module):

    # LIF neuron

    # Threshold and rest potential
    v_threshold: float = 1.0
    v_rest: float = 0.0

    # LIF's RC characteristic time, approx. 5x spike time
    LIF_factor: float = math.exp(-1 / 6.0)


class NetworkData(eqx.Module):

    params:                     NetworkParams

    # Membrane Potential
    v:                    Float[Array, "N_neurons"]

    # Forward connections (index)
    forward_connections:    Int[Array, "N_neurons degree"]

    # Weights
    weights:              Float[Array, "N_neurons degree"]

    # I/O
    # Index of perceptor / effector neurons
    perceptors:             Int[Array, "N_perceptors"]
    effectors:              Int[Array, "N_effectors"]
