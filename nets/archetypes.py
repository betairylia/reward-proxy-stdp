"""Archetype Protocols for SNN systems.

Each Protocol declares the component fields a system needs.
Systems use a TypeVar bound to an archetype so they accept any
concrete type that satisfies the constraint and return that same type.

Example
-------
    T = TypeVar('T', bound=LIFArchetype)

    def step_LIF(net: T, perceptions: Array) -> T:
        ...  # accepts NetworkData, or any richer type that has these fields
"""

from typing import Protocol, TypeVar

from jaxtyping import Array, Float, Int

from .net_data import NetworkParams


class LIFArchetype(Protocol):
    """Required components for LIF neuron dynamics."""

    params:                 NetworkParams
    v:                      Float[Array, "N_neurons"]
    forward_connections:    Int[Array, "N_neurons degree"]
    weights:                Float[Array, "N_neurons degree"]
    perceptors:             Int[Array, "N_perceptors"]
    effectors:              Int[Array, "N_effectors"]


T_LIF = TypeVar("T_LIF", bound=LIFArchetype)
