"""Public API for the `nets` package.

This module re-exports the core data structures, execution functions,
and initialization utilities for spiking neural networks.

Example usage::

    from nets import NetworkData, NetworkParams, step_LIF
    from nets.initialization import init_watts_strogatz
"""

from .net_data import NetworkData, NetworkParams
from .net_exec import step_LIF

__all__ = [
    "NetworkData",
    "NetworkParams",
    "step_LIF",
]
