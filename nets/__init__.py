"""Public API for the `nets` package.

Example usage::

    from nets import LIFNet, NetworkData, NetworkParams, step_LIF
    from nets.factory import create_network
    from nets.initialization import init_watts_strogatz
"""

from .net_data import NetworkData, NetworkParams, LIFNet
from .net_exec import LIFArchetype, T_LIF, step_LIF

__all__ = [
    # Components
    "NetworkData",
    "NetworkParams",
    # Entities
    "LIFNet",
    # Archetypes + TypeVars
    "LIFArchetype",
    "T_LIF",
    # Systems
    "step_LIF",
]
