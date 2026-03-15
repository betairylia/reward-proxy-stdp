"""Public API for the `nets` package.

Example usage::

    import equinox as eqx
    from nets import NetworkData, NetworkParams, LIFArchetype, T_LIF, step_LIF
    from nets.factory import create_network
    from nets.initialization import init_watts_strogatz

    class LIFNet(eqx.Module):
        data:   NetworkData
        params: NetworkParams

    net = create_network(LIFNet)
    net = init_watts_strogatz(jax.random.PRNGKey(0), net)
"""

from .net_data import NetworkData, NetworkParams
from .net_exec import LIFArchetype, T_LIF, step_LIF

__all__ = [
    # Components
    "NetworkData",
    "NetworkParams",
    # Archetypes + TypeVars
    "LIFArchetype",
    "T_LIF",
    # Systems
    "step_LIF",
]
