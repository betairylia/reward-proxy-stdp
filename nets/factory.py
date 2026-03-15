"""Network factory.

Creates a concrete network entity from an archetype identifier and a TOML
config file — no need to pass dimensions manually in application code.

Usage
-----
    from nets.factory import create_network
    import jax

    net = create_network("lif", jax.random.PRNGKey(0))
    net = create_network("lif", jax.random.PRNGKey(0), config_path="my_config.toml")

Adding a new archetype
----------------------
1. Define the concrete entity type in net_data.py  (e.g. STDPNet).
2. Add its archetype id → type mapping to NET_TYPES below.
3. Add a ``[<id>_params]`` section to config.toml for any extra params.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, overload

import jax

from .net_data import LIFNet, NetworkParams
from .net_exec import LIFArchetype
from .initialization import init_watts_strogatz

# ── Registry ──────────────────────────────────────────────────────────────────

NET_TYPES: dict[str, type] = {
    "lif": LIFNet,
}

_PARAM_TYPES: dict[str, type] = {
    "lif": NetworkParams,
}

# ── Typed overloads (extend as new archetypes are added) ──────────────────────

@overload
def create_network(
    archetype_id: Literal["lif"],
    key: jax.Array,
    config_path: str | Path = ...,
) -> LIFNet: ...

@overload
def create_network(
    archetype_id: str,
    key: jax.Array,
    config_path: str | Path = ...,
) -> LIFArchetype: ...


# ── Factory ───────────────────────────────────────────────────────────────────

def create_network(
    archetype_id: str,
    key: jax.Array,
    config_path: str | Path = "config.toml",
) -> LIFArchetype:
    """Create a network entity from an archetype id and TOML config.

    Parameters
    ----------
    archetype_id : str
        Which archetype to build (e.g. ``"lif"``).
        Must be a key in the NET_TYPES registry.
    key : jax.Array
        JAX PRNG key.
    config_path : str | Path
        Path to the TOML config file. Defaults to ``config.toml`` in the
        current working directory.

    Returns
    -------
    LIFArchetype (or a more specific type when overloads match)
        Initialised network entity.

    Raises
    ------
    KeyError
        If *archetype_id* is not in the registry.
    FileNotFoundError
        If *config_path* does not exist.
    """
    if archetype_id not in NET_TYPES:
        raise KeyError(
            f"Unknown archetype '{archetype_id}'. "
            f"Available: {list(NET_TYPES)}"
        )

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    net_type    = NET_TYPES[archetype_id]
    param_type  = _PARAM_TYPES[archetype_id]
    net_cfg     = cfg["network"]
    params_cfg  = cfg.get(f"{archetype_id}_params", {})
    params      = param_type(**params_cfg) if params_cfg else param_type()

    return init_watts_strogatz(key, net_type, params=params, **net_cfg)
