"""Network factory.

Instantiates a concrete entity type by building each of its components from
a TOML config file.  The mapping from component type → how to build it is an
open registry, so new components (e.g. STDPData) are supported by calling
``register_component`` once at import time — no changes to the factory needed.

Typical workflow
----------------
    # 1. Define your entity type wherever you like (e.g. main.py)
    class LIFNet(eqx.Module):
        data:   NetworkData
        params: NetworkParams

    # 2. Create a shell with correct dimensions/hyperparams from config.toml
    net = create_network(LIFNet)

    # 3. Initialise topology separately
    net = init_watts_strogatz(jax.random.PRNGKey(0), net)

Adding a new component
----------------------
    register_component(STDPData, lambda cfg: STDPData(
        traces=jnp.zeros(cfg["network"]["N_neurons"])
    ))
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any, Callable, TypeVar, get_type_hints

import jax.numpy as jnp

from .net_data import NetworkData, NetworkParams, SpikeData

T = TypeVar("T")

# ── Component builder registry ────────────────────────────────────────────────

_BUILDERS: dict[type, Callable[[dict], Any]] = {}


def register_component(typ: type, builder: Callable[[dict], Any]) -> None:
    """Register a factory function for a component type.

    Parameters
    ----------
    typ :
        The component class (e.g. ``NetworkData``).
    builder :
        Callable ``(cfg: dict) -> instance`` where *cfg* is the full parsed
        TOML dict.
    """
    _BUILDERS[typ] = builder


# ── Built-in component builders ───────────────────────────────────────────────

def _build_network_data(cfg: dict) -> NetworkData:
    net = cfg["network"]
    N, d, Np, Ne = net["N_neurons"], net["degree"], net["N_perceptors"], net["N_effectors"]
    return NetworkData(
        v=jnp.zeros(N),
        forward_connections=jnp.zeros((N, d), dtype=jnp.int32),
        weights=jnp.zeros((N, d)),
        perceptors=jnp.arange(Np, dtype=jnp.int32),
        effectors=jnp.arange(N - Ne, N, dtype=jnp.int32),
    )


def _build_network_params(cfg: dict) -> NetworkParams:
    return NetworkParams(**cfg.get("lif_params", {}))


def _build_spike_data(cfg: dict) -> SpikeData:
    N = cfg["network"]["N_neurons"]
    return SpikeData(s=jnp.zeros(N, dtype=jnp.bool_))


register_component(NetworkData, _build_network_data)
register_component(NetworkParams, _build_network_params)
register_component(SpikeData, _build_spike_data)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_network(
    net_type: type[T],
    config_path: str | Path = "config.toml",
) -> T:
    """Build an entity by instantiating each of its component fields from config.

    Inspects ``net_type``'s fields, looks up a builder for each component type,
    and calls ``net_type(field=component, ...)``.

    Parameters
    ----------
    net_type :
        Any ``eqx.Module`` subclass whose fields are registered component types.
    config_path :
        Path to the TOML config file.

    Returns
    -------
    T
        A new instance of ``net_type`` with all components built from config.

    Raises
    ------
    KeyError
        If a field's type has no registered builder.
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    hints = get_type_hints(net_type)
    components: dict[str, Any] = {}

    for field in dataclasses.fields(net_type):  # type: ignore[arg-type]
        field_type = hints[field.name]
        if field_type not in _BUILDERS:
            raise KeyError(
                f"No builder registered for component type '{field_type.__name__}'. "
                f"Call register_component({field_type.__name__}, builder) to add one."
            )
        components[field.name] = _BUILDERS[field_type](cfg)

    return net_type(**components)
