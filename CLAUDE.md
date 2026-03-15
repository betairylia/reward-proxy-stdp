# reward-proxy-stdp

## Rule Zero

Always make checkpoint commits before implementation.

Spiking neural network (SNN) library using LIF neurons with small-world connectivity, built on JAX + Equinox.

## ECS-like Architecture

The framework separates **components** (data), **archetypes** (structural constraints), and **systems** (dynamics) — inspired by ECS but adapted for functional/JAX patterns.

### Components — `nets/net_data.py`

Plain `eqx.Module` structs. No methods, no logic.

```python
class NetworkData(eqx.Module):
    v: Float[Array, "N"]          # membrane potentials
    forward_connections: ...      # topology
    weights: ...

class NetworkParams(eqx.Module):
    v_threshold: float
    v_rest: float
    LIF_factor: float
```

### Archetypes — co-located with systems

Protocols that declare which components a system requires. No base class — structural subtyping.

```python
class LIFArchetype(Protocol):
    data:   NetworkData
    params: NetworkParams

T_LIF = TypeVar("T_LIF", bound=LIFArchetype)
```

### Entity types — defined by the user, not the library

The user composes components into a concrete type. The library never defines `LIFNet`.

```python
class LIFNet(eqx.Module):
    data:   NetworkData
    params: NetworkParams
    # add more components freely
```

### Systems — `nets/net_exec.py`

Generic functions over the archetype TypeVar. Always return the same type they receive.

```python
def step_LIF(net: T_LIF, perceptions: Array) -> T_LIF:
    ...
    return eqx.tree_at(lambda n: n.data.v, net, new_v)
```

`eqx.tree_at()` is used for all state updates — preserves the concrete type and stays JAX-transformable.

### Factory — `nets/factory.py`

Builds entity instances from `config.toml` via an open registry. Users register builders alongside their component definitions.

```python
register_component(NetworkData, lambda cfg: NetworkData(...))
register_component(NetworkParams, lambda cfg: NetworkParams(...))

net = create_network(LIFNet)   # discovers fields, calls registered builders
```

Adding a new component requires only `register_component(...)` — no changes to the factory.

### Initialization — `nets/initialization/`

Topology is filled separately from construction:

```python
net = create_network(LIFNet)          # allocates arrays
net = init_watts_strogatz(key, net)   # fills connectivity + weights
```

### Learning rules — `nets/learning/`

Follow the same archetype pattern. STDP is in progress (`nets/learning/stdp.py`): `STDPData`, `STDPParams`, `STDPArchetype`, `T_STDP`, and a `step_STDP` skeleton are defined.

## Configuration

All hyperparameters live in `config.toml`:

```toml
[network]
N_neurons = 1000
degree    = 20

[lif_params]
v_threshold = 1.0
LIF_factor  = 0.8464...   # exp(-1/6)
```

## Conventions

- Components are `eqx.Module` — immutable, JAX pytrees.
- State updates use `eqx.tree_at()`, never mutation.
- Archetypes use `Protocol` + `TypeVar(bound=...)` — not inheritance.
- Entity types live outside the library (e.g., in `main.py` or user code).
- `nets/__init__.py` is the public API — import from there.
