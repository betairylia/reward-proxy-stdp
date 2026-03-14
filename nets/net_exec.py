from jaxtyping import Array, Float, Int, Bool
import jax
import jax.numpy as jnp

from .net_data import NetworkData

def step_LIF(
    net: NetworkData,
    perceptions: Float[Array, "N_perceptors"]
) -> NetworkData:
    """Single timestep of LIF neuron dynamics.
    
    Parameters
    ----------
    net : NetworkData
        Current network state
    perceptions : Float[Array, "N_perceptors"]
        Input values for perceptor neurons
    
    Returns
    -------
    NetworkData
        Updated network state
    """
    # Start with current membrane potentials
    v = net.v
    
    # LIF Decay
    v = net.params.v_rest + (v - net.params.v_rest) * net.params.LIF_factor
    
    # Apply inputs to perceptor neurons
    v = v.at[net.perceptors].add(perceptions)
    
    # Mark binary spikes
    spikes: Bool[Array, "N_neurons"] = v >= net.params.v_threshold
    
    # Reset spiked neurons to rest potential
    v = jnp.where(spikes, net.params.v_rest, v)
    
    # Propagate spikes through connections
    spike_contributions = net.weights * spikes[:, None]
    v = v.at[net.forward_connections].add(spike_contributions)
    
    return net._replace(v=v)
