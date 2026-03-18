"""STDP demo: one MNIST image presented repeatedly, weights updated each step."""

import jax
import jax.numpy as jnp
import equinox as eqx

from nets import NetworkData, NetworkParams, step_LIF, step_normalize_weight
from nets.net_data import SpikeData
from nets.factory import create_network
from nets.initialization import init_watts_strogatz
from nets.learning.stdp import STDPData, STDPParams, step_STDP
from datasets import load_mnist


# ── Entity type ───────────────────────────────────────────────────────────────

class STDPNet(eqx.Module):
    """LIF network with STDP learning. Satisfies both LIFArchetype and STDPArchetype."""
    data:        NetworkData
    params:      NetworkParams
    spikes:      SpikeData
    stdp_data:   STDPData
    stdp_params: STDPParams


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Reward-Proxy STDP demo")
    print("=" * 50)

    print("\n1. Loading MNIST...")
    test_data = load_mnist(train=False)
    image, label = test_data[0]
    perceptions = image
    print(f"   Image label: {label}")

    print("\n2. Initializing network...")
    net = create_network(STDPNet)
    net = init_watts_strogatz(jax.random.PRNGKey(42), net)
    N, degree = net.data.v.shape[0], net.data.weights.shape[1]
    print(f"   {N} neurons, degree {degree}, small-world topology")

    print("\n3. Running STDP loop (100 steps, same image)...")
    w_before = float(net.data.weights.mean())

    for t in range(100):
        net = step_LIF(net, perceptions)
        net = step_STDP(net)
        net = step_normalize_weight(net)

        if (t + 1) % 25 == 0:
            spike_rate = float(net.spikes.s.astype(float).mean())
            w_mean     = float(net.data.weights.mean())
            trace_mean = float(net.stdp_data.spike_trace.mean())
            print(f"   step {t+1:3d} | spikes: {spike_rate:.3f} | "
                  f"mean w: {w_mean:.5f} | mean trace: {trace_mean:.4f}")

    w_after = float(net.data.weights.mean())
    print(f"\n   Weight drift: {w_before:.5f} → {w_after:.5f}  "
          f"(Δ = {w_after - w_before:+.5f})")

    predicted = int(jnp.argmax(net.data.v[net.data.effectors]))
    print(f"   Effector argmax: {predicted}  (label: {label})")


if __name__ == "__main__":
    main()
