"""Network initialization strategies.

This module provides various network topology initialization methods
for spiking neural networks.
"""

from .small_world import init_watts_strogatz

__all__ = [
    "init_watts_strogatz",
]
