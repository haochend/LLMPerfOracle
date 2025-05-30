"""Virtual hardware layer module."""

from .virtual_hardware import (
    HardwareResourceProfile,
    VirtualComputeDevice,
    VirtualHardwarePlatform,
    VirtualNetworkLink,
)

__all__ = [
    "HardwareResourceProfile",
    "VirtualComputeDevice",
    "VirtualNetworkLink",
    "VirtualHardwarePlatform",
]