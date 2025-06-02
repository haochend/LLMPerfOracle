# Virtual Hardware Layer

This module models the performance characteristics of hardware components (GPUs, CPUs, memory, networks) in an abstract, parameterized way.

## Overview

The Virtual Hardware Layer provides parametric cost models for estimating operation execution times without requiring actual hardware. It uses the Roofline model to determine whether operations are compute-bound or memory-bound.

## Components

### VirtualHardwarePlatform

The main class managing all virtual hardware devices and providing resource request APIs.

**Key Features:**
- Models multiple compute devices (GPUs/CPUs)
- Simulates network links with bandwidth and latency
- Tracks resource contention using SimPy resources
- Provides parametric cost models for operation timing
- Supports Level of Detail (LoD) for simulation speedup
- Accurate modeling of collective operations (AllReduce, AllGather)

### VirtualComputeDevice

Represents a single compute device with:
- Peak TFLOPS for different precisions (fp16, int8)
- Memory capacity and bandwidth
- Processing units (e.g., GPU SMs)
- SimPy resources for contention modeling

### VirtualNetworkLink

Models network connections between nodes with:
- Bandwidth (bits per second)
- Latency (seconds)
- Utilization tracking

## Key Methods

- `submit_computation_task(device_id, task_description)`: Execute a compute task
- `allocate_memory(device_id, size_bytes)`: Allocate device memory
- `free_memory(device_id, size_bytes)`: Release device memory
- `submit_network_transfer_task(source, dest, size_bytes)`: Transfer data

## Cost Model

The module uses a simplified Roofline model:

1. **Compute Time**: `flops_required / (peak_tflops * 1e12)`
2. **Memory Time**: `bytes_moved / (memory_bandwidth_gbps * 1e9)`
3. **Effective Time**: Based on arithmetic intensity and operation type

## Configuration Example

```yaml
hardware_profile:
  compute_devices:
    - device_id: "gpu0"
      device_type: "GPU"
      peak_tflops: {"fp16": 312, "int8": 624}
      memory_capacity_bytes: 80_000_000_000  # 80GB
      memory_gbps: 2000
      processing_units: 108
  
  network_links:
    - link_id: "client_to_server"
      source_id: "client_node_0"
      dest_id: "framework_entry_0"
      bandwidth_bps: 10_000_000_000  # 10 Gbps
      latency_s: 0.0001  # 100 µs
```

## Model Characteristics Database

The module uses a JSON database to store model-specific parameters like FLOPs per token and memory requirements. See `configs/model_params.json` for examples.