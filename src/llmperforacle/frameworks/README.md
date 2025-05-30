# LLM Framework Simulation Modules

This module provides pluggable simulations of different LLM serving frameworks.

## Overview

Each framework implementation models the unique scheduling, batching, and resource management strategies of real LLM serving systems. All frameworks inherit from a common abstract base class to ensure consistent interfaces.

## Abstract Base Class

### AbstractLLMFramework

Defines the common interface that all framework implementations must follow:

**Required Methods:**
- `handle_incoming_request(request)`: Accept new requests
- `processing_loop()`: Main scheduling and processing logic
- `get_status()`: Report framework state

**Provided Utilities:**
- `_estimate_prefill_ops()`: Calculate prefill computational requirements
- `_estimate_decode_op()`: Calculate decode step requirements
- `_estimate_kv_cache_request_bytes()`: Estimate KV cache size

## Implemented Frameworks

### VLLMFramework

Simulates the vLLM serving framework with:

**Key Features:**
- **PagedAttention KV Cache**: Block-based memory management
- **Continuous Batching**: Dynamic batch composition per iteration
- **Request States**: waiting → running → completed
- **Memory Management**: SimPy containers for KV cache blocks
- **Preemption Support**: Handles KV cache exhaustion

**Configuration:**
```yaml
frameworks_to_test:
  - name: "vllm_instance"
    type: "VLLM"
    config:
      model_profile_id: "Llama2-7B"
      gpu_id: "gpu0"
      block_size: 16  # tokens per KV cache block
      max_num_seqs: 256
      max_num_batched_tokens: 4096
      scheduler_iteration_delay_s: 0.0001
```

**Internal Logic:**
1. **Admission Control**: Check KV cache availability
2. **Batch Formation**: Combine prefill and decode operations
3. **Resource Allocation**: Manage KV cache blocks per sequence
4. **Completion Handling**: Release resources on finish/failure

## Adding New Frameworks

To add a new framework:

1. Create a new class inheriting from `AbstractLLMFramework`
2. Implement the three required methods
3. Add framework-specific logic (batching, scheduling, etc.)
4. Register in `FRAMEWORK_CLASS_MAP` in orchestration module

Example structure:
```python
class MyFramework(AbstractLLMFramework):
    def __init__(self, ...):
        super().__init__(...)
        # Framework-specific initialization
    
    def handle_incoming_request(self, request):
        # Queue the request
        yield self.request_arrival_queue.put(request)
    
    def processing_loop(self):
        while True:
            # Scheduling logic
            # Batch formation
            # Hardware interaction
            # Metrics logging
            yield self.simpy_env.timeout(delay)
    
    def get_status(self):
        return {
            "queue_length": len(self.request_arrival_queue.items),
            # Other status info
        }
```

## Framework-Specific Features

### vLLM
- PagedAttention with block allocation
- Continuous batching scheduler
- Automatic prefix caching (simulated)
- Streaming response support

### Future Frameworks
- **TensorRT-LLM**: In-flight batching, scheduling policies
- **SGLang**: RadixAttention, structured decoding
- **Dynama**: Disaggregated serving, KV-aware routing
- **Triton**: Multi-backend support, dynamic batching