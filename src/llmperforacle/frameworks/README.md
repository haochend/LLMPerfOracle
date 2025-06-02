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
- **Prefix Caching**: Conversational and cross-request KV cache reuse
- **Chunked Prefill**: Handles large prompts exceeding batch size

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
      # Advanced features
      enable_prefix_caching: true
      enable_cross_request_caching: true
      enable_chunked_prefill: true
      prefill_chunk_size: 4096
```

**Internal Logic:**
1. **Admission Control**: Check KV cache availability
2. **Batch Formation**: Combine prefill and decode operations
3. **Resource Allocation**: Manage KV cache blocks per sequence
4. **Completion Handling**: Release resources on finish/failure

### ParallelVLLMFramework

Extends VLLMFramework with multi-GPU parallelism support:

**Parallelism Strategies:**
- **Tensor Parallelism (TP)**: Shards model layers across GPUs
- **Pipeline Parallelism (PP)**: Splits model into stages
- **Data Parallelism (DP)**: Multiple model replicas
- **Hybrid Strategies**: TP+PP, TP+DP, TP+PP+DP

**Configuration:**
```yaml
frameworks_to_test:
  - name: "vllm_parallel"
    type: "ParallelVLLM"
    config:
      model_profile_id: "Llama2-13B"
      parallelism:
        strategy: "TP"  # or "PP", "TP_PP"
        tp_degree: 4
        pp_stages: 2
        gpu_ids: ["gpu0", "gpu1", "gpu2", "gpu3"]
```

**Key Features:**
- Accurate communication overhead modeling
- Pipeline bubble simulation
- Collective operation timing (AllReduce, AllGather)
- Memory distribution across devices
- Dynamic routing for load balancing

## Advanced Features

### Prefix Caching

Both framework implementations support advanced prefix caching:

**Conversational Caching:**
- Tracks KV cache state per session
- Reuses cache for multi-turn conversations
- 98% TTFT improvement for cached requests

**Cross-Request Caching:**
- Global prefix store with SHA256 hashing
- LRU eviction policy
- Configurable cache size and minimum prefix length
- 30-70% prefill computation reduction

### Chunked Prefill

Handles large prompts that exceed max batch size:
- Automatically splits prompts into chunks
- Processes chunks iteratively
- Enables 10,000+ token prompt handling
- Dynamic batch size based on hardware

### Level of Detail (LoD) Support

Frameworks respect simulation LoD settings:
- **High LoD**: Layer-by-layer processing
- **Medium LoD**: Aggregated operations
- 5-20x simulation speedup with medium LoD

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
- Conversational prefix caching (70%+ hit rates)
- Cross-request prefix caching (shared system prompts)
- Chunked prefill for large prompts
- Streaming response support
- Dynamic batch size calculation

### Future Frameworks
- **TensorRT-LLM**: In-flight batching, scheduling policies
- **SGLang**: RadixAttention, structured decoding
- **Dynama**: Disaggregated serving, KV-aware routing
- **Triton**: Multi-backend support, dynamic batching