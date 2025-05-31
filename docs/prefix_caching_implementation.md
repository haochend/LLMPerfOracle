# Prefix Caching Implementation Guide

## Overview

This document describes the implementation of prefix caching (KV cache reuse) in LLMPerfOracle, following the design in Document 8.

## Current Implementation Status

### ✅ Completed (Phase 1)

1. **Data Models**
   - Extended `SequenceState` with prefix caching fields:
     - `cached_prefix_length_used`: Number of tokens reused from cache
     - `num_tokens_requiring_prefill`: Actual tokens that need prefill computation
     - `prompt_tokens_fully_processed`: Total prompt tokens with KV cache available
   - Added `SessionCacheInfo` to track KV cache state per session

2. **Core Framework Support**
   - Implemented conversational prefix caching in `VLLMFramework`:
     - `_check_prefix_cache()`: Detects cache hits for conversational turns
     - `_update_session_cache()`: Updates session cache after sequence completion
     - Modified admission logic to allocate blocks only for new tokens
     - Updated prefill execution to skip computation for cached tokens

3. **Metrics Collection**
   - Added `log_prefix_cache_event()` to track cache events
   - Extended `RequestMetricsEntry` with prefix cache metrics
   - Added prefix cache statistics to summary reports:
     - Overall hit rate
     - Conversational hit rate
     - Prefill reduction ratio
     - Total tokens saved

4. **Testing**
   - Unit tests for prefix caching logic (`test_prefix_caching.py`)
   - Integration tests for full simulations (`test_prefix_caching_integration.py`)

### ⚠️ Known Issues

1. **Workload Generator Limitation**
   - The current workload generator does not accumulate tokens for conversational turns
   - Follow-up requests have independent token counts rather than growing prompts
   - This prevents proper testing of conversational prefix caching

2. **Simplified Memory Model**
   - Currently uses simplified KV block management
   - Cached prefixes are assumed to not consume additional blocks
   - More accurate reference counting is needed for production use

### ❌ Not Implemented (Phase 2-3)

1. **Cross-Request Prefix Caching**
   - Global prefix store for shared system prompts
   - Prefix hashing and matching logic
   - Cache eviction policies

2. **Detailed KV Block Reference Counting**
   - Reference counting for shared KV blocks
   - Accurate memory footprint tracking
   - Block deduplication

## Usage

### Enabling Prefix Caching

```yaml
frameworks_to_test:
  - name: "vllm_with_caching"
    type: "VLLM"
    config:
      enable_prefix_caching: true  # Default: true
```

### Workload Configuration

For prefix caching to be effective, use conversational workloads:

```yaml
workload:
  client_profiles:
    - profile_name: "conversational"
      conversational_probability: 0.8
      follow_up_inter_arrival_time_dist_config:
        type: "Constant"
        value: 0.5
```

**Note**: Currently, the workload generator needs enhancement to properly simulate growing conversation histories.

## Metrics and Reporting

Prefix caching metrics appear in the simulation summary:

```json
{
  "prefix_caching": {
    "overall_hit_rate": 0.45,
    "conversational_hit_rate": 0.78,
    "event_counts": {
      "CONVERSATIONAL_HIT": 234,
      "MISS_FULL": 156,
      "FULL_HIT_NO_PREFILL_NEEDED": 12
    },
    "average_cached_prefix_length": 450.5,
    "prefill_reduction_ratio": 0.42,
    "total_tokens_saved": 105300
  }
}
```

CSV output includes per-request prefix cache information:
- `prefix_cache_event`: Event type (HIT/MISS)
- `cached_prefix_length`: Tokens reused
- `tokens_actually_prefilled`: New tokens processed

## Implementation Details

### Cache Hit Detection

The framework checks for cache hits when admitting new requests:

1. **Conversational Hit**: If `is_conversational_turn` and session exists in cache
2. **Length Check**: New prompt must be longer than cached content
3. **Cache Reuse**: Skip prefill for cached portion, only process new tokens

### Session Cache Management

After each request completes:
1. Update `active_sessions_kv_state` with total tokens (prompt + response)
2. Store KV block references for potential reuse
3. Track last update time for future eviction policies

### Performance Impact

When cache hits occur:
- Prefill computation is reduced by `cached_prefix_length` tokens
- TTFT improves proportionally to the reduction in prefill work
- Memory usage is more efficient (shared KV blocks)

## Future Enhancements

### 1. Workload Generator Enhancement
- Modify `_create_request()` to accumulate tokens for conversations
- Add option for full history vs. incremental prompts
- Support different conversation growth patterns

### 2. Cross-Request Caching
- Implement global prefix store
- Add prefix matching algorithms
- Support common system prompts and few-shot examples

### 3. Advanced Memory Management
- Implement reference counting for KV blocks
- Add block-level eviction policies
- Support memory pressure handling

### 4. Additional Metrics
- Cache hit rate by prefix length
- Memory savings from deduplication
- Impact on scheduling efficiency

## Testing Recommendations

1. **Unit Tests**: Test cache logic with mock requests
2. **Integration Tests**: Full simulations with conversational workloads
3. **Performance Tests**: Compare with/without caching
4. **Memory Tests**: Verify block allocation and reuse

## Configuration Examples

### High Cache Hit Scenario
```yaml
# Long conversations with growing context
client_profiles:
  - conversational_probability: 0.9
    prompt_tokens_dist_config:
      type: "Linear"
      start: 100
      end: 2000
```

### Low Cache Hit Scenario
```yaml
# Independent requests with no reuse
client_profiles:
  - conversational_probability: 0.0
    prompt_tokens_dist_config:
      type: "Uniform"
      low: 50
      high: 500
```