# Phase 2: Cross-Request Prefix Caching Implementation

## Overview

Phase 2 extends the prefix caching system to support caching and reusing prefixes across different requests, not just within conversational sessions. This enables significant performance improvements for workloads with common patterns like system prompts, few-shot examples, or instruction templates.

## Implementation Summary

### Completed Tasks ✅
1. **Global Prefix Store Data Structures** - Hash-based storage with LRU eviction
2. **Prefix Hashing Mechanism** - SHA256-based token sequence hashing
3. **Cross-Request Detection Logic** - Enhanced cache checking with priority system
4. **LRU Eviction Policy** - Reference counting and configurable cache limits
5. **Workload Profiles for Testing** - Synthetic token generation for common patterns
6. **Comprehensive Tests** - 6 unit tests + 5 integration test scenarios
7. **Metrics Updates** - Separate cross-request hit rate tracking

### Key Files Modified/Added
- `src/llmperforacle/frameworks/prefix_utils.py` - Hashing utilities
- `src/llmperforacle/workload/token_generator.py` - Token generation
- `src/llmperforacle/frameworks/vllm_framework.py` - Core implementation
- `tests/unit/test_cross_request_caching.py` - Unit tests
- `tests/integration/test_cross_request_caching_integration.py` - Integration tests
- `configs/example_cross_request_caching.yaml` - Example configuration

## Key Components

### 1. Global Prefix Store

Added to `vllm_framework.py`:
- `global_prefix_store: Dict[str, GlobalPrefixCacheInfo]` - Stores cached prefixes
- Configurable size limit with LRU eviction
- Reference counting to prevent eviction of active prefixes

### 2. Prefix Hashing (`prefix_utils.py`)

- `hash_token_sequence()` - Creates SHA256 hash of token sequences
- `find_longest_prefix_match()` - Searches for longest matching prefix
- `should_cache_prefix()` - Determines if a prefix should be cached

### 3. Enhanced Request Model

Extended `Request` class with:
- `prompt_tokens: Optional[List[int]]` - Actual token IDs for prefix matching

### 4. Token Generation (`token_generator.py`)

Synthetic token generation for testing:
- System prompts (50-80 tokens)
- Few-shot examples (200-500 tokens)  
- Instruction templates (100-150 tokens)
- Mixed patterns

### 5. Enhanced Workload Generator

Added support for:
- `generate_prompt_tokens` flag to enable token generation
- `prefix_patterns` configuration for controlling prefix distribution
- Token generation based on client profiles

## Implementation Details

### Cache Hit Detection Priority

1. **Conversational prefix** - Check session cache first
2. **Cross-request prefix** - Check global cache if no conversational hit
3. **Cache miss** - No matching prefix found

### Global Cache Population

After request completion:
1. Check if request was a cache miss
2. Verify prefix meets minimum length requirement
3. Add to global cache if space available (or evict LRU entry)
4. Support multiple prefix lengths per request (50, 100, 200, etc.)

### LRU Eviction Policy

- Track `last_access_time` for each cached prefix
- Only evict entries with `reference_count == 0`
- Update access statistics on each hit

### Metrics Enhancement

Added to `MetricsCollector`:
- `cross_request_hit_rate` - Hit rate for non-conversational requests
- `cross_request_hits` - Total cross-request cache hits
- Event type: `"CROSS_REQUEST_HIT"`

## Configuration

```yaml
frameworks_to_test:
  - name: "vllm_cross_request"
    type: "VLLM"
    config:
      enable_prefix_caching: true
      enable_cross_request_caching: true
      min_prefix_cache_length: 50      # Minimum tokens to cache
      max_prefix_cache_size: 100       # Max cached prefixes
      prefix_eviction_policy: "lru"    # Eviction policy

workload:
  generate_prompt_tokens: true
  prefix_patterns:
    patterns:
      - type: "system"
        name: "helpful_assistant"
        weight: 0.3
      - type: "few_shot"
        name: "classification_3shot"
        weight: 0.2
```

## Test Coverage

### Unit Tests (`test_cross_request_caching.py`)
1. Cache miss for first request
2. Cache population after processing
3. Cache hit for subsequent requests
4. LRU eviction when cache full
5. Protection of entries with active references
6. Prefix hashing utilities

### Integration Tests (`test_cross_request_caching_integration.py`)
1. System prompt caching across requests
2. Few-shot example caching
3. Mixed prefix patterns with cache competition
4. Performance comparison with/without cross-request caching
5. Verification that caching can be disabled

## Performance Benefits

Expected improvements with cross-request caching:
- **50-80% cache hit rate** for workloads with common prefixes
- **30-70% reduction in prefill computation**
- **Significant TTFT improvements** for cached requests
- **Better GPU utilization** by skipping redundant computation

## Limitations and Future Work

1. **Simplified memory model** - Cached prefixes assumed to not consume additional memory
2. **No prefix compression** - Each unique prefix stored separately
3. **Static prefix lengths** - Could benefit from dynamic prefix length selection
4. **No prefix sharing across frameworks** - Each framework has its own cache

## Usage Example

```python
# Workload with common system prompts
config = {
    "workload": {
        "generate_prompt_tokens": True,
        "prefix_patterns": {
            "patterns": [
                {"type": "system", "name": "helpful_assistant", "weight": 0.8},
                {"type": "random", "weight": 0.2}
            ]
        }
    }
}
```

This configuration will generate requests where 80% share the same system prompt prefix, allowing the cross-request cache to achieve high hit rates after the initial warm-up period.