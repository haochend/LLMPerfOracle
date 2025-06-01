"""Data models for LLM framework simulation."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SequenceState:
    """Tracks the state of a sequence being processed by an LLM framework."""
    
    request_id: str
    request: Any  # Will be Request object from workload module
    status: str  # 'WAITING_FOR_PREFILL', 'PREFILLING', 'DECODING', 'COMPLETED'
    prompt_tokens_processed: int = 0
    output_tokens_generated: int = 0
    allocated_kv_blocks: List[Any] = field(default_factory=list)
    prefill_start_time: float = -1.0  # Simulation time
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    # Prefix caching fields
    cached_prefix_length_used: int = 0  # Number of tokens reused from cache
    num_tokens_requiring_prefill: int = 0  # Actual tokens that need prefill computation
    prompt_tokens_fully_processed: int = 0  # Total prompt tokens with KV cache available
    kv_block_ids_for_prompt: List[Any] = field(default_factory=list)  # For detailed block tracking
    prefill_end_time_sim: Optional[float] = None  # For TTFT calculation when prefill is skipped
    
    # Chunked prefill fields
    prefill_chunks_completed: int = 0  # Number of chunks already processed
    total_prefill_chunks: int = 1  # Total chunks needed for this sequence
    current_prefill_position: int = 0  # Current position in the prompt for chunked prefill


@dataclass
class SessionCacheInfo:
    """Information about KV cache state for an active session."""
    
    session_id: str
    total_tokens_in_cache: int  # Length of prefix + generated response in KV cache
    prompt_part_length: int  # Length of the prompt part
    response_part_length: int  # Length of the response part
    associated_sequence_id: str  # The request_id currently holding this session's cache
    kv_block_ids: List[Any] = field(default_factory=list)  # Optional, for detailed block management
    last_update_time: float = 0.0  # Simulation time, for potential eviction policies


@dataclass
class GlobalPrefixCacheInfo:
    """Information about a globally cached prefix that can be shared across requests."""
    
    prefix_hash: str  # Hash of the token sequence
    prefix_length: int  # Number of tokens in the prefix
    kv_block_ids: List[Any] = field(default_factory=list)  # KV cache blocks storing this prefix
    reference_count: int = 0  # Number of active sequences using this prefix
    last_access_time: float = 0.0  # For LRU eviction
    creation_time: float = 0.0  # When this prefix was first cached
    access_count: int = 0  # Total number of times this prefix was reused
    # Optional: store actual prefix tokens for verification (in production, this would be memory-intensive)
    prefix_tokens: Optional[List[int]] = None