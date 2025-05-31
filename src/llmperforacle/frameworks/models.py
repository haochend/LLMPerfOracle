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