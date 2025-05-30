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