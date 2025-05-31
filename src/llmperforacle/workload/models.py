"""Data models for workload generation."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Request:
    """Represents an LLM service request."""
    
    request_id: str
    client_id: str
    session_id: str
    arrival_time: float
    prompt_str: Optional[str] = None  # Optional, for realism/logging
    prompt_num_tokens: int = 0
    max_output_tokens: int = 0
    is_conversational_turn: bool = False
    streaming_response: bool = True
    user_priority: int = 0
    prompt_tokens: Optional[List[int]] = None  # Actual token IDs for prefix matching


@dataclass
class ClientProfile:
    """Defines behavior and request characteristics for a specific client type."""
    
    profile_name: str
    weight: float  # For selecting this profile from multiple
    inter_arrival_time_dist_config: Dict[str, any]
    prompt_tokens_dist_config: Dict[str, any]
    max_output_tokens_dist_config: Dict[str, any]
    conversational_probability: float = 0.0  # 0.0 to 1.0
    streaming_response_probability: float = 1.0  # 0.0 to 1.0
    user_priority_dist_config: Optional[Dict[str, any]] = None
    follow_up_inter_arrival_time_dist_config: Optional[Dict[str, any]] = None  # For conversations