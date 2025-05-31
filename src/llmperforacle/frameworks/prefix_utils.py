"""Utilities for prefix caching, including hashing and prefix detection."""

import hashlib
from typing import List, Optional, Tuple


def hash_token_sequence(tokens: List[int], max_length: Optional[int] = None) -> str:
    """Create a hash for a sequence of tokens.
    
    Args:
        tokens: List of token IDs
        max_length: If provided, only hash the first max_length tokens
        
    Returns:
        Hexadecimal hash string
    """
    if max_length is not None:
        tokens = tokens[:max_length]
    
    # Convert tokens to bytes for hashing
    token_bytes = b''.join(token.to_bytes(4, 'little') for token in tokens)
    
    # Use SHA256 for a good balance of speed and collision resistance
    return hashlib.sha256(token_bytes).hexdigest()


def find_longest_prefix_match(
    prompt_tokens: List[int], 
    prefix_hashes: dict, 
    min_prefix_length: int = 50,
    max_prefix_length: Optional[int] = None
) -> Tuple[Optional[str], int]:
    """Find the longest matching prefix in the global cache.
    
    Args:
        prompt_tokens: The full prompt token sequence
        prefix_hashes: Dictionary mapping prefix hashes to GlobalPrefixCacheInfo
        min_prefix_length: Minimum prefix length to consider
        max_prefix_length: Maximum prefix length to consider
        
    Returns:
        Tuple of (matching_hash, prefix_length) or (None, 0) if no match
    """
    if len(prompt_tokens) < min_prefix_length:
        return None, 0
    
    # Determine the maximum prefix length to check
    if max_prefix_length is None:
        max_check_length = len(prompt_tokens)
    else:
        max_check_length = min(max_prefix_length, len(prompt_tokens))
    
    # Check from longest to shortest prefix for best cache utilization
    for length in range(max_check_length, min_prefix_length - 1, -1):
        prefix_hash = hash_token_sequence(prompt_tokens, length)
        if prefix_hash in prefix_hashes:
            return prefix_hash, length
    
    return None, 0


def should_cache_prefix(
    prefix_length: int,
    min_cache_length: int = 50,
    max_cache_size: int = 100,
    current_cache_size: int = 0
) -> bool:
    """Determine if a prefix should be added to the global cache.
    
    Args:
        prefix_length: Length of the prefix to potentially cache
        min_cache_length: Minimum length for a prefix to be worth caching
        max_cache_size: Maximum number of prefixes to store
        current_cache_size: Current number of cached prefixes
        
    Returns:
        True if the prefix should be cached
    """
    if prefix_length < min_cache_length:
        return False
    
    if current_cache_size >= max_cache_size:
        return False  # Would need eviction logic
    
    return True


def get_common_prefix_patterns() -> List[List[int]]:
    """Return common prefix patterns for testing.
    
    In a real system, these would be actual token sequences for:
    - System prompts
    - Few-shot examples
    - Common instructions
    
    For simulation, we use synthetic token sequences.
    """
    return [
        # System prompt: "You are a helpful assistant..."
        list(range(1000, 1050)),
        
        # Translation instruction: "Translate the following text to French:"
        list(range(2000, 2030)),
        
        # Code instruction: "Write a Python function that..."
        list(range(3000, 3040)),
        
        # Analysis instruction: "Analyze the following data and provide insights:"
        list(range(4000, 4045)),
        
        # Few-shot example prefix
        list(range(5000, 5100)),
    ]