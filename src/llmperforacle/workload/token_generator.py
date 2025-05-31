"""Token generation utilities for cross-request prefix caching testing."""

import random
from typing import List, Dict, Any


class TokenGenerator:
    """Generates synthetic token sequences for testing prefix caching."""
    
    def __init__(self, seed: int = None):
        """Initialize token generator with optional seed."""
        if seed is not None:
            random.seed(seed)
        
        # Define common prefix patterns
        self.system_prompts = {
            "helpful_assistant": list(range(1000, 1050)),  # 50 tokens
            "code_assistant": list(range(2000, 2050)),      # 50 tokens
            "translation": list(range(3000, 3040)),         # 40 tokens
            "summarization": list(range(4000, 4060)),       # 60 tokens
            "creative_writing": list(range(5000, 5080)),    # 80 tokens
        }
        
        self.few_shot_examples = {
            "classification_3shot": list(range(10000, 10300)),  # 300 tokens
            "qa_5shot": list(range(11000, 11500)),             # 500 tokens
            "translation_2shot": list(range(12000, 12200)),     # 200 tokens
        }
        
        self.instruction_templates = {
            "analyze_data": list(range(20000, 20100)),      # 100 tokens
            "write_code": list(range(21000, 21150)),        # 150 tokens
            "explain_concept": list(range(22000, 22120)),   # 120 tokens
        }
    
    def generate_tokens_with_prefix(
        self, 
        prefix_type: str, 
        prefix_name: str, 
        variable_length: int
    ) -> List[int]:
        """Generate tokens with a specific prefix pattern.
        
        Args:
            prefix_type: Type of prefix ('system', 'few_shot', 'instruction')
            prefix_name: Name of the specific prefix pattern
            variable_length: Number of additional variable tokens
            
        Returns:
            List of token IDs
        """
        # Get the prefix tokens
        if prefix_type == "system":
            prefix_tokens = self.system_prompts.get(prefix_name, [])
        elif prefix_type == "few_shot":
            prefix_tokens = self.few_shot_examples.get(prefix_name, [])
        elif prefix_type == "instruction":
            prefix_tokens = self.instruction_templates.get(prefix_name, [])
        else:
            prefix_tokens = []
        
        # Generate variable tokens (simulating user input)
        variable_tokens = [
            random.randint(50000, 60000) for _ in range(variable_length)
        ]
        
        return prefix_tokens + variable_tokens
    
    def generate_mixed_prefix_tokens(
        self,
        components: List[Dict[str, Any]],
        variable_length: int
    ) -> List[int]:
        """Generate tokens with multiple prefix components.
        
        Args:
            components: List of dicts with 'type' and 'name' keys
            variable_length: Number of additional variable tokens
            
        Returns:
            List of token IDs
        """
        all_tokens = []
        
        for component in components:
            prefix_type = component['type']
            prefix_name = component['name']
            
            if prefix_type == "system":
                tokens = self.system_prompts.get(prefix_name, [])
            elif prefix_type == "few_shot":
                tokens = self.few_shot_examples.get(prefix_name, [])
            elif prefix_type == "instruction":
                tokens = self.instruction_templates.get(prefix_name, [])
            else:
                tokens = []
            
            all_tokens.extend(tokens)
        
        # Add variable tokens
        variable_tokens = [
            random.randint(50000, 60000) for _ in range(variable_length)
        ]
        all_tokens.extend(variable_tokens)
        
        return all_tokens
    
    def generate_random_tokens(self, length: int) -> List[int]:
        """Generate completely random tokens (no prefix)."""
        return [random.randint(1, 50000) for _ in range(length)]