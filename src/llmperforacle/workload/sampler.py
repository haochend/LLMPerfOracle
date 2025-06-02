"""Statistical distribution sampler for workload generation."""

import logging
import random
from typing import Dict, Optional, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DistributionSampler:
    """Provides methods to sample values from various statistical distributions."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the sampler with optional random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def sample(self, distribution_config: Dict[str, any]) -> Union[float, int]:
        """Sample a value from the specified distribution.
        
        Args:
            distribution_config: Configuration dict with 'type' and distribution parameters.
                Examples:
                - {'type': 'Exponential', 'rate': 5.0}
                - {'type': 'LogNormal', 'mean': 200, 'sigma': 50, 'is_int': True}
                - {'type': 'Uniform', 'low': 10, 'high': 100, 'is_int': True}
                
        Returns:
            Sampled value (float or int based on 'is_int' parameter)
        """
        dist_type = distribution_config.get("type", "Constant")
        is_int = distribution_config.get("is_int", False)
        
        try:
            if dist_type == "Constant":
                value = distribution_config.get("value", 1.0)
            
            elif dist_type == "Exponential":
                rate = distribution_config.get("rate", 1.0)
                value = np.random.exponential(1.0 / rate)
            
            elif dist_type == "Uniform":
                low = distribution_config.get("low", 0.0)
                high = distribution_config.get("high", 1.0)
                value = np.random.uniform(low, high)
            
            elif dist_type == "Normal":
                mean = distribution_config.get("mean", 0.0)
                # Support both 'std' and 'sigma' parameter names
                std = distribution_config.get("std", distribution_config.get("sigma", 1.0))
                value = np.random.normal(mean, std)
                # Ensure non-negative for most use cases
                value = max(0, value)
            
            elif dist_type == "LogNormal":
                # Parameters are for the underlying normal distribution
                # mean and sigma are the parameters of the log-space normal distribution
                mean = distribution_config.get("mean", 0.0)
                sigma = distribution_config.get("sigma", 1.0)
                value = np.random.lognormal(mean, sigma)
            
            elif dist_type == "Pareto":
                shape = distribution_config.get("shape", 1.0)  # alpha parameter
                scale = distribution_config.get("scale", 1.0)  # xm parameter
                value = (np.random.pareto(shape) + 1) * scale
            
            elif dist_type == "Weibull":
                shape = distribution_config.get("shape", 1.0)  # k parameter
                scale = distribution_config.get("scale", 1.0)  # lambda parameter
                value = scale * np.random.weibull(shape)
            
            elif dist_type == "Gamma":
                shape = distribution_config.get("shape", 2.0)  # k or alpha
                scale = distribution_config.get("scale", 1.0)  # theta or beta
                value = np.random.gamma(shape, scale)
            
            elif dist_type == "Mixture":
                # Support for mixture distributions (e.g., Pareto + LogNormal)
                components = distribution_config.get("components", [])
                weights = distribution_config.get("weights", None)
                
                if not components:
                    logger.warning("Mixture distribution has no components, returning 1.0")
                    return 1.0
                
                if weights is None:
                    weights = [1.0 / len(components)] * len(components)
                
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Select component
                component_idx = np.random.choice(len(components), p=weights)
                component_config = components[component_idx]
                
                # Recursively sample from selected component
                value = self.sample(component_config)
            
            elif dist_type == "Fixed":
                # Fixed/constant value (alias for Constant)
                value = distribution_config.get("value", 1.0)
            
            else:
                logger.warning(f"Unknown distribution type: {dist_type}, using constant value 1.0")
                value = 1.0
            
            # Convert to int if requested
            if is_int:
                value = int(round(value))
                # Ensure at least 1 for counts
                value = max(1, value)
            
            return value
        
        except Exception as e:
            logger.error(f"Error sampling from distribution {dist_type}: {e}")
            return 1.0 if not is_int else 1