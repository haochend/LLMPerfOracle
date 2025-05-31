# Workload Generator

This module generates realistic LLM service request streams based on configurable statistical distributions and client profiles.

## Overview

The Workload Generator creates request patterns that mimic production LLM usage, including:
- Bursty arrival patterns
- Heterogeneous token length distributions
- Conversational context and multi-turn interactions
- Multiple client profiles with different behaviors

## Components

### WorkloadGenerator

The main SimPy process that generates and dispatches requests.

**Key Features:**
- Per-client composition with weighted profiles
- Statistical distribution sampling for request characteristics
- Conversational state tracking
- Network transfer simulation

### Request

Data class representing an LLM service request with:
- Unique identifiers (request_id, client_id, session_id)
- Token counts (prompt and max output)
- Conversational flags
- Streaming preferences
- User priority

### ClientProfile

Defines behavior patterns for different client types:
- Inter-arrival time distributions
- Token length distributions
- Conversation probability
- Streaming preferences

### DistributionSampler

Utility for sampling from various statistical distributions:
- Exponential, Gamma, Weibull (for arrival patterns)
- LogNormal, Pareto (for token lengths)
- Uniform, Normal, Constant
- Mixture distributions

## Configuration Example

```yaml
workload:
  total_duration: 300
  client_profiles:
    - profile_name: "interactive_chat"
      weight: 0.7
      inter_arrival_time_dist_config:
        type: "Exponential"
        rate: 5.0  # 5 req/s average
      prompt_tokens_dist_config:
        type: "LogNormal"
        mean: 128
        sigma: 64
        is_int: true
      max_output_tokens_dist_config:
        type: "Uniform"
        low: 100
        high: 500
        is_int: true
      conversational_probability: 0.6
      streaming_response_probability: 0.9
```

## Supported Distributions

- **Exponential**: For memoryless arrival patterns
- **Gamma/Weibull**: For bursty arrivals
- **LogNormal**: For token lengths with long tail
- **Pareto**: For heavy-tailed distributions
- **Mixture**: Combine multiple distributions

## Conversational Modeling

The generator tracks conversation sessions:
- Creates session IDs for related requests
- Manages follow-up timing
- Tracks conversation turn counts
- Simulates realistic multi-turn patterns
- **Token Accumulation**: Accumulates tokens across conversational turns to simulate growing context
  - Enable/disable with `accumulate_conversational_tokens` config (default: True)
  - Each turn includes: previous prompt + previous response + new user input
  - Allows proper testing of prefix caching benefits

### Token Accumulation Example

For a conversation with 3 turns:
- Turn 1: 100 prompt tokens, 50 output tokens
- Turn 2: 100 + 50 + 80 = 230 prompt tokens (accumulated)
- Turn 3: 230 + 60 + 100 = 390 prompt tokens (accumulated)

This realistic simulation enables frameworks to demonstrate prefix caching effectiveness.