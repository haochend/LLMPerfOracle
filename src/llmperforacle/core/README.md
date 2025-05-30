# Core Simulation Engine

This module provides the foundational discrete-event simulation capabilities for LLMPerfOracle.

## Overview

The Core Simulation Engine wraps SimPy to manage simulation time, event scheduling, and process orchestration. It acts as the heartbeat of the entire virtual testing environment.

## Components

### SimulationEnvironment

The main class that manages the simulation lifecycle.

**Key Features:**
- Wraps `simpy.Environment` for discrete-event simulation
- Manages global simulation state and configuration
- Provides controlled execution with configurable time limits
- Handles random seed setting for reproducible simulations

**Usage:**
```python
from llmperforacle.core import SimulationEnvironment

config = {
    "max_simulation_time": 300,  # 5 minutes
    "random_seed": 42
}

sim_env = SimulationEnvironment(config)
sim_env.schedule_process(my_process_generator)
sim_env.run()
```

## Key Methods

- `schedule_process(process_generator_func, *args, **kwargs)`: Schedule a SimPy process
- `run()`: Execute the simulation until completion or time limit
- `now()`: Get current simulation time
- `get_simpy_env()`: Access the underlying SimPy environment

## Configuration

The simulation engine accepts these configuration parameters:
- `max_simulation_time`: Maximum simulation duration in seconds
- `random_seed`: Optional seed for reproducibility

## Dependencies

- SimPy 4.0+
- NumPy (for random seed management)