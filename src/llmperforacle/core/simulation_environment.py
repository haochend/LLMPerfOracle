"""Core simulation environment wrapper around SimPy."""

import logging
from typing import Any, Callable, Dict, Optional

import simpy

logger = logging.getLogger(__name__)


class SimulationEnvironment:
    """Wrapper around simpy.Environment to manage global simulation state and execution.
    
    This class acts as the heartbeat of the entire virtual testing environment,
    providing discrete-event simulation capabilities and orchestrating the execution
    of simulation processes.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the simulation environment.
        
        Args:
            config: Simulation-specific configuration containing:
                - max_simulation_time: Maximum duration for the simulation in simulated seconds
                - random_seed (optional): Random seed for reproducibility
                - lod (optional): Level of Detail - "high" (default) or "medium"
        """
        self.env: simpy.Environment = simpy.Environment()
        self.config: Dict[str, Any] = config
        self.active_processes: list = []
        
        # Set random seed if provided
        if "random_seed" in config:
            import random
            import numpy as np
            seed = config["random_seed"]
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"Random seed set to: {seed}")
        
        # Set Level of Detail (LoD)
        self.lod = config.get("lod", "high")
        if self.lod not in ["high", "medium"]:
            raise ValueError(f"Invalid LoD setting: {self.lod}. Must be 'high' or 'medium'")
        logger.info(f"Level of Detail (LoD) set to: {self.lod}")
        
        logger.info("SimulationEnvironment initialized")
    
    def schedule_process(self, process_generator_func: Callable, *args, **kwargs) -> simpy.Process:
        """Schedule a main SimPy process (a generator function).
        
        Args:
            process_generator_func: A generator function that yields SimPy events
            *args: Positional arguments for the generator function
            **kwargs: Keyword arguments for the generator function
            
        Returns:
            The SimPy Process object
        """
        process = self.env.process(process_generator_func(*args, **kwargs))
        self.active_processes.append(process)
        logger.debug(f"Scheduled process: {process_generator_func.__name__}")
        return process
    
    def run(self) -> None:
        """Start and run the simulation.
        
        Runs until max_simulation_time is reached or no more events are scheduled.
        Handles exceptions during simulation and logs simulation progress.
        """
        max_simulation_time = self.config.get("max_simulation_time", float("inf"))
        
        logger.info(f"Starting simulation (max time: {max_simulation_time}s)")
        
        try:
            self.env.run(until=max_simulation_time)
            logger.info(f"Simulation completed successfully at time {self.env.now}")
        except Exception as e:
            logger.error(f"Error during simulation at time {self.env.now}: {e}")
            raise
        finally:
            logger.info(f"Simulation ended at time {self.env.now}")
    
    def now(self) -> float:
        """Get the current simulation time.
        
        Returns:
            Current simulation time in seconds
        """
        return self.env.now
    
    def get_simpy_env(self) -> simpy.Environment:
        """Provide access to the raw SimPy environment.
        
        This allows components that need direct SimPy features (e.g., creating
        simpy.Event, simpy.Store) to access the underlying environment.
        
        Returns:
            The SimPy Environment instance
        """
        return self.env
    
    def get_lod(self) -> str:
        """Get the Level of Detail setting.
        
        Returns:
            Level of Detail string ("high" or "medium")
        """
        return self.lod