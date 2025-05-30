Document 1: Core Simulation Engine
1. Purpose:
To provide the foundational discrete-event simulation capabilities, manage simulation time, and orchestrate the execution of simulation processes (events). This module acts as the "heartbeat" of the entire virtual testing environment.
2. Underlying Technology:
* SimPy (Python-based discrete-event simulation framework)
3. Key Components/Classes:
* SimulationEnvironment Class:
   * Purpose: A wrapper around simpy.Environment to manage global simulation state, configuration, and execution control.
   * Attributes:
      * env: simpy.Environment: The core SimPy environment instance.
      * config: dict: Simulation-specific configurations (e.g., max_simulation_time).
      * active_processes: list: (Optional) A list to keep track of major SimPy processes if needed for explicit management beyond SimPy's own scheduling.
   * Methods:
      * __init__(self, config: dict):
         * Initializes self.env = simpy.Environment().
         * Stores self.config.
      * schedule_process(self, process_generator_func, *args, **kwargs):
         * Purpose: Helper to schedule a main SimPy process (a generator function).
         * Logic: Calls self.env.process(process_generator_func(*args, **kwargs)).
      * run(self):
         * Purpose: Starts and runs the simulation.
         * Logic:
            1. Retrieves max_simulation_time from self.config.
            2. Calls self.env.run(until=max_simulation_time).
            3. Handles potential exceptions during simulation.
            4. Logs simulation start and end.
      * now(self) -> float:
         * Purpose: Returns the current simulation time.
         * Logic: Returns self.env.now.
      * get_simpy_env(self) -> simpy.Environment:
         * Purpose: Provides access to the raw SimPy environment for components that need to directly use SimPy features (e.g., creating simpy.Event, simpy.Store).
         * Logic: Returns self.env.
4. Core Logic:
* Simulation Initialization: The ExperimentOrchestrator (Document 6) will instantiate SimulationEnvironment.
* Event Scheduling: Primarily handled by SimPy's internal mechanisms when yield statements are used within SimPy processes (e.g., yield env.timeout(delay), yield resource.request()).
* Simulation Execution Loop: Managed by simpy.Environment.run().
* Termination Conditions:
   * Simulation reaches max_simulation_time.
   * No more events scheduled in the SimPy environment.
5. Interfaces/APIs (as consumed by other modules like ExperimentOrchestrator):
* SimulationEnvironment(config): Constructor.
* sim_env.schedule_process(process_func, ...): To start top-level processes like the workload generator or framework processing loops.
* sim_env.run(): To execute the simulation.
* sim_env.now(): For any component to get the current simulation time.
* sim_env.get_simpy_env(): To pass the SimPy environment to components that need it for creating SimPy-specific entities like events, resources, stores, or processes.
6. Configuration Parameters (passed to SimulationEnvironment constructor):
* max_simulation_time: float: The total duration the simulation should run in simulated seconds.
* (Potentially others like random_seed if global seed management is centralized here, though often SimPy processes manage their own randomness).
7. Dependencies:
* SimPy library.