"""Experiment orchestrator for managing simulation execution."""

import json
import logging
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional

import yaml

from ..core import SimulationEnvironment
from ..frameworks import AbstractLLMFramework, VLLMFramework
from ..frameworks.parallel_vllm_framework import ParallelVLLMFramework
from ..hardware import HardwareResourceProfile, VirtualHardwarePlatform
from ..metrics import MetricsCollector
from ..workload import WorkloadGenerator

logger = logging.getLogger(__name__)

# Framework class mapping
FRAMEWORK_CLASS_MAP = {
    "VLLM": VLLMFramework,
    "ParallelVLLM": ParallelVLLMFramework,
    # Add more framework mappings here as they are implemented
    # "TRTLLM": TRTLLMFramework,
    # "SGLang": SGLangFramework,
    # "Dynama": DynamaFramework,
}


class ExperimentOrchestrator:
    """Main entry point to set up and run simulation experiments."""
    
    def __init__(self, config_data: Dict[str, Any]):
        """Initialize the orchestrator with experiment configuration.
        
        Args:
            config_data: Complete experiment configuration dictionary
        """
        self.config = config_data
        self._validate_config()
        
        # Component instances (initialized in setup_simulation)
        self.sim_env_wrapper: Optional[SimulationEnvironment] = None
        self.virtual_hardware: Optional[VirtualHardwarePlatform] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.workload_generator: Optional[WorkloadGenerator] = None
        self.llm_framework_instances: List[AbstractLLMFramework] = []
        self.model_characteristics_db: Dict[str, Any] = {}
        
        # Load model characteristics database
        self._load_model_characteristics()
        
        logger.info("ExperimentOrchestrator initialized")
    
    def _validate_config(self) -> None:
        """Validate the experiment configuration structure."""
        required_sections = [
            "simulation",
            "hardware_profile",
            "model_characteristics_db_path",
            "workload",
            "frameworks_to_test",
            "metrics_config",
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate simulation config
        if "max_simulation_time" not in self.config["simulation"]:
            raise ValueError("simulation.max_simulation_time is required")
        
        # Validate frameworks
        if not self.config["frameworks_to_test"]:
            raise ValueError("At least one framework must be configured")
        
        logger.info("Configuration validated successfully")
    
    def _load_model_characteristics(self) -> None:
        """Load the model characteristics database."""
        db_path = Path(self.config["model_characteristics_db_path"])
        
        if not db_path.exists():
            raise FileNotFoundError(f"Model characteristics database not found: {db_path}")
        
        with open(db_path, "r") as f:
            self.model_characteristics_db = json.load(f)
        
        logger.info(f"Loaded {len(self.model_characteristics_db)} model profiles")
    
    def setup_simulation(self) -> None:
        """Initialize all simulation components."""
        logger.info("Setting up simulation components...")
        
        # 1. Initialize Core Simulation Engine
        self.sim_env_wrapper = SimulationEnvironment(self.config["simulation"])
        simpy_env = self.sim_env_wrapper.get_simpy_env()
        
        # Add metadata to simpy_env for framework access
        simpy_env.metadata = {
            'lod': self.sim_env_wrapper.get_lod()
        }
        
        # 2. Initialize Metrics Collector
        self.metrics_collector = MetricsCollector(simpy_env, self.config["metrics_config"])
        
        # 3. Initialize Virtual Hardware Platform
        self.virtual_hardware = VirtualHardwarePlatform()
        hardware_profile = HardwareResourceProfile(
            compute_devices=self.config["hardware_profile"]["compute_devices"],
            memory_devices=self.config["hardware_profile"].get("memory_devices", []),
            network_links=self.config["hardware_profile"]["network_links"],
        )
        self.virtual_hardware.initialize(
            hardware_profile, simpy_env, self.model_characteristics_db
        )
        
        # 4. Initialize LLM Framework Instances
        self.llm_framework_instances = []
        for fw_config in self.config["frameworks_to_test"]:
            framework_instance = self._create_framework_instance(fw_config, simpy_env)
            self.llm_framework_instances.append(framework_instance)
        
        # 5. Initialize Workload Generator
        target_frameworks_map = self._create_target_frameworks_map()
        self.workload_generator = WorkloadGenerator(
            simpy_env,
            self.config["workload"],
            target_frameworks_map,
            self.metrics_collector,
            self.virtual_hardware,
        )
        
        logger.info("Simulation setup complete")
    
    def _create_framework_instance(
        self, fw_config: Dict[str, Any], simpy_env: Any
    ) -> AbstractLLMFramework:
        """Create a framework instance from configuration."""
        framework_type = fw_config["type"]
        framework_id = fw_config["name"]
        
        # Get framework class
        FrameworkClass = FRAMEWORK_CLASS_MAP.get(framework_type)
        if not FrameworkClass:
            raise ValueError(f"Unknown framework type: {framework_type}")
        
        # Get model profile
        model_profile_id = fw_config["config"].get("model_profile_id")
        if not model_profile_id:
            raise ValueError(f"model_profile_id not specified for framework {framework_id}")
        
        model_profile = self.model_characteristics_db.get(model_profile_id)
        if not model_profile:
            raise ValueError(
                f"Model profile {model_profile_id} not found for framework {framework_id}"
            )
        
        # Create instance
        instance = FrameworkClass(
            framework_id=framework_id,
            simpy_env=simpy_env,
            framework_specific_config=fw_config["config"],
            virtual_hardware=self.virtual_hardware,
            metrics_collector=self.metrics_collector,
            model_profile=model_profile,
        )
        
        logger.info(f"Created {framework_type} instance: {framework_id}")
        return instance
    
    def _create_target_frameworks_map(self) -> Dict[str, Any]:
        """Create mapping of framework IDs to request queues and entry devices."""
        target_map = {}
        
        for i, fw_instance in enumerate(self.llm_framework_instances):
            # Find the original config to check if it's a target
            fw_config = next(
                fc for fc in self.config["frameworks_to_test"]
                if fc["name"] == fw_instance.framework_id
            )
            
            if fw_config.get("is_target_for_workload", True):
                # Determine the entry device for this framework
                if hasattr(fw_instance, 'primary_gpu_id'):
                    entry_device = fw_instance.primary_gpu_id
                elif hasattr(fw_instance, 'gpu_id'):
                    entry_device = fw_instance.gpu_id
                else:
                    entry_device = fw_instance.gpu_ids[0] if fw_instance.gpu_ids else 'gpu0'
                
                # Store both queue and entry device
                target_map[fw_instance.framework_id] = {
                    'queue': fw_instance.request_arrival_queue,
                    'entry_device': entry_device
                }
        
        if not target_map:
            raise ValueError("No frameworks marked as workload targets")
        
        return target_map
    
    def run(self) -> Dict[str, Any]:
        """Run the complete simulation experiment.
        
        Returns:
            Summary report dictionary
        """
        # Setup if not already done
        if self.sim_env_wrapper is None:
            self.setup_simulation()
        
        logger.info("=" * 60)
        logger.info("STARTING SIMULATION EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Configuration: {pformat(self.config)}")
        
        # Start LLM Framework Processing Loops
        for fw_instance in self.llm_framework_instances:
            self.sim_env_wrapper.schedule_process(fw_instance.processing_loop)
            logger.info(f"Started processing loop for {fw_instance.framework_id}")
        
        # Start Workload Generator
        self.sim_env_wrapper.schedule_process(self.workload_generator.generate_requests_process)
        logger.info("Started workload generator")
        
        # Run the simulation
        self.sim_env_wrapper.run()
        
        # Generate and save reports
        simulation_duration = self.sim_env_wrapper.now()
        summary_report = self.metrics_collector.generate_summary_report(simulation_duration)
        
        # Save summary report if configured
        summary_path = self.config["metrics_config"].get("output_summary_json_path")
        if summary_path:
            summary_file = Path(summary_path)
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_file, "w") as f:
                json.dump(summary_report, f, indent=2)
            logger.info(f"Saved summary report to {summary_file}")
        
        # Save detailed metrics if configured
        csv_path = self.config["metrics_config"].get("output_requests_csv_path")
        if csv_path and hasattr(self.metrics_collector, "get_all_request_metrics_df"):
            csv_file = Path(csv_path)
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            df = self.metrics_collector.get_all_request_metrics_df()
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved detailed metrics to {csv_file}")
        
        logger.info("=" * 60)
        logger.info("SIMULATION EXPERIMENT COMPLETED")
        logger.info("=" * 60)
        
        return summary_report
    
    @classmethod
    def from_yaml_file(cls, config_path: str) -> "ExperimentOrchestrator":
        """Create an orchestrator from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ExperimentOrchestrator instance
        """
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        return cls(config_data)
    
    @classmethod
    def from_json_file(cls, config_path: str) -> "ExperimentOrchestrator":
        """Create an orchestrator from a JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            ExperimentOrchestrator instance
        """
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        return cls(config_data)