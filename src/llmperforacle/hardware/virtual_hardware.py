"""Virtual hardware platform implementation for LLM serving simulation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import simpy

logger = logging.getLogger(__name__)


@dataclass
class HardwareResourceProfile:
    """Hardware configuration profile containing all component specifications."""
    
    compute_devices: List[Dict[str, Any]]
    memory_devices: List[Dict[str, Any]]
    network_links: List[Dict[str, Any]]


class VirtualComputeDevice:
    """Represents a single compute device (GPU or CPU) with simulated resources."""
    
    def __init__(
        self,
        device_id: str,
        device_type: str,
        simpy_env: simpy.Environment,
        config: Dict[str, Any],
        model_characteristics_db: Dict[str, Any],
    ):
        """Initialize a virtual compute device.
        
        Args:
            device_id: Unique identifier for the device
            device_type: Type of device ("GPU" or "CPU")
            simpy_env: SimPy environment instance
            config: Device configuration containing:
                - peak_tflops: Dict mapping precision to TFLOPS (e.g., {"fp16": 312})
                - memory_capacity_bytes: Total memory capacity in bytes
                - memory_gbps: Memory bandwidth in GB/s
                - processing_units: Number of parallel execution units
            model_characteristics_db: Database of model characteristics
        """
        self.device_id = device_id
        self.device_type = device_type
        self.simpy_env = simpy_env
        self.model_characteristics_db = model_characteristics_db
        
        # Performance characteristics
        self.peak_tflops = config.get("peak_tflops", {"fp16": 100, "int8": 200})
        self.memory_capacity_bytes = config.get("memory_capacity_bytes", 80_000_000_000)
        self.memory_gbps = config.get("memory_gbps", 1000)
        
        # SimPy resources for modeling contention
        processing_units = config.get("processing_units", 108)
        self.processing_units_resource = simpy.Resource(simpy_env, capacity=processing_units)
        
        # Memory resources
        self.memory_available_container = simpy.Container(
            simpy_env, capacity=self.memory_capacity_bytes, init=self.memory_capacity_bytes
        )
        self.memory_bandwidth_resource = simpy.Resource(simpy_env, capacity=1)
        
        logger.info(
            f"Initialized {device_type} {device_id}: "
            f"{self.peak_tflops} TFLOPS, {self.memory_capacity_bytes / 1e9:.1f} GB, "
            f"{self.memory_gbps} GB/s bandwidth"
        )


class VirtualNetworkLink:
    """Represents a network link between two points with bandwidth and latency."""
    
    def __init__(
        self,
        link_id: str,
        source_id: str,
        dest_id: str,
        simpy_env: simpy.Environment,
        config: Dict[str, Any],
    ):
        """Initialize a virtual network link.
        
        Args:
            link_id: Unique identifier for the link
            source_id: Source node identifier
            dest_id: Destination node identifier
            simpy_env: SimPy environment instance
            config: Link configuration containing:
                - bandwidth_bps: Bandwidth in bits per second
                - latency_s: Latency in seconds
        """
        self.link_id = link_id
        self.source_id = source_id
        self.dest_id = dest_id
        self.simpy_env = simpy_env
        
        self.bandwidth_bps = config.get("bandwidth_bps", 10_000_000_000)  # 10 Gbps default
        self.latency_s = config.get("latency_s", 0.0001)  # 100 µs default
        
        # Resource to model link utilization
        self.link_utilization_resource = simpy.Resource(simpy_env, capacity=1)
        
        logger.info(
            f"Initialized network link {link_id}: {source_id} -> {dest_id}, "
            f"{self.bandwidth_bps / 1e9:.1f} Gbps, {self.latency_s * 1e6:.1f} µs latency"
        )


class VirtualHardwarePlatform:
    """Manages all virtual hardware devices and provides resource request APIs."""
    
    # Efficiency factors for realistic hardware utilization
    # These represent typical achieved performance vs theoretical peak
    MFU_EFFICIENCY = 0.5  # 50% Model FLOPs Utilization (typical for LLMs)
    MBU_EFFICIENCY = 0.8  # 80% Memory Bandwidth Utilization (typical for memory-bound ops)
    
    def __init__(self):
        """Initialize an empty hardware platform."""
        self.simpy_env: Optional[simpy.Environment] = None
        self.compute_devices: Dict[str, VirtualComputeDevice] = {}
        self.network_links: Dict[Tuple[str, str], VirtualNetworkLink] = {}
        self.model_characteristics_db: Dict[str, Any] = {}
        
        # Instance-level efficiency factors (can be overridden)
        self.mfu_efficiency = self.MFU_EFFICIENCY
        self.mbu_efficiency = self.MBU_EFFICIENCY
    
    def initialize(
        self,
        profile: HardwareResourceProfile,
        simpy_env: simpy.Environment,
        model_chars_db: Dict[str, Any],
    ) -> None:
        """Initialize the platform with hardware profile and SimPy environment.
        
        Args:
            profile: Hardware resource profile with device configurations
            simpy_env: SimPy environment instance
            model_chars_db: Model characteristics database
        """
        self.simpy_env = simpy_env
        self.model_characteristics_db = model_chars_db
        
        # Initialize compute devices
        for device_config in profile.compute_devices:
            device = VirtualComputeDevice(
                device_id=device_config["device_id"],
                device_type=device_config["device_type"],
                simpy_env=simpy_env,
                config=device_config,
                model_characteristics_db=model_chars_db,
            )
            self.compute_devices[device.device_id] = device
        
        # Initialize network links
        for link_config in profile.network_links:
            link = VirtualNetworkLink(
                link_id=link_config["link_id"],
                source_id=link_config["source_id"],
                dest_id=link_config["dest_id"],
                simpy_env=simpy_env,
                config=link_config,
            )
            # Store with bidirectional access
            self.network_links[(link.source_id, link.dest_id)] = link
            if link_config.get("bidirectional", True):
                # Create reverse link with same properties
                reverse_link = VirtualNetworkLink(
                    link_id=f"{link_config['link_id']}_reverse",
                    source_id=link_config["dest_id"],
                    dest_id=link_config["source_id"],
                    simpy_env=simpy_env,
                    config=link_config,
                )
                self.network_links[(reverse_link.source_id, reverse_link.dest_id)] = reverse_link
        
        logger.info(
            f"VirtualHardwarePlatform initialized with {len(self.compute_devices)} "
            f"compute devices and {len(self.network_links)} network links"
        )
        logger.info(
            f"Hardware efficiency factors - MFU: {self.mfu_efficiency:.1%}, "
            f"MBU: {self.mbu_efficiency:.1%}"
        )
    
    def submit_computation_task(
        self, device_id: str, task_description: Dict[str, Any]
    ) -> simpy.Process:
        """Submit a computation task to a specific device.
        
        Args:
            device_id: Target device identifier
            task_description: Task specification containing:
                - task_id: Unique task identifier
                - flops_required_fp16: FLOPs required at FP16 precision
                - memory_read_bytes: Bytes to read from memory
                - memory_write_bytes: Bytes to write to memory
                - num_processing_units_requested: Number of units needed
                - is_memory_bound_hint: Whether task is memory-bound
        
        Returns:
            SimPy process that completes when the task finishes
        """
        return self.simpy_env.process(
            self._computation_task_process(device_id, task_description)
        )
    
    def _computation_task_process(
        self, device_id: str, task_description: Dict[str, Any]
    ) -> simpy.events.Event:
        """Internal process for executing a computation task."""
        device = self.compute_devices[device_id]
        task_id = task_description.get("task_id", "unknown")
        
        # Request processing units
        num_units = task_description.get("num_processing_units_requested", 1)
        with device.processing_units_resource.request() as proc_req:
            yield proc_req
            
            # Request memory bandwidth
            with device.memory_bandwidth_resource.request() as mem_req:
                yield mem_req
                
                # Calculate compute time with MFU efficiency
                flops_required = task_description.get("flops_required_fp16", 0)
                peak_tflops_fp16 = device.peak_tflops.get("fp16", 100)
                # Apply MFU efficiency factor to account for real-world utilization
                effective_tflops = peak_tflops_fp16 * self.mfu_efficiency
                compute_time_s = flops_required / (effective_tflops * 1e12)
                
                # Calculate memory access time with MBU efficiency
                memory_bytes = (
                    task_description.get("memory_read_bytes", 0) +
                    task_description.get("memory_write_bytes", 0)
                )
                # Apply MBU efficiency factor to account for real-world utilization
                effective_bandwidth_gbps = device.memory_gbps * self.mbu_efficiency
                memory_access_time_s = memory_bytes / (effective_bandwidth_gbps * 1e9)
                
                # Determine effective time (simplified Roofline model)
                is_memory_bound = task_description.get("is_memory_bound_hint", False)
                if is_memory_bound:
                    effective_time_s = memory_access_time_s
                else:
                    # Use max of compute and memory time
                    effective_time_s = max(compute_time_s, memory_access_time_s)
                
                # Simulate the task execution
                yield self.simpy_env.timeout(effective_time_s)
                
                # Calculate actual utilization metrics
                if effective_time_s > 0:
                    # MFU: Actual FLOPs achieved / Peak FLOPs possible in the time taken
                    actual_mfu = flops_required / (peak_tflops_fp16 * 1e12 * effective_time_s)
                    # MBU: Actual bandwidth achieved / Peak bandwidth
                    actual_mbu = memory_bytes / (device.memory_gbps * 1e9 * effective_time_s)
                else:
                    actual_mfu = 0.0
                    actual_mbu = 0.0
                
                logger.debug(
                    f"Task {task_id} completed on {device_id} in {effective_time_s:.6f}s "
                    f"(compute: {compute_time_s:.6f}s, memory: {memory_access_time_s:.6f}s) "
                    f"MFU: {actual_mfu:.1%}, MBU: {actual_mbu:.1%}"
                )
    
    def allocate_memory(self, device_id: str, size_bytes: int) -> simpy.Process:
        """Allocate memory on a device.
        
        Args:
            device_id: Target device identifier
            size_bytes: Amount of memory to allocate in bytes
            
        Returns:
            SimPy process that completes when memory is allocated
        """
        return self.simpy_env.process(self._allocate_memory_process(device_id, size_bytes))
    
    def _allocate_memory_process(self, device_id: str, size_bytes: int) -> simpy.events.Event:
        """Internal process for memory allocation."""
        device = self.compute_devices[device_id]
        yield device.memory_available_container.get(size_bytes)
        logger.debug(f"Allocated {size_bytes / 1e6:.1f} MB on {device_id}")
    
    def free_memory(self, device_id: str, size_bytes: int) -> simpy.Process:
        """Free memory on a device.
        
        Args:
            device_id: Target device identifier
            size_bytes: Amount of memory to free in bytes
            
        Returns:
            SimPy process that completes when memory is freed
        """
        return self.simpy_env.process(self._free_memory_process(device_id, size_bytes))
    
    def _free_memory_process(self, device_id: str, size_bytes: int) -> simpy.events.Event:
        """Internal process for memory deallocation."""
        device = self.compute_devices[device_id]
        yield device.memory_available_container.put(size_bytes)
        logger.debug(f"Freed {size_bytes / 1e6:.1f} MB on {device_id}")
    
    def submit_network_transfer_task(
        self, source_node_id: str, dest_node_id: str, data_size_bytes: int
    ) -> simpy.Process:
        """Submit a network transfer task between nodes.
        
        Args:
            source_node_id: Source node identifier
            dest_node_id: Destination node identifier
            data_size_bytes: Size of data to transfer in bytes
            
        Returns:
            SimPy process that completes when transfer finishes
        """
        return self.simpy_env.process(
            self._network_transfer_process(source_node_id, dest_node_id, data_size_bytes)
        )
    
    def _network_transfer_process(
        self, source_node_id: str, dest_node_id: str, data_size_bytes: int
    ) -> simpy.events.Event:
        """Internal process for network transfer."""
        link_key = (source_node_id, dest_node_id)
        link = self.network_links.get(link_key)
        
        if not link:
            logger.warning(
                f"No link found from {source_node_id} to {dest_node_id}, using default"
            )
            # Create a default link if not found
            yield self.simpy_env.timeout(0.001)  # 1ms default
            return
        
        with link.link_utilization_resource.request() as req:
            yield req
            
            # Apply latency
            yield self.simpy_env.timeout(link.latency_s)
            
            # Calculate transfer time
            transfer_time_s = (data_size_bytes * 8) / link.bandwidth_bps
            yield self.simpy_env.timeout(transfer_time_s)
            
            logger.debug(
                f"Network transfer {source_node_id} -> {dest_node_id}: "
                f"{data_size_bytes / 1e6:.1f} MB in {transfer_time_s:.6f}s"
            )
    
    def get_device_info(self, device_id: str) -> Optional[VirtualComputeDevice]:
        """Get information about a specific device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            VirtualComputeDevice object or None if not found
        """
        return self.compute_devices.get(device_id)
    
    def set_efficiency_factors(self, mfu: Optional[float] = None, mbu: Optional[float] = None):
        """Set custom efficiency factors for hardware utilization.
        
        Args:
            mfu: Model FLOPs Utilization (0.0 to 1.0)
            mbu: Memory Bandwidth Utilization (0.0 to 1.0)
        """
        if mfu is not None:
            if not 0.0 <= mfu <= 1.0:
                raise ValueError(f"MFU must be between 0.0 and 1.0, got {mfu}")
            self.mfu_efficiency = mfu
            logger.info(f"Updated MFU efficiency to {mfu:.1%}")
            
        if mbu is not None:
            if not 0.0 <= mbu <= 1.0:
                raise ValueError(f"MBU must be between 0.0 and 1.0, got {mbu}")
            self.mbu_efficiency = mbu
            logger.info(f"Updated MBU efficiency to {mbu:.1%}")