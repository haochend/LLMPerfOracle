"""
Configuration validation system to prevent parameter-related issues.

This module provides comprehensive validation for:
- Model configurations
- Hardware configurations
- Framework configurations
- Experiment configurations
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ModelConfigValidator:
    """Validates model configuration parameters."""
    
    REQUIRED_FIELDS = {
        'parameters',
        'hidden_size',
        'num_layers',
        'num_attention_heads',
        'num_kv_heads',
        'intermediate_size',
        'vocab_size'
    }
    
    REQUIRED_STATS = {
        'prefill_op_stats': {'flops_per_token', 'memory_bytes_per_token'},
        'decode_op_stats': {'flops_per_token', 'memory_bytes_per_token'}
    }
    
    @classmethod
    def validate(cls, model_name: str, config: Dict[str, Any]) -> List[str]:
        """Validate a single model configuration."""
        errors = []
        
        # Check required fields
        missing_fields = cls.REQUIRED_FIELDS - set(config.keys())
        if missing_fields:
            errors.append(f"Model {model_name} missing required fields: {missing_fields}")
        
        # Check parameter_bytes_fp16
        if 'parameter_bytes_fp16' not in config:
            if 'parameters' in config:
                # Auto-fix by calculating
                config['parameter_bytes_fp16'] = config['parameters'] * 2
                logger.warning(f"Model {model_name}: Added missing parameter_bytes_fp16 field")
            else:
                errors.append(f"Model {model_name}: Missing both parameters and parameter_bytes_fp16")
        else:
            # Verify consistency
            if 'parameters' in config:
                expected_bytes = config['parameters'] * 2
                if abs(config['parameter_bytes_fp16'] - expected_bytes) > 1000:
                    errors.append(
                        f"Model {model_name}: Inconsistent parameter_bytes_fp16 "
                        f"({config['parameter_bytes_fp16']} != {expected_bytes})"
                    )
        
        # Check operation stats
        for stat_type, required_fields in cls.REQUIRED_STATS.items():
            if stat_type not in config:
                errors.append(f"Model {model_name}: Missing {stat_type}")
            else:
                missing = required_fields - set(config[stat_type].keys())
                if missing:
                    errors.append(f"Model {model_name}: {stat_type} missing fields: {missing}")
        
        # Check KV cache configuration
        if 'kv_cache_bytes_per_token_per_layer' not in config:
            # Calculate from model dimensions
            if all(k in config for k in ['hidden_size', 'num_kv_heads', 'num_layers']):
                kv_dim = config['hidden_size'] * config['num_kv_heads'] // config['num_attention_heads']
                kv_bytes = 2 * kv_dim * 2  # 2 for K,V and 2 bytes for FP16
                config['kv_cache_bytes_per_token_per_layer'] = kv_bytes
                logger.warning(f"Model {model_name}: Added missing kv_cache_bytes_per_token_per_layer")
            else:
                errors.append(f"Model {model_name}: Cannot calculate KV cache size")
        
        # Validate layer types if present
        if 'layer_types' in config:
            for layer_type in ['attention', 'mlp']:
                if layer_type not in config['layer_types']:
                    errors.append(f"Model {model_name}: Missing layer type '{layer_type}'")
        
        # Check aggregated ops for LoD support
        if 'aggregated_ops' not in config:
            logger.warning(f"Model {model_name}: Missing aggregated_ops for LoD support")
            # Auto-generate if possible
            if all(k in config for k in ['prefill_op_stats', 'decode_op_stats']):
                config['aggregated_ops'] = {
                    'prefill': {
                        'total_flops_per_prompt_token': config['prefill_op_stats']['flops_per_token'],
                        'total_memory_bytes_per_prompt_token': config['prefill_op_stats']['memory_bytes_per_token'],
                        'critical_path_factor': 1.0
                    },
                    'decode': {
                        'total_flops_per_token_in_batch': config['decode_op_stats']['flops_per_token'],
                        'total_memory_bytes_per_token_in_batch': config['decode_op_stats']['memory_bytes_per_token'],
                        'critical_path_factor': 1.0
                    }
                }
        
        return errors
    
    @classmethod
    def validate_all(cls, model_db: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate all models in the database."""
        all_errors = []
        
        for model_name, config in model_db.items():
            errors = cls.validate(model_name, config)
            all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors


class HardwareConfigValidator:
    """Validates hardware configuration."""
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> List[str]:
        """Validate hardware configuration."""
        errors = []
        
        # Check compute devices
        if 'compute_devices' not in config:
            errors.append("Missing compute_devices in hardware config")
            return errors
        
        devices = config['compute_devices']
        device_ids = set()
        
        for device in devices:
            # Check required fields
            required = {'device_id', 'device_type', 'peak_tflops', 'memory_capacity_bytes', 'memory_gbps'}
            missing = required - set(device.keys())
            if missing:
                errors.append(f"Device {device.get('device_id', 'unknown')} missing fields: {missing}")
            
            # Check for duplicate IDs
            dev_id = device.get('device_id')
            if dev_id in device_ids:
                errors.append(f"Duplicate device ID: {dev_id}")
            device_ids.add(dev_id)
            
            # Validate memory capacity
            if 'memory_capacity_bytes' in device:
                mem_gb = device['memory_capacity_bytes'] / 1e9
                if mem_gb < 1:
                    errors.append(f"Device {dev_id}: Unrealistic memory capacity {mem_gb:.1f} GB")
            
            # Validate peak_tflops
            if 'peak_tflops' in device:
                if isinstance(device['peak_tflops'], dict):
                    if 'fp16' not in device['peak_tflops']:
                        errors.append(f"Device {dev_id}: Missing fp16 in peak_tflops")
                else:
                    errors.append(f"Device {dev_id}: peak_tflops should be a dict with fp16/int8 keys")
        
        # Check network links
        if 'network_links' in config:
            cls._validate_network_links(config['network_links'], device_ids, errors)
        
        return errors
    
    @classmethod
    def _validate_network_links(cls, links: List[Dict[str, Any]], device_ids: Set[str], errors: List[str]) -> None:
        """Validate network links configuration."""
        link_pairs = set()
        
        for link in links:
            # Check required fields
            required = {'link_id', 'source_id', 'dest_id', 'bandwidth_bps', 'latency_s'}
            missing = required - set(link.keys())
            if missing:
                errors.append(f"Link {link.get('link_id', 'unknown')} missing fields: {missing}")
                continue
            
            src, dst = link['source_id'], link['dest_id']
            
            # Check if devices exist (allow client_node_0 as special case)
            if src != 'client_node_0' and src not in device_ids:
                errors.append(f"Link {link['link_id']}: Unknown source device {src}")
            if dst != 'client_node_0' and dst not in device_ids:
                errors.append(f"Link {link['link_id']}: Unknown destination device {dst}")
            
            # Track link pairs
            link_pairs.add((src, dst))
            if link.get('bidirectional', False):
                link_pairs.add((dst, src))
        
        # Check for missing reverse links (warning only)
        for src, dst in list(link_pairs):
            if (dst, src) not in link_pairs and src != 'client_node_0' and dst != 'client_node_0':
                logger.warning(f"Missing reverse link from {dst} to {src}")


class FrameworkConfigValidator:
    """Validates framework configuration."""
    
    @classmethod
    def validate(cls, fw_config: Dict[str, Any], hardware_config: Dict[str, Any], 
                 model_db: Dict[str, Any]) -> List[str]:
        """Validate framework configuration."""
        errors = []
        
        # Check framework type
        if 'type' not in fw_config:
            errors.append("Missing framework type")
            return errors
        
        fw_type = fw_config['type']
        config = fw_config.get('config', {})
        
        # Check model profile
        model_id = config.get('model_profile_id')
        if not model_id:
            errors.append("Missing model_profile_id in framework config")
        elif model_id not in model_db:
            errors.append(f"Unknown model profile: {model_id}")
        
        # Type-specific validation
        if fw_type == 'VLLM':
            errors.extend(cls._validate_vllm_config(config, hardware_config))
        elif fw_type == 'ParallelVLLM':
            errors.extend(cls._validate_parallel_vllm_config(config, hardware_config))
        
        return errors
    
    @classmethod
    def _validate_vllm_config(cls, config: Dict[str, Any], hardware: Dict[str, Any]) -> List[str]:
        """Validate VLLM-specific configuration."""
        errors = []
        
        # Check GPU assignment
        gpu_id = config.get('gpu_id')
        if not gpu_id:
            errors.append("VLLM config missing gpu_id")
        else:
            device_ids = {d['device_id'] for d in hardware.get('compute_devices', [])}
            if gpu_id not in device_ids:
                errors.append(f"VLLM gpu_id '{gpu_id}' not found in hardware")
        
        # Validate block size
        block_size = config.get('block_size', 16)
        if block_size <= 0 or block_size > 256:
            errors.append(f"Invalid block_size: {block_size} (should be 1-256)")
        
        # Validate max_num_seqs
        max_seqs = config.get('max_num_seqs', 256)
        if max_seqs <= 0:
            errors.append(f"Invalid max_num_seqs: {max_seqs}")
        
        return errors
    
    @classmethod
    def _validate_parallel_vllm_config(cls, config: Dict[str, Any], hardware: Dict[str, Any]) -> List[str]:
        """Validate ParallelVLLM-specific configuration."""
        errors = []
        
        # First validate base VLLM config (excluding gpu_id for parallel)
        base_errors = cls._validate_vllm_config({k: v for k, v in config.items() if k != 'gpu_id'}, hardware)
        errors.extend([e for e in base_errors if 'gpu_id' not in e])
        
        # Check parallelism config
        parallelism = config.get('parallelism', {})
        if not parallelism:
            errors.append("ParallelVLLM missing parallelism config")
            return errors
        
        strategy = parallelism.get('strategy')
        if strategy not in ['TP', 'PP', 'TP_PP']:
            errors.append(f"Invalid parallelism strategy: {strategy}")
        
        gpu_ids = parallelism.get('gpu_ids', [])
        if not gpu_ids:
            errors.append("ParallelVLLM missing gpu_ids")
        else:
            device_ids = {d['device_id'] for d in hardware.get('compute_devices', [])}
            for gpu_id in gpu_ids:
                if gpu_id not in device_ids:
                    errors.append(f"ParallelVLLM gpu_id '{gpu_id}' not found in hardware")
        
        # Strategy-specific validation
        if 'TP' in strategy:
            tp_degree = parallelism.get('tp_degree', 1)
            if tp_degree <= 0:
                errors.append(f"Invalid tp_degree: {tp_degree}")
            if 'PP' not in strategy and tp_degree != len(gpu_ids):
                errors.append(f"TP degree {tp_degree} doesn't match GPU count {len(gpu_ids)}")
        
        if 'PP' in strategy:
            pp_stages = parallelism.get('pp_stages', 1)
            if pp_stages <= 0:
                errors.append(f"Invalid pp_stages: {pp_stages}")
            
            num_microbatches = parallelism.get('num_microbatches_per_request', 1)
            if num_microbatches <= 0:
                errors.append(f"Invalid num_microbatches_per_request: {num_microbatches}")
            
            # For TP+PP, check total GPU count
            if strategy == 'TP_PP':
                tp_degree = parallelism.get('tp_degree', 1)
                expected_gpus = tp_degree * pp_stages
                if len(gpu_ids) != expected_gpus:
                    errors.append(f"TP_PP expects {expected_gpus} GPUs (tp={tp_degree} * pp={pp_stages}), got {len(gpu_ids)}")
        
        return errors


class ExperimentConfigValidator:
    """Validates complete experiment configuration."""
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate complete experiment configuration."""
        all_errors = []
        
        # Check top-level required fields
        required_top = {'simulation', 'hardware_profile', 'workload', 'frameworks_to_test'}
        missing_top = required_top - set(config.keys())
        if missing_top:
            all_errors.append(f"Missing top-level fields: {missing_top}")
            return False, all_errors
        
        # Load model database if specified
        model_db = {}
        if 'model_characteristics_db_path' in config:
            model_path = Path(config['model_characteristics_db_path'])
            if not model_path.exists():
                all_errors.append(f"Model database not found: {model_path}")
            else:
                import json
                with open(model_path) as f:
                    model_db = json.load(f)
                
                # Validate model database
                valid, errors = ModelConfigValidator.validate_all(model_db)
                all_errors.extend(errors)
        
        # Validate hardware configuration
        hw_errors = HardwareConfigValidator.validate(config['hardware_profile'])
        all_errors.extend(hw_errors)
        
        # Validate each framework
        for fw_config in config.get('frameworks_to_test', []):
            fw_errors = FrameworkConfigValidator.validate(fw_config, config['hardware_profile'], model_db)
            all_errors.extend(fw_errors)
        
        # Validate workload configuration
        workload_errors = cls._validate_workload(config.get('workload', {}))
        all_errors.extend(workload_errors)
        
        # Validate simulation configuration
        sim_errors = cls._validate_simulation(config.get('simulation', {}))
        all_errors.extend(sim_errors)
        
        return len(all_errors) == 0, all_errors
    
    @classmethod
    def _validate_workload(cls, workload: Dict[str, Any]) -> List[str]:
        """Validate workload configuration."""
        errors = []
        
        if 'total_duration' not in workload:
            errors.append("Workload missing total_duration")
        elif workload['total_duration'] <= 0:
            errors.append(f"Invalid workload total_duration: {workload['total_duration']}")
        
        if 'client_profiles' not in workload:
            errors.append("Workload missing client_profiles")
        else:
            for i, profile in enumerate(workload['client_profiles']):
                # Check distribution configs
                for dist_field in ['inter_arrival_time_dist_config', 'prompt_tokens_dist_config', 
                                 'max_output_tokens_dist_config']:
                    if dist_field not in profile:
                        errors.append(f"Client profile {i} missing {dist_field}")
                    else:
                        dist = profile[dist_field]
                        if 'type' not in dist:
                            errors.append(f"Client profile {i} {dist_field} missing type")
        
        return errors
    
    @classmethod
    def _validate_simulation(cls, simulation: Dict[str, Any]) -> List[str]:
        """Validate simulation configuration."""
        errors = []
        
        if 'max_simulation_time' not in simulation:
            errors.append("Simulation missing max_simulation_time")
        elif simulation['max_simulation_time'] <= 0:
            errors.append(f"Invalid max_simulation_time: {simulation['max_simulation_time']}")
        
        # Check LoD if specified
        if 'lod' in simulation:
            if simulation['lod'] not in ['high', 'medium', 'low']:
                errors.append(f"Invalid LoD: {simulation['lod']} (must be high/medium/low)")
        
        return errors


def validate_and_fix_config(config_path: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Load, validate, and attempt to fix a configuration file.
    
    Returns:
        (is_valid, errors, fixed_config)
    """
    import json
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    
    with open(config_path) as f:
        if config_file.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Validate
    is_valid, errors = ExperimentConfigValidator.validate(config)
    
    if not is_valid:
        logger.warning(f"Configuration has {len(errors)} validation errors")
        for error in errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")
    
    return is_valid, errors, config


def validate_model_db(model_db_path: str) -> Tuple[bool, List[str]]:
    """Validate a model database file."""
    import json
    
    with open(model_db_path) as f:
        model_db = json.load(f)
    
    return ModelConfigValidator.validate_all(model_db)