"""Command-line interface for LLMPerfOracle."""

import logging
import sys
from pathlib import Path

import click

from llmperforacle.orchestration import ExperimentOrchestrator
from llmperforacle.utils.config_validator import validate_and_fix_config, validate_model_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.group()
@click.version_option(version="0.1.0", prog_name="LLMPerfOracle")
def cli():
    """LLMPerfOracle: Virtual testing environment for LLM serving frameworks."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--format", "-f", type=click.Choice(["yaml", "json"]), default="yaml",
    help="Configuration file format"
)
@click.option(
    "--log-level", "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def run(config_file: str, format: str, log_level: str):
    """Run a simulation experiment from a configuration file."""
    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    click.echo(f"Loading configuration from {config_file}...")
    
    try:
        # Create orchestrator based on file format
        if format == "yaml":
            orchestrator = ExperimentOrchestrator.from_yaml_file(config_file)
        else:
            orchestrator = ExperimentOrchestrator.from_json_file(config_file)
        
        # Run the experiment
        click.echo("Starting simulation...")
        summary = orchestrator.run()
        
        # Print key results
        click.echo("\nSimulation completed!")
        click.echo(f"Total requests: {summary['requests']['total']}")
        click.echo(f"Success rate: {summary['requests']['success_rate']:.1%}")
        click.echo(f"Throughput: {summary['throughput']['output_tokens_per_second']:.1f} tokens/s")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output", "-o", default="example_config.yaml",
    help="Output file path"
)
@click.option(
    "--format", "-f", type=click.Choice(["yaml", "json"]), default="yaml",
    help="Configuration file format"
)
def generate_config(output: str, format: str):
    """Generate an example configuration file."""
    example_config = {
        "simulation": {
            "max_simulation_time": 300,
            "random_seed": 42,
        },
        "model_characteristics_db_path": "./configs/model_params.json",
        "hardware_profile": {
            "compute_devices": [
                {
                    "device_id": "gpu0",
                    "device_type": "GPU",
                    "peak_tflops": {"fp16": 312, "int8": 624},
                    "memory_capacity_bytes": 80_000_000_000,
                    "memory_gbps": 2000,
                    "processing_units": 108,
                }
            ],
            "network_links": [
                {
                    "link_id": "client_to_fw",
                    "source_id": "client_node_0",
                    "dest_id": "framework_entry_0",
                    "bandwidth_bps": 10_000_000_000,
                    "latency_s": 0.0001,
                }
            ],
        },
        "workload": {
            "total_duration": 300,
            "bytes_per_token_estimate_for_network": 2,
            "random_seed": 123,
            "client_profiles": [
                {
                    "profile_name": "interactive_chat",
                    "weight": 1.0,
                    "inter_arrival_time_dist_config": {
                        "type": "Exponential",
                        "rate": 10.0,
                    },
                    "prompt_tokens_dist_config": {
                        "type": "LogNormal",
                        "mean": 50,
                        "sigma": 20,
                        "is_int": True,
                    },
                    "max_output_tokens_dist_config": {
                        "type": "Uniform",
                        "low": 50,
                        "high": 200,
                        "is_int": True,
                    },
                    "conversational_probability": 0.3,
                    "streaming_response_probability": 0.9,
                }
            ],
        },
        "frameworks_to_test": [
            {
                "name": "vllm_instance_1",
                "type": "VLLM",
                "is_target_for_workload": True,
                "config": {
                    "model_profile_id": "Llama2-7B",
                    "gpu_id": "gpu0",
                    "block_size": 16,
                    "max_num_seqs": 256,
                    "max_num_batched_tokens": 4096,
                    "scheduler_iteration_delay_s": 0.0001,
                    "bytes_per_token_estimate_for_network": 2,
                },
            }
        ],
        "metrics_config": {
            "percentiles_to_calculate": [0.5, 0.9, 0.95, 0.99],
            "warm_up_duration_s": 30,
            "output_summary_json_path": "experiments/results/summary.json",
            "output_requests_csv_path": "experiments/results/requests.csv",
        },
    }
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "yaml":
        import yaml
        with open(output_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    else:
        import json
        with open(output_path, "w") as f:
            json.dump(example_config, f, indent=2)
    
    click.echo(f"Generated example configuration at {output_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--fix", "-f", is_flag=True,
    help="Attempt to fix configuration issues automatically"
)
@click.option(
    "--validate-models", "-m", is_flag=True,
    help="Also validate the model characteristics database"
)
def validate(config_file: str, fix: bool, validate_models: bool):
    """Validate a configuration file without running the simulation."""
    click.echo(f"Validating configuration: {config_file}")
    
    try:
        # Validate configuration
        is_valid, errors, config = validate_and_fix_config(config_file)
        
        if is_valid:
            click.echo(click.style("✓ Configuration is valid", fg="green"))
        else:
            click.echo(click.style(f"✗ Configuration has {len(errors)} errors:", fg="red"))
            for i, error in enumerate(errors[:20], 1):
                click.echo(f"  {i}. {error}")
            if len(errors) > 20:
                click.echo(f"  ... and {len(errors) - 20} more errors")
        
        # Validate model database if requested
        if validate_models and config and "model_characteristics_db_path" in config:
            model_db_path = config["model_characteristics_db_path"]
            click.echo(f"\nValidating model database: {model_db_path}")
            
            model_valid, model_errors = validate_model_db(model_db_path)
            if model_valid:
                click.echo(click.style("✓ Model database is valid", fg="green"))
            else:
                click.echo(click.style(f"✗ Model database has {len(model_errors)} errors:", fg="red"))
                for i, error in enumerate(model_errors[:10], 1):
                    click.echo(f"  {i}. {error}")
        
        # Exit with appropriate code
        sys.exit(0 if is_valid else 1)
        
    except Exception as e:
        click.echo(click.style(f"Error validating configuration: {e}", fg="red"))
        sys.exit(1)


if __name__ == "__main__":
    cli()