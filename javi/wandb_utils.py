import wandb
import os
import json
import numpy as np
from datetime import datetime

def init_wandb(config, project_name="SNN-Anomaly-Detection", entity=None, name=None):
    """
    Initialize a wandb run with the given configuration.
    
    Args:
        config: Dictionary containing model configuration and hyperparameters
        project_name: Name of the wandb project
        entity: Optional wandb entity (team or username)
        name: Optional name for this specific run
        
    Returns:
        wandb run object
    """
    # Increase timeout for slower systems
    settings = wandb.Settings(init_timeout=120)
    
    # Create a unique name if none provided
    if name is None:
        name = f"snn-trial-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize the run
    run = wandb.init(
        project=project_name,
        entity=entity,
        name=name,
        config=config,
        settings=settings
    )
    
    return run

def log_training_metrics(run, epoch, spikes_input, spikes_output, target=None):
    """
    Log training metrics for each epoch or batch to wandb.
    
    Args:
        run: wandb run object
        epoch: Current epoch or batch number
        spikes_input: Input layer spikes
        spikes_output: Output layer spikes
        target: Optional target values for comparison
    """
    # Calculate spike statistics
    input_spike_rate = np.mean(spikes_input)
    output_spike_rate = np.mean(spikes_output)
    
    # Log metrics
    metrics = {
        "epoch": epoch,
        "input_spike_rate": input_spike_rate,
        "output_spike_rate": output_spike_rate,
    }
    
    run.log(metrics)

def log_evaluation_results(run, mse_B, mse_C=None, dataset_name=None):
    """
    Log evaluation results to wandb.
    
    Args:
        run: wandb run object
        mse_B: MSE for layer B
        mse_C: Optional MSE for convolutional layer C
        dataset_name: Name of the dataset used for evaluation
    """
    # Log MSE scores
    metrics = {
        "mse_layer_B": mse_B
    }
    
    if mse_C is not None:
        metrics["mse_layer_C"] = mse_C
    
    if dataset_name:
        metrics["dataset"] = dataset_name
        
    run.log(metrics)

def log_model_architecture(run, network_config):
    """
    Log SNN architecture details to wandb.
    
    Args:
        run: wandb run object
        network_config: Dictionary containing network architecture details
    """
    # Log network architecture as a Table
    columns = ["Layer", "Type", "Neurons", "Parameters"]
    data = []
    
    # Add input layer
    data.append(["A", "Input", network_config["snn_input_layer_neurons_size"], 0])
    
    # Add hidden layer
    n_params_B = (network_config["snn_input_layer_neurons_size"] * network_config["snn_process_layer_neurons_size"] + 
                  network_config["snn_process_layer_neurons_size"]**2)
    data.append(["B", "LIF", network_config["snn_process_layer_neurons_size"], n_params_B])
    
    # Add conv layer if used
    if network_config.get("use_conv_layer", True):
        data.append(["C", "LIF (Conv)", network_config["snn_process_layer_neurons_size"], 
                    network_config["snn_process_layer_neurons_size"]**2])
    
    # Create and log table
    table = wandb.Table(data=data, columns=columns)
    run.log({"network_architecture": table})

def finish_wandb_run():
    """
    Finish the current wandb run.
    """
    if wandb.run is not None:
        wandb.finish()