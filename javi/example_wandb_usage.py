import wandb
from wandb_utils import init_wandb, log_training_metrics, log_model_architecture, finish_wandb_run
from utils import crear_red, ejecutar_red, guardar_resultados, convertir_data
import torch
import pandas as pd
import numpy as np

# Define your hyperparameters
config = {
    "nu1": 0.001,
    "nu2": 0.002,
    "threshold": -52.0,
    "decay": 100.0,
    "T": 250,
    "snn_input_layer_neurons_size": 10,
    "snn_process_layer_neurons_size": 100,
    "use_conv_layer": True,
    "dataset": "your_dataset_name",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Initialize wandb
run = init_wandb(
    config=config,
    project_name="SNN-Anomaly-Detection",
    entity="jgs00069-university-of-ja-n",  # Replace with your wandb username or team
    name=f"snn-n{config['snn_process_layer_neurons_size']}-trial"
)

try:
    # Load your data
    data_train = pd.read_csv("path/to/train_data.csv")
    data_test = pd.read_csv("path/to/test_data.csv")
    
    # Process your data and prepare it for SNN
    # ...
    
    # Create your SNN
    network, source_monitor, target_monitor, conv_monitor = crear_red(
        snn_input_layer_neurons_size=config["snn_input_layer_neurons_size"],
        decaimiento=config["decay"],
        umbral=config["threshold"],
        nu1=config["nu1"],
        nu2=config["nu2"],
        n=config["snn_process_layer_neurons_size"],
        T=config["T"],
        use_conv_layer=config["use_conv_layer"],
        device=config["device"]
    )
    
    # Log network architecture
    log_model_architecture(run, config)
    
    # Train your SNN
    secuencias_train = convertir_data(data_train, config["T"], cuantiles, 
                                     config["snn_input_layer_neurons_size"], 
                                     is_train=True, device=config["device"])
    
    # For tracking training progress
    for i, seq in enumerate(secuencias_train):
        sp0, sp1, sp_conv, network = ejecutar_red(
            [seq], network, source_monitor, target_monitor, conv_monitor, 
            config["T"], config["use_conv_layer"], config["device"]
        )
        
        # Log training metrics
        log_training_metrics(run, i, sp0, sp1)
    
    # Test your SNN
    secuencias_test = convertir_data(data_test, config["T"], cuantiles, 
                                    config["snn_input_layer_neurons_size"], 
                                    is_train=False, device=config["device"])
    
    sp0, sp1, sp_conv, network = ejecutar_red(
        secuencias_test, network, source_monitor, target_monitor, conv_monitor, 
        config["T"], config["use_conv_layer"], config["device"]
    )
    
    # Save results and log to wandb
    mse_B, mse_C = guardar_resultados(
        sp1, sp_conv, data_test, 
        config["snn_process_layer_neurons_size"],
        config["snn_input_layer_neurons_size"], 
        1, "run_date", "dataset_name",
        config["snn_process_layer_neurons_size"],
        type('obj', (object,), {'params': config})
    )
    
    # Optionally log model weights
    # wandb.save("path/to/saved/model.pt")
    
finally:
    # Always finish the wandb run
    finish_wandb_run()