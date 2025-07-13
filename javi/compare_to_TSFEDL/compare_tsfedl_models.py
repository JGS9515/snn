import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
javi_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(javi_dir))

try:
    from utils import *
    from wandb_utils import *
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    UTILS_AVAILABLE = False

# TSFEDL imports
TSFEDL_AVAILABLE = False
try:
    import s_tsfe_dl
    from s_tsfe_dl.models import CNNBiLSTM, CNNBiGRU, CNNBiRNN, CNNLSTM, CNNGRU, CNNsimpleRNN
    from s_tsfe_dl.models import MLP, VariationalAutoencoder, ConvolutionalAutoencoder
    from s_tsfe_dl.models import DeepAnT, LSTMAE, GRU_AE, AnomalyTransformer
    from s_tsfe_dl.models import RNNBased, LSTMBased, GRUBased, BiLSTMBased, BiGRUBased, BiRNNBased
    from s_tsfe_dl.models import CNN1D, CNN2D, InceptionTime, ResNetBased
    TSFEDL_AVAILABLE = True
    print("TSFEDL library loaded successfully")
except ImportError as e:
    print(f"TSFEDL not available: {e}")
    print("Continuing without TSFEDL models. Will use basic models instead.")
    TSFEDL_AVAILABLE = False

class TSFEDLModelWrapper:
    """Wrapper class to standardize TSFEDL model interface"""
    
    def __init__(self, model_class, model_name, **kwargs):
        self.model_class = model_class
        self.model_name = model_name
        self.model = None
        self.kwargs = kwargs
        
    def prepare_data(self, X, y, sequence_length=50):
        """Prepare data for TSFEDL models"""
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length-1])
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the TSFEDL model"""
        try:
            # Initialize model
            input_shape = X_train.shape[1:]
            self.model = self.model_class(
                input_shape=input_shape,
                **self.kwargs
            )
            
            # Train model
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
            else:
                self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
            
            return True
        except Exception as e:
            print(f"Error training {self.model_name}: {e}")
            return False
    
    def predict(self, X_test):
        """Make predictions"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            predictions = self.model.predict(X_test)
            return predictions
        except Exception as e:
            print(f"Error predicting with {self.model_name}: {e}")
            return None

def load_and_prepare_data(path, test_size=0.2, sequence_length=50):
    """Load and prepare data for both SNN and TSFEDL models"""
    
    # Load data
    data = pd.read_csv(path, na_values=['NA'])
    data['value'] = data['value'].astype('float64')
    data['label'] = data['label'].astype('Int64')
    data.loc[data['label'].isna(), 'label'] = 0
    
    # Split data
    split = len(data) // 2
    data_train = data[:split].reset_index(drop=True)
    data_test = data[split:].reset_index(drop=True)
    
    # Prepare for TSFEDL models
    X_train = data_train['value'].values.reshape(-1, 1)
    y_train = data_train['label'].values
    X_test = data_test['value'].values.reshape(-1, 1)
    y_test = data_test['label'].values
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'data_train': data_train,
        'data_test': data_test,
        'scaler': scaler
    }

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    
    # Convert predictions to binary if needed
    if len(np.unique(y_pred)) > 2:
        # For regression-like outputs, threshold at median
        threshold = np.median(y_pred)
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)
    precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        'model': model_name,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'mse': mse,
        'predictions': y_pred_binary
    }

def get_tsfedl_models():
    """Get list of TSFEDL models to test"""
    
    models = []
    
    if not TSFEDL_AVAILABLE:
        print("TSFEDL not available, skipping TSFEDL models")
        return models
    
    # CNN-RNN Hybrid models
    models.extend([
        TSFEDLModelWrapper(CNNBiLSTM, "CNN-BiLSTM", num_classes=2),
        TSFEDLModelWrapper(CNNBiGRU, "CNN-BiGRU", num_classes=2),
        TSFEDLModelWrapper(CNNBiRNN, "CNN-BiRNN", num_classes=2),
        TSFEDLModelWrapper(CNNLSTM, "CNN-LSTM", num_classes=2),
        TSFEDLModelWrapper(CNNGRU, "CNN-GRU", num_classes=2),
        TSFEDLModelWrapper(CNNsimpleRNN, "CNN-SimpleRNN", num_classes=2),
    ])
    
    # Pure RNN models
    models.extend([
        TSFEDLModelWrapper(LSTMBased, "LSTM", num_classes=2),
        TSFEDLModelWrapper(GRUBased, "GRU", num_classes=2),
        TSFEDLModelWrapper(BiLSTMBased, "BiLSTM", num_classes=2),
        TSFEDLModelWrapper(BiGRUBased, "BiGRU", num_classes=2),
        TSFEDLModelWrapper(RNNBased, "RNN", num_classes=2),
    ])
    
    # CNN models
    models.extend([
        TSFEDLModelWrapper(CNN1D, "CNN1D", num_classes=2),
        TSFEDLModelWrapper(InceptionTime, "InceptionTime", num_classes=2),
        TSFEDLModelWrapper(ResNetBased, "ResNet", num_classes=2),
    ])
    
    # Autoencoder models for anomaly detection
    models.extend([
        TSFEDLModelWrapper(LSTMAE, "LSTM-AE", latent_dim=32),
        TSFEDLModelWrapper(GRU_AE, "GRU-AE", latent_dim=32),
        TSFEDLModelWrapper(VariationalAutoencoder, "VAE", latent_dim=32),
        TSFEDLModelWrapper(ConvolutionalAutoencoder, "Conv-AE", latent_dim=32),
    ])
    
    # Other models
    models.extend([
        TSFEDLModelWrapper(MLP, "MLP", num_classes=2),
        TSFEDLModelWrapper(DeepAnT, "DeepAnT", num_classes=2),
    ])
    
    return models

def run_snn_experiment(data_dict, config_path=None):
    """Run the original SNN experiment for comparison"""
    
    print("Running SNN experiment...")
    
    if not UTILS_AVAILABLE:
        print("Utils module not available. Skipping SNN experiment.")
        return {
            'model': 'SNN',
            'f1_score': 0,
            'precision': 0,
            'recall': 0,
            'mse': float('inf'),
            'f1_score_conv': 0,
            'precision_conv': 0,
            'recall_conv': 0,
            'mse_conv': float('inf')
        }
    
    # Use default config or load from file
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config from your experiment
        config = {
            'nu1': 0.001657625626991273,
            'nu2': -0.3546294590908031,
            'threshold': -62.416319068986716,
            'decay': 145.29873304569423
        }
    
    # Parameters
    a = 0.1
    r = 0.05
    T = 250
    expansion = 100
    n = 100
    
    # Load data
    data_train = data_dict['data_train']
    data_test = data_dict['data_test']
    
    # Expand labels
    data_train['label'] = expandir(data_train['label'], expansion)
    
    # Get min/max
    minimo = min(data_train['value'][data_train['label'] != 1])
    maximo = max(data_train['value'][data_train['label'] != 1])
    
    # Create quantiles
    amplitud = maximo - minimo
    cuantiles = torch.FloatTensor(np.arange(minimo-a*amplitud, maximo+amplitud*a, (maximo-minimo)*r))
    snn_input_layer_neurons_size = len(cuantiles) - 1
    
    # Create network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network, source_monitor, target_monitor, conv_monitor = crear_red(
        snn_input_layer_neurons_size, config['decay'], config['threshold'], 
        (config['nu1'], config['nu1']), (config['nu2'], config['nu2']), 
        n, T, use_conv_layer=True, conv_params=None, device=device
    )
    
    # Train
    data_train_sequences = dividir(data_train, T)
    data_test_padded = padd(data_test, T)
    
    network.learning = True
    for s in data_train_sequences:
        secuencias2train = convertir_data(s, T, cuantiles, snn_input_layer_neurons_size, is_train=True, device=device)
        spikes_input, spikes, spikes_conv, network = ejecutar_red(
            secuencias2train, network, source_monitor, target_monitor, conv_monitor, T,
            use_conv_layer=True, conv_processing_type='weighted_sum', device=device
        )
        network = reset_voltajes(network)
    
    # Test
    network.learning = False
    secuencias2test = convertir_data(data_test_padded, T, cuantiles, snn_input_layer_neurons_size, is_train=False, device=device)
    spikes_input, spikes, spikes_conv, network = ejecutar_red(
        secuencias2test, network, source_monitor, target_monitor, conv_monitor, T,
        use_conv_layer=True, conv_processing_type='weighted_sum', device=device
    )
    
    # Evaluate
    date_starting_trials = datetime.now().strftime('%Y_%m_%d-%H_%M')
    mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C = guardar_resultados(
        spikes, spikes_conv, data_test, n, snn_input_layer_neurons_size, 1,
        date_starting_trials, "comparison", 100, trial=None
    )
    
    return {
        'model': 'SNN',
        'f1_score': f1_B if f1_B is not None else 0,
        'precision': precision_B if precision_B is not None else 0,
        'recall': recall_B if recall_B is not None else 0,
        'mse': mse_B if mse_B is not None else float('inf'),
        'f1_score_conv': f1_C if f1_C is not None else 0,
        'precision_conv': precision_C if precision_C is not None else 0,
        'recall_conv': recall_C if recall_C is not None else 0,
        'mse_conv': mse_C if mse_C is not None else float('inf')
    }

def run_comparison_experiment(data_path, output_dir, sequence_length=50, epochs=50):
    """Run comprehensive comparison experiment"""
    
    print(f"Starting comparison experiment with data: {data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    data_dict = load_and_prepare_data(data_path, sequence_length=sequence_length)
    
    # Initialize results
    results = []
    
    # Run SNN experiment
    try:
        snn_results = run_snn_experiment(data_dict)
        results.append(snn_results)
        print(f"SNN - F1: {snn_results['f1_score']:.4f}, MSE: {snn_results['mse']:.4f}")
    except Exception as e:
        print(f"SNN experiment failed: {e}")
    
    # Run TSFEDL experiments
    if TSFEDL_AVAILABLE:
        models = get_tsfedl_models()
        
        for model_wrapper in models:
            try:
                print(f"\nTraining {model_wrapper.model_name}...")
                
                # Prepare sequences
                X_train_seq, y_train_seq = model_wrapper.prepare_data(
                    data_dict['X_train'], data_dict['y_train'], sequence_length
                )
                X_test_seq, y_test_seq = model_wrapper.prepare_data(
                    data_dict['X_test'], data_dict['y_test'], sequence_length
                )
                
                # Train model
                success = model_wrapper.fit(
                    X_train_seq, y_train_seq, 
                    epochs=epochs, batch_size=32
                )
                
                if success:
                    # Make predictions
                    predictions = model_wrapper.predict(X_test_seq)
                    
                    if predictions is not None:
                        # Evaluate
                        model_results = evaluate_model(
                            y_test_seq, predictions, model_wrapper.model_name
                        )
                        results.append(model_results)
                        print(f"{model_wrapper.model_name} - F1: {model_results['f1_score']:.4f}, MSE: {model_results['mse']:.4f}")
                    else:
                        print(f"{model_wrapper.model_name} - Prediction failed")
                else:
                    print(f"{model_wrapper.model_name} - Training failed")
                    
            except Exception as e:
                print(f"Error with {model_wrapper.model_name}: {e}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if len(results) > 0:
        results_df_sorted = results_df.sort_values('f1_score', ascending=False)
        print(f"{'Model':<20} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'MSE':<10}")
        print("-"*60)
        
        for _, row in results_df_sorted.iterrows():
            print(f"{row['model']:<20} {row['f1_score']:<10.4f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['mse']:<10.4f}")
    
    return results

def main():
    """Main execution function"""
    
    # Configuration - using absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(project_root, 'Nuevos datasets', 'iops', 'preliminar', 'train_procesado_javi', '1c35dbf57f55f5e4_filled.csv')
    output_dir = os.path.join(script_dir, f'comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        return None
    
    # Run comparison
    results = run_comparison_experiment(
        data_path=data_path,
        output_dir=output_dir,
        sequence_length=50,
        epochs=30  # Reduced for faster testing
    )
    
    print(f"\nResults saved to: {output_dir}")
    return results

if __name__ == "__main__":
    main() 