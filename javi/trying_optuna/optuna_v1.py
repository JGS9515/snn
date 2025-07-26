import torch
import pandas as pd
import numpy as np
import os
import argparse
import json
import optuna
from sklearn.model_selection import TimeSeriesSplit
from iago.dependencias import *
from javi.dependencias_javi import *
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support


def objective(trial):
    # Definir los parámetros a optimizar
    config = {
        'nu1': trial.suggest_float('nu1', -0.5, 0.5),
        'nu2': trial.suggest_float('nu2', -0.5, 0.5),
        # 'snn_process_layer_neurons_size': trial.suggest_int('snn_process_layer_neurons_size', 100, 400),
        # Probar todos los experimentos con las 3 configuraciones, o buscar mejor resultados con 100, luego 200 luego 400.
        'threshold': trial.suggest_float('threshold', 0.1, 1.0),
        'decay': trial.suggest_float('decay', 0.8, 1.5),
        'kernel_type': trial.suggest_categorical('kernel_type', ['gaussian', 'exponential']),

        
        'ampliacion': trial.suggest_float('ampliacion', 0.05, 0.5),
        # Investigar que valor fijo deber'ia ser mejor para este par'ametro
        'resolucion': trial.suggest_float('resolucion', 0.005, 0.05),
        # Investigar que valor fijo deber'ia ser mejor para este par'ametro

        'epochs': trial.suggest_int('epochs', 5, 50),
        # Dejarlo por defecto en un mismo valor
        'recurrencia': True
        # Dejar siempre en true
    }
    print(f"config: {config}")
    
    TIEMPO=100 #Valor razonablemente pequeño para no eliminar muchas muestras del entremiento.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar y preparar los datos
    data = pd.read_csv(args.data_path, na_values=['NA'])
    data['value'] = data['value'].astype('float64')
    data['label'] = data['label'].astype('Int64')
    data.loc[data['label'].isna(), 'label'] = 0

    # Implementar validación cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    split = 0
    for train_index, test_index in tscv.split(data):
        split+=1
        print(f"Split {split}:")
        print(f"Train indices: {train_index}")
        print(f"Test indices: {test_index}")
        print()
        
        
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        
        # Resetear índices
        data_train = data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)

        # Preparar los datos
        data_train['value'] = data_train['value'].astype('float64')
        data_test['label'] = data_test['label'].astype('Int64')

        # Normalizar los datos
        minimo = min(data_train['value'][data_train['label'] != 1])
        maximo = max(data_train['value'][data_train['label'] != 1])
        
        #Anulamos valores donde el label sea 1, para el conjunto de entrenamiento:
        #!!!!!!!!!!!!!Revisar porqué perplexity quitó este paso
        data_train.loc[data_train['label']==1,'value']=np.nan
        
        ancho_datos = maximo - minimo

        cuantiles = torch.FloatTensor(
            np.arange(
                minimo - config['ampliacion'] * ancho_datos,
                maximo + ancho_datos * config['ampliacion'],
                (maximo - minimo) * config['resolucion']
            )
        )

        snn_input_layer_neurons_size = len(cuantiles) - 1

        # Crear y entrenar la red
        network, source_monitor, target_monitor = crear_red_convo(
            snn_input_layer_neurons_size,
            TIEMPO,
            config['snn_process_layer_neurons_size'],
            config['threshold'],
            config['decay'],
            config['nu1'],
            config['nu2'],
            config['recurrencia'],
            config['kernel_type'],
            device
        )
        
        #!!!!!!!!!!!!!Revisar porqué perplexity quitó este paso
        #Para cada secuencia del train, tenemos que pasarla y entrenar la red:
        network.learning=True

        # Entrenar la red
        for _ in range(config['epochs']):
            network.learning = True
            network = reset_voltajes(network, device)
            
            secuencias2train = convertir_data(data_train, TIEMPO, cuantiles, snn_input_layer_neurons_size, device, is_train=True)
            print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
            spikes_input, spikes, network = ejecutar_red(secuencias2train, network, source_monitor, target_monitor, TIEMPO)

        # Evaluar la red
        network.learning = False
        network = reset_voltajes(network, device)
        secuencias2test = convertir_data(data_test, TIEMPO, cuantiles, snn_input_layer_neurons_size, device)
        print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
        spikes_list = []
        for seq in secuencias2test:
            if seq.shape[1] < TIEMPO:
                # Pad sequence to match TIEMPO
                padded_seq = torch.zeros((seq.shape[0], TIEMPO), device=device)
                padded_seq[:, :seq.shape[1]] = seq
                seq = padded_seq
            elif seq.shape[1] > TIEMPO:
                # Truncate sequence to match TIEMPO
                seq = seq[:, :TIEMPO]
                
            spikes_input, spikes_seq, network = ejecutar_red([seq], network, source_monitor, target_monitor, TIEMPO)
            spikes_seq_tensor = torch.from_numpy(spikes_seq).to(device)
            spikes_list.append(spikes_seq_tensor)
            
        spikes = torch.cat(spikes_list, dim=0)
        
        # spikes_input, spikes, network = ejecutar_red(secuencias2test, network, source_monitor, target_monitor, TIEMPO)

        # Calcular métricas
        spikes = torch.cat(spikes_list, dim=0)
        
        # Convert to float for statistical calculations
        spikes = spikes.float()
        
        # Calculate metrics using PyTorch functions
        umbral_anomalia = spikes.mean() + 2 * spikes.std()
        
        # Convert to numpy for predictions
        spikes_np = spikes.cpu().numpy()
        predictions = (spikes_np.sum(axis=1) > umbral_anomalia.item()).astype(int)
        
        # Ensure the lengths match
        if len(predictions) != len(data_test['label']):
            raise ValueError(f"Inconsistent number of samples: {len(data_test['label'])} in data_test['label'] and {len(predictions)} in predictions")
        
        precision, recall, f1, _ = precision_recall_fscore_support(data_test['label'], predictions, average='binary')
        
        scores.append(f1)

    return np.mean(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros con Optuna.')
    parser.add_argument('-d', '--data_path', type=str, default='Nuevos datasets\\Callt2\\preliminar\\train_label_filled.csv', help='Ruta al archivo de datos CSV')
    parser.add_argument('-n', '--n_trials', type=int, default=100, help='Número de trials para Optuna')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    print('Mejor configuración encontrada:')
    print(study.best_params)
    print(f'Mejor F1-score: {study.best_value}')

    # Guardar la mejor configuración
    with open('best_config.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
