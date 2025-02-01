import torch
import pandas as pd
import numpy as np
import os
import argparse
import json
import optuna
from sklearn.model_selection import TimeSeriesSplit
from iago.dependencias import *
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from dependencias_javi import *

def objective(trial):
    config = {
        'nu1': trial.suggest_float('nu1', -0.5, 0.5),
        'nu2': trial.suggest_float('nu2', -0.5, 0.5),
        'snn_process_layer_neurons_size': 100,
        'threshold': trial.suggest_float('threshold', 0.1, 1.0),
        'decay': trial.suggest_float('decay', 0.8, 1.5),
        'ampliacion': trial.suggest_float('ampliacion', 0.05, 0.5),
        'resolucion': trial.suggest_float('resolucion', 0.005, 0.05),
        'epochs': trial.suggest_int('epochs', 5, 50),
        'recurrencia': True,
        'kernel_type': 'gaussian'
    }
    
    TIEMPO = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse = torch.nn.MSELoss()

    data = pd.read_csv(args.data_path, na_values=['NA'])
    data['value'] = data['value'].astype('float64')
    data['label'] = data['label'].astype('Int64')
    data.loc[data['label'].isna(), 'label'] = 0

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    reconstruction_losses = []

    for train_index, test_index in tscv.split(data):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        data_train = data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)

        # Preprocesamiento
        data_train['value'] = data_train['value'].astype('float64')
        data_test['label'] = data_test['label'].astype('Int64')
        data_train.loc[data_train['label']==1, 'value'] = np.nan
        
        minimo = min(data_train['value'][data_train['label'] != 1])
        maximo = max(data_train['value'][data_train['label'] != 1])
        ancho_datos = maximo - minimo

        # cuantiles = torch.FloatTensor(
        #     np.arange(
        #         minimo - config['ampliacion'] * ancho_datos,
        #         maximo + ancho_datos * config['ampliacion'],
        #         (maximo - minimo) * config['resolucion']
        #     )
        # )
        cuantiles = torch.linspace(0, 1, steps=11)
        snn_input_layer = len(cuantiles) - 1

        # Crear red con capa convolucional
        network, source_monitor, target_monitor, conv_monitor = crear_red_convo(
            snn_input_layer,
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

        # Entrenamiento
        network.learning = True
        for _ in range(config['epochs']):
            network = reset_voltajes(network, device)
            secuencias_train = convertir_data_convo(data_train, TIEMPO, cuantiles, snn_input_layer, device, is_train=True)
            sp0_train, sp1_train, sp2_train, conv_voltage_train, network = ejecutar_red_convo(
                secuencias_train, network, source_monitor, target_monitor, conv_monitor, TIEMPO
            )
        # Evaluación
        network.learning = False
        network = reset_voltajes(network, device)
        secuencias_test = convertir_data_convo(data_test, TIEMPO, cuantiles, snn_input_layer, device)

        
        total_loss = 0
        predictions = []
        true_labels = []
        
        for seq in secuencias_test:
            if seq.shape[1] < TIEMPO:
                seq = torch.nn.functional.pad(seq, (0, TIEMPO - seq.shape[1]))
            elif seq.shape[1] > TIEMPO:
                seq = seq[:, :TIEMPO]
            
            # Ejecutar red y obtener reconstrucción
            sp0_train, sp1_train, sp2_train, conv_voltage_train, network = ejecutar_red_convo([seq], network, source_monitor, target_monitor,conv_monitor, TIEMPO)
            
            # Obtener voltaje de la capa convolucional
            conv_voltage = network.monitors['Z'].get("v").squeeze()
            
            # Calcular pérdida de reconstrucción
            target = torch.FloatTensor(data_test['value'].values[:len(conv_voltage)]).to(device)
            total_loss += mse(conv_voltage, target).item()
            
            # Detección de anomalías
            spikes = network.monitors['Y'].get("s").sum(axis=2).cpu().numpy()
            predictions.extend(spikes.mean(axis=1) > (spikes.mean() + 2*spikes.std()))
            true_labels.extend(data_test['label'].values[:len(spikes)])

        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        scores.append(f1)
        reconstruction_losses.append(total_loss/len(secuencias_test))

    # Combinar F1 y pérdida de reconstrucción
    final_score = np.mean(scores) - 0.1 * np.mean(reconstruction_losses)
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros con Optuna.')
    parser.add_argument('-d', '--data_path', type=str, default='Nuevos datasets\\Callt2\\preliminar\\train_label_filled.csv', help='Ruta al archivo de datos CSV')
    parser.add_argument('-n', '--n_trials', type=int, default=100, help='Número de trials para Optuna')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    print('Mejor configuración:')
    print(study.best_params)
    print(f'Mejor score: {study.best_value:.4f}')

    with open('best_config.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)