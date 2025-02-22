from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
import torch, pandas as pd, numpy as np, os
from bindsnet.learning import PostPre
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os


def reset_voltajes(network):
    network.layers['B'].v=torch.full(network.layers['B'].v.shape,-65)
    return network


def dividir(data,minimo):
    #Función que divide los datos de entrenamiento, para considerar aisladamente cada subsecuencia de datos normales.
    #Tomamos los intervalos:
    intervals = []
    in_sequence = False
    
    #Iteramos para identificar los intervalos:
    for i in range(len(data)):
        if data.loc[i, 'label'] == 0:
            if not in_sequence:
                start_idx = i
                in_sequence = True
            end_idx = i+1
        else:
            if in_sequence:
                intervals.append((start_idx, end_idx))
                in_sequence = False
    
    # Agrega la posición del último elemento de los datos de entrada:
    if in_sequence:
        intervals.append((start_idx, end_idx))
    
    #Creamos un dataframe con los intervalos encontrados:
    intervals_df = pd.DataFrame(intervals, columns=['inicio', 'final'])
    
    subs=[]
    #Iteramos para dividir:
    for i,row in intervals_df.iterrows():
        inicio_tmp=row['inicio']
        final_tmp=row['final']
        if final_tmp-inicio_tmp>=minimo:
            subs.append(data.iloc[inicio_tmp:final_tmp].reset_index(drop=True))
    
    return subs


def padd(data, T):
    lon = len(data)
    # Calcular el múltiplo más cercano de T superior al número actual de filas
    lon2 = ((lon // T) + 1) * T
    # Calcular el número de filas adicionales necesarias
    lon_adicional = lon2 - lon
    
    # Crear un DataFrame con filas adicionales llenas de NaN
    if lon_adicional > 0:
        datanul = pd.DataFrame(np.nan, index=range(lon_adicional), columns=data.columns)
        # Concatenar el DataFrame original con el DataFrame de padding
        data = pd.concat([data, datanul], ignore_index=True)
    
    return data


def expandir(serie, n):
    # Crea gemelo de la serie:
    serie2 = np.zeros_like(serie)
    
    # Identificar los índices donde hay un 1:
    indices = np.where(serie == 1)[0]
    
    # Poner a 1 los valores en el rango [índice-n, índice+n]
    for idx in indices:
        start = max(0, idx - n)
        end = min(len(serie), idx + n + 1)
        serie2[start:end] = 1
    
    return pd.Series(serie2, index=serie.index)


#Función para convertir a spikes las entradas:
def podar(x,q1,q2,cuantiles=None):
    #Función que devuelve 1 (spike) si x está en el rango [q1,q2), y 0 en caso contrario.
    #Es parte de la codificación de los datos.
    
    s=torch.zeros_like(x)
    
    s[(x>=q1) & (x<q2)]=1
    return s


def convertir_data(data,T,cuantiles,snn_input_layer_neurons_size,is_train=False):
    #Función que lee los datos y los prepara.
    
    #Esta parte debe ser modificada para obtener la serie temporal de interés
    #almacenada en la variable serie.
    #
    serie=torch.FloatTensor(data['value'])
    
    #Tomamos la longitud de la serie.
    long=serie.shape[0]
    
    #Los valores inferiores al mínimo del vector de cuantiles se sustituyen por ese mínimo.
    #Los valores mayores que el máximo de los cuantiles se sustituyen por ese máximo.
    #Esto sirve para no perder spikes cuando se procesan los datos de prueba.
    
    serie[serie<torch.min(cuantiles)]=torch.min(cuantiles)
    serie[serie>torch.max(cuantiles)]=torch.max(cuantiles)
    
    #Construimos el tensor con los datos codificados. Básicamente, para cada dato de entrada tendremos una secuencia donde 
    #todos los valores son 0 menos 1, correspondiente al cuantil correspondiente al dato de entrada.
    serie2input=torch.cat([serie.unsqueeze(0)] * snn_input_layer_neurons_size, dim=0)
    
    for i in range(snn_input_layer_neurons_size):
        serie2input[i,:]=podar(serie2input[i,:],cuantiles[i],cuantiles[i+1])
    
    #Lo dividimos en función del tiempo de exposición T:
    secuencias = torch.split(serie2input,T,dim=1)
    
    if is_train:
        #Encaso de estar entrenando, tendríamos que quitar la última secuencia:
        secuencias=secuencias[0:len(secuencias)-1]
    
    return secuencias


# Function to create a Gaussian kernel
def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    # Create a 1D Gaussian kernel
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    gaussian = torch.exp(-x**2 / (2*sigma**2))
    gaussian = gaussian / gaussian.sum()  # Normalize
    return gaussian.view(1, 1, -1)  # Shape for 1D convolution


def crear_red(snn_input_layer_neurons_size, decaimiento, umbral, nu1, nu2, n, T):
    # Create the network
    network = Network()
    
    # Create layers: input -> internal -> conv
    source_layer = Input(n=snn_input_layer_neurons_size, traces=True)
    target_layer = LIFNodes(n=n, traces=True, thresh=umbral, tc_decay=decaimiento)
    conv_layer = LIFNodes(n=n, traces=True, thresh=umbral, tc_decay=decaimiento)  # Output convolutional layer

    network.add_layer(layer=source_layer, name="A")
    network.add_layer(layer=target_layer, name="B")
    network.add_layer(layer=conv_layer, name="C")
    
    #Creamos conexiones entre las capas de entrada y la recurrente:
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0).repeat(n, 1, 1)
    kernel_size=5
    sigma=1.0
    
    # Create connections between input layer and recurrent layer
    forward_connection = Connection(
        source=source_layer,
        target=target_layer,
        w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),
        update_rule=PostPre, nu=nu1#nu=(1e-4, 1e-2)    # Normal(0.05, 0.01) weights.
    )
    
    network.add_connection(
        connection=forward_connection, source="A", target="B"
    )
    
    # Creamos la conexión recurrente con pesos ligeramente negativos (si no, estamos metiendo posible ruido en el procesamiento de la red):
    recurrent_connection = Connection(
        source=target_layer,
        target=target_layer,
        w=0.025 * (torch.eye(target_layer.n) - 1), 
        update_rule=PostPre, nu=nu2#nu=(1e-4, 1e-2)
    )
    
    network.add_connection(
        connection=recurrent_connection, source="B", target="B"
    )
    
     # Conexión convolucional B->C con matriz Toeplitz
    weights = torch.zeros(n, n)
    center = kernel_size // 2
    for i in range(n):
        start = max(0, i - center)
        end = min(n, i + center + 1)
        k_start = max(0, center - i)
        k_end = kernel_size - max(0, i + center + 1 - n)
        weights[i, start:end] = kernel[0,0,k_start:k_end]
    
    conv_connection = Connection(
        source=target_layer,
        target=conv_layer,
        w=weights,
        update_rule=PostPre,
        nu=nu2,
        norm=0.5 * kernel_size  # Normalización por tamaño del kernel
    )
    network.add_connection(conv_connection, "B", "C")
    
    # # Connect layer B to convolutional layer
    # conv_connection = Connection(
    #     source=target_layer,
    #     target=conv_layer,
    #     w=torch.ones(target_layer.n, conv_layer.n)  # Initial weights for conv connection
    # )

    # network.add_connection(connection=conv_connection, source="B", target="C")
    
    #Creamos los monitores. Sirven para registrar los spikes y voltajes:
    #Spikes de entrada (para depurar que se esté haciendo bien, si se quiere):
    # Create monitors
    source_monitor = Monitor(
        obj=source_layer,
        state_vars=("s",),  #Registramos sólo los spikes.
        time=T,
    )
    #Spikes de la capa recurrente (lo que nos interesa):
    target_monitor = Monitor(
        obj=target_layer,
        state_vars=("s", "v"),  #Registramos spikes y voltajes, por si nos interesa lo segundo también.
        time=T,
    )
    conv_monitor = Monitor(
        obj=conv_layer,
        state_vars=("s", "v"),
        time=T,
    )
    
    network.add_monitor(monitor=source_monitor, name="X")
    network.add_monitor(monitor=target_monitor, name="Y")
    network.add_monitor(monitor=conv_monitor, name="Conv_mon")
    
    return [network, source_monitor, target_monitor, conv_monitor]


def ejecutar_red(secuencias, network, source_monitor, target_monitor, conv_monitor, T):
    #Función para ejecutar la red con los datos que se quieran, ya sea para entrenamiento o evaluación.
    
    #Creamos listas en que almacenaremos los resultados:
    sp0 = []
    sp1 = []
    sp_conv = []
    
    j = 1
    for i in secuencias:
        print(f'Ejecutando secuencia {j}')
        j += 1
        
        # Prepare inputs
        inputs = {'A': i.T}
        
        network.run(inputs=inputs, time=T)
        
        # Get spikes from all layers
        spikes = {
            "X": source_monitor.get("s"),
            "B": target_monitor.get("s"),
            "C": conv_monitor.get("s")
        }
        
        # Apply Gaussian convolution to B layer spikes
        b_spikes = spikes["B"].float()  # Convert to float for convolution [T, 1, n]
        
        # Reshape para la convolución: [1, 1, T]
        b_spikes_sum = b_spikes.sum(dim=2).transpose(0, 1)  # Sumamos sobre neuronas y transponemos
        
        conv_spikes = F.conv1d(
            b_spikes_sum,  # [1, 1, T]
            create_gaussian_kernel(),
            padding='same'
        )
        
        sp0.append(spikes['X'].sum(axis=2))
        sp1.append(spikes['B'].sum(axis=2))
        sp_conv.append(conv_spikes.squeeze())  # Eliminamos dimensiones extras
        
        network = reset_voltajes(network)
    
    sp0 = torch.cat(sp0)
    sp0 = sp0.detach().numpy()
    
    sp1 = torch.cat(sp1)
    sp1 = sp1.detach().numpy()
    
    sp_conv = torch.cat(sp_conv)
    sp_conv = sp_conv.detach().numpy()
    
    return [sp0, sp1, sp_conv, network]

def guardar_resultados(spikes, spikes_conv, data_test, n, snn_input_layer_neurons_size, n_trial,date_starting_trials):
   
    # Create directory structure
    
    base_path = f'resultados/{date_starting_trials}/trial_{n_trial}'
    os.makedirs(base_path, exist_ok=True)

    # Save spikes
    np.savetxt(f'{base_path}/spikes', spikes, delimiter=',')
    np.savetxt(f'{base_path}/spikes_conv', spikes_conv, delimiter=',')

    # Convert and save labels - handle NA values properly
    labels = data_test['label'].replace([np.inf, -np.inf], np.nan)
    labels = labels.astype(float)
    labels = labels.to_numpy()
    np.savetxt(f'{base_path}/label', labels, delimiter=',')

    # Convert and save values - handle NA values properly 
    values = data_test['value'].replace([np.inf, -np.inf], np.nan)
    values = values.astype(float)
    values = values.to_numpy()
    np.savetxt(f'{base_path}/value', values, delimiter=',')

    # Save timestamps
    timestamps = data_test['timestamp'].replace([np.inf, -np.inf], np.nan)
    timestamps = timestamps.astype(float)
    timestamps = timestamps.to_numpy()
    np.savetxt(f'{base_path}/timestamp', timestamps, delimiter=',')

    # Create DataFrame with 1D arrays
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'label': labels
    })

    # Save to CSV with same format as original
    results_df.to_csv(f'{base_path}/data_test.csv', 
                     index=False,
                     float_format='%.6f')

    # Reshape/flatten spikes to 1D if needed
    spikes_1d = spikes.sum(axis=1) if len(spikes.shape) > 1 else spikes

    # Create DataFrame with 1D arrays
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'label': spikes_1d
    })

    # Save to CSV with same format as original
    results_df.to_csv(f'{base_path}/results.csv', 
                     index=False,
                     float_format='%.6f')

    spikes_conv_1d = spikes_conv.sum(axis=1) if len(spikes_conv.shape) > 1 else spikes_conv

    results_conv_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'label': spikes_conv_1d
    })

    results_conv_df.to_csv(f'{base_path}/results_conv.csv', 
                          index=False,
                          float_format='%.6f')

    with open(f'{base_path}/n1', 'w') as n1:
        n1.write(f'{snn_input_layer_neurons_size}\n')

    with open(f'{base_path}/n2', 'w') as n2:
        n2.write(f'{n}\n')

    # Calculate MSE scores
    y_true = data_test['label'].astype(float).to_numpy()
    y_true = np.nan_to_num(y_true, nan=0.0)

    spikes_1d = spikes_1d.astype(float)
    spikes_1d = np.nan_to_num(spikes_1d, nan=0.0)

    mse_B = mean_squared_error(y_true, spikes_1d)
    print("MSE capa B:", mse_B)
    with open(f'{base_path}/MSE_capa_B', 'w') as n2:
        n2.write(f'{mse_B}\n')

    # Repetir el proceso para spikes_conv_1d si es necesario
    spikes_conv_1d = spikes_conv_1d.astype(float)
    spikes_conv_1d = np.nan_to_num(spikes_conv_1d, nan=0.0)

    mse_C = mean_squared_error(y_true, spikes_conv_1d)    
    print("MSE capa C:", mse_C)
    with open(f'{base_path}/MSE_capa_C', 'w') as n2:
        n2.write(f'{mse_C}\n')
        
    return mse_B, mse_C