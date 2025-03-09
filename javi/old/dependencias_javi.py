from bindsnet.network.topology import Connection
import torch, pandas as pd, numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.learning import PostPre
from bindsnet.network.monitors import Monitor
from datetime import datetime


# Añadir funciones para kernels predefinidos
def gaussian_kernel(size=3, sigma=1.0):
    """Genera un kernel gaussiano 1D"""
    x = torch.linspace(-sigma, sigma, steps=size)
    g = torch.exp(-x**2/(2*sigma**2))
    return g/g.sum()

def exponential_kernel(size=3, tau=1.0):
    """Genera un kernel exponencial 1D"""
    x = torch.arange(0, size)
    e = torch.exp(-x/tau)
    return e/e.sum()

# Modificar la función crear_red
def crear_red_convo(snn_input_layer, T, snn_process_layer_neurons_size, threshold, decay, nu1, nu2, recurrencia, kernel_type, device):
    network = Network()
    
    # Capa de entrada
    source_layer = Input(n=snn_input_layer, traces=True)
    network.add_layer(source_layer, "A")
    
    # Capa procesamiento
    process_layer = LIFNodes(n=snn_process_layer_neurons_size, traces=True, thresh=threshold, tc_decay=decay)
    network.add_layer(process_layer, "B")
    
    # Conexión entrada-proceso
    forward_conn = Connection(
        source=source_layer, 
        target=process_layer,
        w=0.05 + 0.1*torch.randn(snn_input_layer, snn_process_layer_neurons_size),
        update_rule=PostPre,
        nu=nu1
    )
    network.add_connection(forward_conn, "A", "B")
    if recurrencia: #Aquí decidimos si metemos o no la capa recurrente.
        # Creamos la conexión recurrente con pesos ligeramente negativos (si no, estamos metiendo posible ruido en el procesamiento de la red):
        recurrent_connection = Connection(
            source=process_layer,
            target=process_layer,
            w=0.025 * (torch.eye(process_layer.n) - 1),
            update_rule=PostPre, nu=nu2#nu=(1e-4, 1e-2)
        )
        
        network.add_connection(
            connection=recurrent_connection, source="B", target="B"
        )
    # Capa convolucional
    conv_layer = LIFNodes(n=1)
    network.add_layer(conv_layer, "C")
    
    # Kernel convolucional
    # kernel_size = 5
    # if kernel_type == 'gaussian':
    #     kernel = gaussian_kernel(kernel_size)
    # else:
    #     kernel = exponential_kernel(kernel_size)
    kernel = torch.randn(100)
    
    conv_conn = Connection(
        source=process_layer,
        target=conv_layer,
        w=kernel.repeat(process_layer.n, 1).view(-1, 1)
    )
    network.add_connection(conv_conn, "B", "C")
    
    # Monitores (ESENCIAL retornarlos)
    source_monitor = Monitor(source_layer, ["s"], T)
    target_monitor = Monitor(process_layer, ["s", "v"], T)
    conv_monitor = Monitor(conv_layer, ["v"], T)
    
    network.add_monitor(source_monitor, "X")
    network.add_monitor(target_monitor, "Y")
    network.add_monitor(conv_monitor, "Z")
    
    return [network, source_monitor, target_monitor, conv_monitor]

def ejecutar_red_convo(secuencias, network, source_monitor, target_monitor, conv_monitor, T):
    sp0 = []
    sp1 = []
    sp2 = []
    
    for seq in secuencias:
        # Asegurar forma [batch=1, T, R]
        inputs = {"A": seq.unsqueeze(0)}  # Añadir dimensión de batch
        
        try:
            network.run(inputs=inputs, time=T)
        except Exception as e:
            print(f"Error durante la ejecución:")
            print(f"- Forma de entrada: {inputs['A'].shape}")
            print(f"- Neuronas capa A: {network.layers['A'].n}")
            print(f"- Tiempo T: {T}")
            raise
        
        # Recolectar spikes
        spikes = {
            "X": source_monitor.get("s"),
            "B": target_monitor.get("s"),
            "C": conv_monitor.get("s")
        }
        sp0.append(spikes["X"].sum(axis=2))
        sp1.append(spikes["B"].sum(axis=2))
        sp2.append(spikes["C"].sum(axis=2))
    
    # Convertir a numpy
    sp0 = torch.cat(sp0).detach().cpu().numpy()
    sp1 = torch.cat(sp1).detach().cpu().numpy()
    sp2 = torch.cat(sp2).detach().cpu().numpy()
    conv_voltage = conv_monitor.get("v").detach().cpu().numpy()
    
    return [sp0, sp1, sp2, conv_voltage, network]



def convertir_data_convo(data, T, cuantiles, R, device, is_train=False):
    # Convertir la serie temporal a tensor
    serie = torch.FloatTensor(data['value'])
    
    # Aplicar clipping a los valores extremos
    serie[serie < torch.min(cuantiles)] = torch.min(cuantiles)
    serie[serie > torch.max(cuantiles)] = torch.max(cuantiles)
    
    # Crear representación one-hot por rangos de cuantiles (forma: [pasos_temporales, R])
    serie_expanded = torch.zeros((len(serie), R), dtype=torch.float32)
    
    for i in range(R):
        lower = cuantiles[i]
        upper = cuantiles[i+1] if i < R-1 else torch.max(cuantiles) + 1e-6
        serie_expanded[:, i] = ((serie >= lower) & (serie < upper)).float()
    
    # Ajustar la longitud para que sea múltiplo de T
    total_steps = serie_expanded.shape[0]
    num_sequences = total_steps // T
    trimmed_length = num_sequences * T
    
    # Recortar o hacer padding en el eje temporal
    if trimmed_length < total_steps:
        trimmed_series = serie_expanded[:trimmed_length, :]
    else:
        padding = trimmed_length - total_steps
        trimmed_series = torch.nn.functional.pad(serie_expanded, (0, 0, 0, padding))
    
    # Dividir en secuencias de forma [num_sequences, T, R]
    sequences = trimmed_series.view(num_sequences, T, R)
    
    # Eliminar última secuencia en entrenamiento
    if is_train and len(sequences) > 1:
        sequences = sequences[:-1]
    
    # Mover a dispositivo y convertir a lista
    secuencias = [seq.to(device) for seq in sequences]
    
    return secuencias


def podar(x,q1,q2,cuantiles=None):
    #Función que devuelve 1 (spike) si x está en el rango [q1,q2), y 0 en caso contrario.
    #Es parte de la codificación de los datos.
    
    s=torch.zeros_like(x)
    
    s[(x>=q1) & (x<q2)]=1
    return s
