import torch, pandas as pd, numpy as np, os
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
import torch.nn.functional as F

import numpy as np

from utils import *

#Código ppal para lanzar la experimentación para la detección de anomalías con bindsnet y STDP.
#Este código se usaría como base para iterar sobre las distintas combinaciones de parámetros.
#'nuu.csv'#
path='Nuevos datasets/iops/preliminar/train_procesado_javi/1c35dbf57f55f5e4_filled.csv'
# path='Nuevos datasets/Callt2/preliminar/train_label_filled.csv'

#Establecemos valores para los parámetros que nos interesan:
nu1_pre=0.1 #Actualización de pesos presinápticos en la capa A. Valores positivos penalizan y negativos excitan.
nu1_post=-0.1 #Actualización de pesos postsinápticos en la capa A. Valores postivos excitan y negativos penalizan.

nu2_pre=0.1 #Actualización de pesos presinápticos en la capa B. Valores positivos penalizan y negativos excitan.
nu2_post=-0.1 #Actualización de pesos postsinápticos en la capa B. Valores postivos excitan y negativos penalizan.

#Parámetros que definen la amplitud del rango de cuantiles.
#La idea es que el valor mínimo para la codificación sea inferior al mínimo de los datos de entrenamiento, por un margen. El valor máximo debe ser también  mayor que el máximo de los datos por un margen.
#Para ello, nos inventamos la variable a, que será la proporción del rango de datos de entrenamiento que inflamos por encima y por debajo:
a=0.1
#La resolución, r, indica cuán pequeños tomamos los rangos al codificar:
r=0.05

#Número de neuronas en la capa B.
n=200

#Umbral de disparo de las neuronas LIF:
threshold=-52

#Decaimiento, en tiempo, de las neuronas LIF:
decay=100

T = 250 #Tiempo de exposición. Puede influir por la parte del entrenamiento, en la inferencia no porque los voltajes se conservan.
#Usar el máximo de T para evitar problemas con los periodos de datos.
expansion=100

#Construimos las tuplas n1 y n2 para pasar al modelo:
nu1=(nu1_pre,nu1_post)
nu2=(nu2_pre,nu2_post)

#Lectura de datos:
#Esperamos que estos datos tengan las columnas 'label' y 'value'.

data=pd.read_csv(path,na_values=['NA'])

#Asegurarse de que los tipos sean correctos:
data['value']=data['value'].astype('float64')
data['label']=data['label'].astype('Int64')

#Y ponemos a 0 los valores nulos del label para no tener problemas al filtrar por esta columna:
data.loc[data['label'].isna(),'label']=0

split = len(data) // 2

data_train=data[:split]
data_test=data[split:]

#Reseteamos el índice:
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

#Expandimos:
data_train['label']=expandir(data_train['label'],expansion)

#Sacamos máximos y mínimos:
minimo=min(data_train['value'][data_train['label']!=1])
maximo=max(data_train['value'][data_train['label']!=1])

#Declaramos el vector de cuantiles. Para ello, tomamos el máximo y mínimo de los datos de entrenamiento (esto hay que sacarlo de esos datos)

amplitud=maximo-minimo
cuantiles=torch.FloatTensor(np.arange(minimo-a*amplitud,maximo+amplitud*a,(maximo-minimo)*r))

#Ahora, establecemos el valor de snn_input_layer_neurons_size, que será el número de neuronas de la capa de entrada:
snn_input_layer_neurons_size=len(cuantiles)-1

#Crea la red.
network, source_monitor, target_monitor, conv_monitor = crear_red(snn_input_layer_neurons_size,decay,threshold,nu1,nu2,n,T)

#Dividimos el train en secuencias:
data_train=dividir(data_train,T)

#Paddeamos el test:
data_test=padd(data_test,T)

#En este punto, entrenamos para cada secuencia consecutiva del train:

#Para cada secuencia del train, tenemos que pasarla y entrenar la red:
network.learning=True

for s in data_train:
    secuencias2train=convertir_data(s,T,cuantiles,snn_input_layer_neurons_size,is_train=True)
    print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
    spikes_input,spikes,spikes_conv,network=ejecutar_red(secuencias2train,network,source_monitor,target_monitor,conv_monitor,T)
    #Reseteamos los voltajes:
    network=reset_voltajes(network)

#Ahora, el test:
network.learning=False
secuencias2test=convertir_data(data_test,T,cuantiles,snn_input_layer_neurons_size)

print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
spikes_input,spikes,spikes_conv,network=ejecutar_red(secuencias2test,network,source_monitor,target_monitor,conv_monitor,T)

guardar_resultados(spikes,spikes_conv,data_test,n,snn_input_layer_neurons_size)