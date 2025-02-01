import torch, pandas as pd, numpy as np, os
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre

#Código ppal para lanzar la experimentación para la detección de anomalías con bindsnet y STDP.
#Este código se usaría como base para iterar sobre las distintas combinaciones de parámetros.
#'nuu.csv'#
path='Nuevos datasets/iops/preliminar/train_procesado_javi/1c35dbf57f55f5e4_filled.csv'

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
r=0.01

#Número de neuronas en la capa B.
n=1000

#Umbral de disparo de las neuronas LIF:
umbral=-52

#Decaimiento, en tiempo, de las neuronas LIF:
decaimiento=100

T = 1000 #Tiempo de exposición. Puede influir por la parte del entrenamiento, en la inferencia no porque los voltajes se conservan.
#Usar el máximo de T para evitar problemas con los periodos de datos.
expansion=100

#Construimos las tuplas n1 y n2 para pasar al modelo:
nu1=(nu1_pre,nu1_post)
nu2=(nu2_pre,nu2_post)

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


def convertir_data(data,T,cuantiles,R,is_train=False):
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
    serie2input=torch.cat([serie.unsqueeze(0)] * R, dim=0)
    
    for i in range(R):
        serie2input[i,:]=podar(serie2input[i,:],cuantiles[i],cuantiles[i+1])
    
    #Lo dividimos en función del tiempo de exposición T:
    secuencias = torch.split(serie2input,T,dim=1)
    
    if is_train:
        #Encaso de estar entrenando, tendríamos que quitar la última secuencia:
        secuencias=secuencias[0:len(secuencias)-1]
    
    return secuencias


def crear_red():
    #Aquí creamos la red.
    
    network = Network()
    
    #Creamos las capas de entrada e interna:
    source_layer = Input(n=R,traces=True)
    target_layer = LIFNodes(n=n,traces=True,thresh=umbral, tc_decay=decaimiento)
    
    network.add_layer(
        layer=source_layer, name="A"
    )
    network.add_layer(
        layer=target_layer, name="B"
    )
    
    #Creamos conexiones entre las capas de entrada y la recurrente:
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
    
    #Creamos los monitores. Sirven para registrar los spikes y voltajes:
    #Spikes de entrada (para depurar que se esté haciendo bien, si se quiere):
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
    
    network.add_monitor(monitor=source_monitor, name="X")
    network.add_monitor(monitor=target_monitor, name="Y")
    
    
    return [network,source_monitor,target_monitor]


def ejecutar_red(secuencias,network,source_monitor,target_monitor,T):
    #Función para ejecutar la red con los datos que se quieran, ya sea para entrenamiento o evaluación.
    
    #Creamos los objetos lista en que almacenaremos los resultados:
    sp0=[]
    sp1=[]
    
    #Entrenamos:
    j=1
    for i in secuencias:
        #Los datos de entrada serán una tupla con tensores de pytorch, pasamos cada una:
        print(f'Ejecutando secuencia {j}')
        j+=1
        inputs={'A':i.T}
        network.run(inputs=inputs, time=T)
        
        #Obtenemos los spikes a lo largo de la simulación:
        spikes = {
            "X": source_monitor.get("s"), "B": target_monitor.get("s")
        }
        sp0.append(spikes['X'].sum(axis=2))
        sp1.append(spikes['B'].sum(axis=2))
        voltages = {"Y": target_monitor.get("v")}
        #Reseteo voltajes, venga:
        network=reset_voltajes(network)
        
    
    #Concatenamos y devolvemos:
    sp0=torch.cat(sp0)
    sp0=sp0.detach().numpy()
    
    sp1=torch.cat(sp1)
    sp1=sp1.detach().numpy()
    
    return [sp0,sp1,network]


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

#Ahora, establecemos el valor de R, que será el número de neuronas de la capa de entrada:
R=len(cuantiles)-1

#Crea la red.
network, source_monitor,target_monitor = crear_red()

#Dividimos el train en secuencias:
data_train=dividir(data_train,T)

#Paddeamos el test:
data_test=padd(data_test,T)

#En este punto, entrenamos para cada secuencia consecutiva del train:

#Para cada secuencia del train, tenemos que pasarla y entrenar la red:
network.learning=True

for s in data_train:
    secuencias2train=convertir_data(s,T,cuantiles,R,is_train=True)
    print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
    spikes_input,spikes,network=ejecutar_red(secuencias2train,network,source_monitor,target_monitor,T)
    #Reseteamos los voltajes:
    network=reset_voltajes(network)

#Ahora, el test:
network.learning=False
secuencias2test=convertir_data(data_test,T,cuantiles,R)

print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
spikes_input,spikes,network=ejecutar_red(secuencias2test,network,source_monitor,target_monitor,T)

np.savetxt('resultados/ejecutar_experimento/spikes',spikes,delimiter=',')

np.savetxt('resultados/ejecutar_experimento/label',np.array(data_test['label']),delimiter=',')

np.savetxt('resultados/ejecutar_experimento/value',np.array(data_test['value']),delimiter=',')

with open('resultados/ejecutar_experimento/n1','w') as n1:
    n1.write(f'{R}\n')

with open('resultados/ejecutar_experimento/n2','w') as n2:
    n2.write(f'{n}\n')
