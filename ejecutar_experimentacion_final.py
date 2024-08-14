import torch, pandas as pd, numpy as np, os, argparse
from sklearn.model_selection import TimeSeriesSplit
from dependencias import *

#Código que realiza la experimentación final.
#Los parámetros configurables se introducirán por línea de comandos, como con los pollos.
#Lectura de las variables iniciales:
parser = argparse.ArgumentParser(description='Experimentación con STDP.')

parser.add_argument('-n1', '--nu1', type=str,help='Parámetro nu de la primera capa.')
parser.add_argument('-n2', '--nu2', type=str, help='Parámetro nu de la segunda capa.')
parser.add_argument('-n','--neurons',type=int,help='Cantidad de neuronas de la segunda capa.')
parser.add_argument('-th', '--threshold', type=float, help='Umbral de disparo de las neuronas.')
parser.add_argument('-d', '--decay', type=float, help='Tiempo de decaimiento del voltaje.')
parser.add_argument('-a', '--ampliacion', type=float, help='Porcentaje de ampliación respecto al rango de datos.')
parser.add_argument('-r', '--resolucion', type=float, help='Resolución de la entrada de datos.')
parser.add_argument('-p', '--path', type=str, help='Ruta de los datos.')
parser.add_argument('-e', '--epochs', type=int,help='Epochs.')

#Evalúa argumentos introducidos:
args=parser.parse_args()
nu1=eval(args.nu1)
nu2=eval(args.nu2)
n=args.neurons
threshold=args.threshold
decay=args.decay
amp=args.ampliacion
path=args.path
epochs=args.epochs
reso=args.resolucion

#Creemos el directorio de salida:
nu1_str=str(nu1).replace('(','').replace(',','_').replace(')','')
nu2_str=str(nu2).replace('(','').replace(',','_').replace(')','')

output_path=f'output/{path}/{nu1_str}/{nu2_str}/{threshold}/{decay}/{amp}/{epochs}/{reso}'
os.makedirs(output_path, exist_ok=True)

T=100 #Valor razonablemente pequeño para no eliminar muchas muestras del entremiento.

#Ahora, es necesario leer los datos de entrada:
#Tienen que tener las columnas 'label' y 'value'.

data=pd.read_csv(path,na_values=['NA'])

#Asegurarse de que los tipos sean correctos:
data['value']=data['value'].astype('float64')
data['label']=data['label'].astype('Int64')

#Anulamos los valores ausentes del label para que funcione bien.
data.loc[data['label'].isna(),'label']=0

#Preparación de la validación cruzada:
tscv=TimeSeriesSplit(n_splits=5)

split=0
for train_index, test_index in tscv.split(data):
    split+=1
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    
    #Reseteamos el índice:
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    
    data_train['value']=data_train['value'].astype('float64')
    data_test['label']=data_test['label'].astype('Int64')
    
    #Sacamos máximos y mínimos:
    minimo=min(data_train['value'][data_train['label']!=1])
    maximo=max(data_train['value'][data_train['label']!=1])
    
    #Anulamos valores donde el label sea 1, para el conjunto de entrenamiento:
    data_train.loc[data_train['label']==1,'value']=np.nan
    
    #Modelamos los rangos de las entradas:
    ancho_datos=maximo-minimo
    cuantiles=torch.FloatTensor(np.arange(minimo-amp*ancho_datos,maximo+ancho_datos*amp,(maximo-minimo)*reso))
    
    #Ahora, establecemos el valor de R, que será el número de neuronas de la capa de entrada:
    R=len(cuantiles)-1
    
    #Crea la red.
    network, source_monitor,target_monitor = crear_red(R,T,n,threshold,decay,nu1,nu2)
    
    #Paddeamos el test:
    data_test=padd(data_test,T)
    
    #Para cada secuencia del train, tenemos que pasarla y entrenar la red:
    network.learning=True
    
    #Procesemos secuencias de entrenamiento:
    secuencias2train=convertir_data(data_train,T,cuantiles,R,is_train=True)
    print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
    for e in range(epochs):
        print(f'Epoch {e}')
        spikes_input,spikes,network=ejecutar_red(secuencias2train,network,source_monitor,target_monitor,T)
        #Tras cada epoch, resetea voltajes:
        reset_voltajes(network)
    
    #Ahora, el test:
    network.learning=False
    secuencias2test=convertir_data(data_test,T,cuantiles,R)
    
    print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
    spikes_input,spikes,network=ejecutar_red(secuencias2test,network,source_monitor,target_monitor,T)
    
    #Ahora, guarda resultados convenientemente:
    np.savetxt(f'{output_path}/spikes_{split}',spikes,delimiter=',')
    np.savetxt(f'{output_path}/label_{split}',np.array(data_test['label']),delimiter=',')
    np.savetxt(f'{output_path}/value_{split}',np.array(data_test['value']),delimiter=',')

