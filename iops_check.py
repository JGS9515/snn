import pandas as pd
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocesamiento/iops'))

# Importar la función get_unique_kpi_ids desde el script iops_check_different_KPIs
from iops_check_different_KPIs import iops_check_different_KPIs
from verify_that_all_KPI_files_exist import verify_that_all_KPI_files_exist
from create_new_dataset_per_every_different_kpi import create_new_dataset_per_every_different_kpi
from iops_fill_missing_timestamps import iops_fill_missing_timestamps


result = iops_check_different_KPIs()
unique_kpi_ids = result['unique_kpi_ids']
df = result['df']
verify_that_all_KPI_files_exist(unique_kpi_ids)

create_new_dataset_per_every_different_kpi(df,unique_kpi_ids)
#Si el valor es True, se añadirá una columna 'status' al DataFrame, si es false no se añadirá. El objetivo de añadir esta columna es para verificar cuantas columnas se han añadido.
for kpi in unique_kpi_ids:
    iops_fill_missing_timestamps(kpi, False)

# # Filtrar el DataFrame para obtener las filas donde 'KPI ID' es '02e99bd4f6cfb33f'
# filtered_df = df[df['KPI ID'] == '02e99bd4f6cfb33f']

# # Obtener los valores de 'timestamp' correspondientes

# # timestamps = filtered_df['timestamp'].values

# # # Imprimir los valores de 'timestamp'
# # print('Timestamps para KPI ID 02e99bd4f6cfb33f:')
# # print(timestamps)
        

# output_path = '/home/javier/Practicas/Nuevos datasets/iops/preliminar/train_procesado_javi/02e99bd4f6cfb33f.csv'

# # # Seleccionar solamente las primeras 3 columnas
# # filtered_df_first_3_columns = filtered_df.iloc[:, :3]

# # # Guardar las filas filtradas en un nuevo archivo CSV con solo las primeras 3 columnas
# # filtered_df_first_3_columns.to_csv(output_path, index=False)

# # Leer el archivo CSV
# df = pd.read_csv(output_path)

# # Asegurarse de que la columna 'timestamp' está en formato numérico
# df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# # Ordenar las filas por la columna 'timestamp'
# df = df.sort_values(by='timestamp')

# # Calcular la diferencia entre valores consecutivos de 'timestamp'
# timestamp_diff = df['timestamp'].diff().dropna()

# # Filtrar las diferencias que son mayores de 60
# diff_greater_than_60 = timestamp_diff[timestamp_diff > 60]

# # Mostrar los índices correspondientes a esas diferencias
# indices = diff_greater_than_60.index
# timestamps = df.loc[indices, 'timestamp']

# print('Índices y valores de timestamp donde la diferencia es mayor de 60:')
# for index, timestamp in zip(indices, timestamps):
#     print(f'Índice: {index}, Timestamp: {timestamp}')

# # Verificar si todas las diferencias son iguales a 60
# # all_diffs_are_60 = (timestamp_diff == 60).all()

# # # Imprimir el resultado
# # if all_diffs_are_60:
# #     print('Todas las diferencias de timestamp son 60.')
# # else:
# #     print('No todas las diferencias de timestamp son 60.')
# #     # Imprimir las diferencias que no son 60
# #     print('Diferencias de timestamp:')
# #     print(timestamp_diff[timestamp_diff != 60])