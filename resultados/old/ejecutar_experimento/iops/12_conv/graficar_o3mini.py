import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Cargar los archivos CSV (asumiendo que se llaman "data_test.csv", "results.csv" y "results_conv.csv")
base_path = os.path.dirname(__file__)

data_test = pd.read_csv(os.path.join(base_path,"data_test.csv"))
results = pd.read_csv(os.path.join(base_path,"results.csv"))
results_conv = pd.read_csv(os.path.join(base_path,"results_conv.csv"))

# Convertir el timestamp a formato datetime si es necesario
data_test['datetime'] = pd.to_datetime(data_test['timestamp'], unit='s')
results['datetime'] = pd.to_datetime(results['timestamp'], unit='s')
results_conv['datetime'] = pd.to_datetime(results_conv['timestamp'], unit='s')

# # 2. Gráfica de la serie temporal original y detección de anomalías

# plt.figure(figsize=(12,6))
# plt.plot(data_test['datetime'], data_test['value'], label="Datos de prueba", color="blue")

# # Suponiendo que el DataFrame results tenga columna 'label' que indique anomalías
# anom_data_test = data_test[results['label'] == 1]
# anom_results = results[results['label'] == 1]
# anom_results_conv = results_conv[results['label'] == 1]
# plt.scatter(anom_data_test['datetime'], anom_data_test['value'], color='red', marker='o', s=50, label="Anomalías data_test")
# plt.scatter(anom_results['datetime'], anom_results['value'], color='blue', marker='o', s=50, label="Anomalías results")
# plt.scatter(anom_results_conv['datetime'], anom_results_conv['value'], color='yellow', marker='o', s=50, label="Anomalías results_conv")

# plt.xlabel("Tiempo")
# plt.ylabel("Valor")
# plt.title("Serie temporal y anomalías detectadas (Capa A y B)")
# plt.legend()
# plt.show()

# 2. Gráfica de la serie temporal original y detección de anomalías

plt.figure(figsize=(12,6))
plt.plot(data_test['datetime'], data_test['value'], label="Datos de prueba", color="blue", alpha=0.6)

# Crear pequeños offsets para separar los puntos
offset1 = 0.1  # Para data_test
offset2 = 0.2  # Para results
offset3 = 0.3  # Para results_conv

# Suponiendo que el DataFrame results tenga columna 'label' que indique anomalías
anom_data_test = data_test[results['label'] == 1]
anom_results = results[results['label'] == 1]
anom_results_conv = results_conv[results['label'] == 1]

# Graficar con offsets
plt.scatter(anom_data_test['datetime'], 
           anom_data_test['value'] + offset1,
           color='red', 
           marker='v', 
           s=600, 
           label="Anomalías data_test",
           alpha=0.7)

plt.scatter(anom_results['datetime'], 
           anom_results['value'] + offset2,
           color='green', 
           marker='s', 
           s=300, 
           label="Anomalías results",
           alpha=0.7)

plt.scatter(anom_results_conv['datetime'], 
           anom_results_conv['value'] + offset3,
           color='orange', 
           marker='*', 
           s=100, 
           label="Anomalías results_conv",
           alpha=0.7)

plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.title("Serie temporal y anomalías detectadas (Capa A y B)")
plt.legend()
plt.show()

# # 3. Comparación entre resultados de la red sin y con capa convolucional

# fig, ax = plt.subplots(2, 1, figsize=(12,10), sharex=True)

# # Primera gráfica: resultados sin la capa convolucional
# ax[0].plot(results['datetime'], results['value'], color="green", label="Resultados Capa A y B")
# ax[0].set_title("Resultados SNN sin capa convolucional")
# ax[0].set_ylabel("Valor")
# ax[0].legend()

# # Segunda gráfica: resultados con la capa convolucional
# ax[1].plot(results_conv['datetime'], results_conv['value'], color="purple", label="Resultados con capa convolucional")
# ax[1].set_title("Resultados SNN con capa convolucional")
# ax[1].set_xlabel("Tiempo")
# ax[1].set_ylabel("Valor")
# ax[1].legend()

# plt.tight_layout()
# plt.show()

# # 4. Visualización comparativa en una sola figura

# plt.figure(figsize=(12,6))
# plt.plot(data_test['datetime'], data_test['value'], label="Datos originales", color="blue", alpha=0.6)
# plt.plot(results['datetime'], results['value'], label="SNN (capa A+B)", color="green")
# plt.plot(results_conv['datetime'], results_conv['value'], label="SNN + capa convolucional", color="purple")
# plt.xlabel("Tiempo")
# plt.ylabel("Valor")
# plt.title("Comparación de la serie original y resultados SNN")
# plt.legend()
# plt.show()

# # 5. Gráficos de distribución y análisis estadístico

# plt.figure(figsize=(10,5))
# sns.histplot(results['value'], bins=50, kde=True, color="blue")
# plt.xlabel("Valores (Resultados sin capa convolucional)")
# plt.title("Distribución de resultados SNN (capa A+B)")
# plt.show()

# 6. Visualizaciones interactivas

# plt.figure(figsize=(10,5))
# sns.histplot(results_conv['value'], bins=50, kde=True, color="purple")
# plt.xlabel("Valores (Resultados con capa convolucional)")
# plt.title("Distribución de resultados SNN con capa convolucional")
# plt.show()


# fig = px.line(results, x="datetime", y="value", title="Resultados SNN sin capa convolucional")
# fig.show()

# fig2 = px.line(results_conv, x="datetime", y="value", title="Resultados SNN con capa convolucional")
# fig2.show()