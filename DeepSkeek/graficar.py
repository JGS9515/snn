import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Construir la ruta correcta al archivo
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train_label_filled.csv')

# Cargar los datos
df = pd.read_csv(csv_path)

# Convertir timestamp a datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# Separar valores normales y anomalías 
normal_data = df[df["label"] == 0]
anomalies = df[df["label"] == 1]

# Configurar el gráfico
plt.figure(figsize=(18, 6))
plt.title("Flujo de personas en CalIt2 con detección de anomalías")
plt.xlabel("Fecha")
plt.ylabel("Número de personas")

# Graficar serie temporal completa
plt.plot(df["timestamp"], df["value"], 
         color="steelblue", 
         linewidth=0.8,
         label="Flujo normal")

# Resaltar anomalías
plt.scatter(anomalies["timestamp"], anomalies["value"],
            color="red",
            s=15,
            label="Anomalías",
            zorder=3)

# Configurar el formato de fecha adaptativo
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))

# Rotar las etiquetas para mejor legibilidad
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.2)  # Ajustar espacio para las etiquetas


# Añadir elementos adicionales
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()