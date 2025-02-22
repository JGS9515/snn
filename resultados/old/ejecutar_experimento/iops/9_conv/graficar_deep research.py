import matplotlib.pyplot as plt
import os
import pandas as pd



base_path = os.path.dirname(__file__)

base_path = os.path.dirname(__file__)

data_test = pd.read_csv(os.path.join(base_path,"data_test.csv"))
results = pd.read_csv(os.path.join(base_path,"results.csv"))
results_conv = pd.read_csv(os.path.join(base_path,"results_conv.csv"))

fig, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(data_test['timestamp'], data_test['value'], color='blue', label='Serie Original')
ax2 = ax1.twinx()
ax2.plot(results['timestamp'], results['label'], color='orange', linestyle='--', label='SNN Base')
ax2.plot(results_conv['timestamp'], results_conv['label'], color='red', linestyle='-.', label='SNN + Conv')
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.title('Comparación de Puntuaciones de Anomalía entre Modelos')
plt.show()
