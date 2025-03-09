import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import optuna
import bindsnet

print(f'torch: {torch.__version__}')
print(f'{torchvision.__version__}')
print(f'numpy: {np.__version__}')
print(f'pandas: {pd.__version__}')
print(f'sklearn: {sklearn.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print(f'optuna: {optuna.__version__}')


# Verify bindsnet installation by importing a specific component
from bindsnet.network import Network
print("bindsnet is installed correctly")

# !pip install torch==2.6.0 pandas==1.4.3 numpy==1.23.5 optuna==4.2.0
