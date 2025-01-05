import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import bindsnet

print(torch.__version__)
print(torchvision.__version__)
print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)
print(matplotlib.__version__)

# Verify bindsnet installation by importing a specific component
from bindsnet.network import Network
print("bindsnet is installed correctly")