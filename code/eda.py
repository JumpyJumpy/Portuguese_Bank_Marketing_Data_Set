import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

bank_X_flattened = pd.read_csv("./data/bank_X_flattened.csv")


plt.figure(figsize = (16, 16))
sn.heatmap(bank_X_flattened.corr(), cmap = "YlGnBu", linewidth = 1)
plt.show()
