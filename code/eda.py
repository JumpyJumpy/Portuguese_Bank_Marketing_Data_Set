import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# %matplotlib inline

pd.set_option("display.max_columns", None)
#%% raw

#%%
bank = pd.read_csv("./data/bank.csv")

bank.info()
#%%
num_var = bank.columns[bank.dtypes != "object"].to_list()
num_var.pop(num_var.index("y"))
cate_var = bank.columns[bank.dtypes == "object"].to_list()
print(f'{len(num_var)}, {len(cate_var)}')
#%%
bank["y"] = bank["y"].map({1: "yes", 0: "no"})

for col in num_var:
    sns.stripplot(x = "age", y = "y", data = bank, jitter = True)
#%% md
"""
## Correlation Heatmap
There are too many variables, it's hard to read.
More detailed and readable plots will follow.
**However, from the graph, it still can be seen that most variables are not correlated with other variables.**
"""
#%%
bank_flattened = pd.read_csv("./data/bank_imputed_knn.csv")

plt.figure(figsize = (20, 16))
sns.set(font_scale = 2)
sns.heatmap(bank_flattened.corr(), cmap = "YlGnBu", linewidth = 1)
plt.show()
#%% md

#%%
