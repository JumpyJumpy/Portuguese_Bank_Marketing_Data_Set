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
fig, ax = plt.subplots(2, 2, figsize = (20, 10))
for i in range(len(num_var)):
    if i < 7:
        sns.stripplot(x = num_var[i], y = "y", data = bank, jitter = True, ax = ax[0, i])
        break
    else:
        sns.stripplot(x = num_var[i], y = "y", data = bank, jitter = True, ax = ax[1, i - 7])

plt.show()
#%% md

#%%
