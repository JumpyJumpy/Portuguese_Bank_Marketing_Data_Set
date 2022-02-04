# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

pd.set_option("display.max_columns", None)

# %% [markdown]
# 

# %%


bank = pd.read_csv("../data/bank.csv")

bank.info()

# %%



num_var = bank.columns[bank.dtypes != "object"].to_list()
num_var.pop(num_var.index("y"))
cate_var = bank.columns[bank.dtypes == "object"].to_list()
print(f'{len(num_var)}, {len(cate_var)}')

# %%


bank["y"] = bank["y"].map({1: "yes", 0: "no"})
fig, ax = plt.subplots(3, 5, figsize = (60, 20))
sns.set_theme(style="white")
for i in range(len(num_var)):
    if i < 5:
        sns.stripplot(x = num_var[i], y = "y", data = bank, jitter = True, ax = ax[0, i], s = 5)
    elif i < 10:
        sns.stripplot(x = num_var[i], y = "y", data = bank, jitter = True, ax = ax[1, i - 5], s = 5)
    else:
        sns.stripplot(x = num_var[i], y = "y", data = bank, jitter = True, ax = ax[2, i - 10], s = 5)




# %%

for i in range(len(num_var)):
    sns.stripplot(x = num_var[i], y = "y", data = bank, jitter = True, s = 1)
plt.tight_layout()


# %%
fig, ax = plt.subplots(2, 3, figsize = (30, 15))
for i in range(len(num_var)):
    if i < 3:
        sns.stripplot(x = cate_var[i], y = "y", data = bank, jitter = True, ax = ax[0, i])
    else:
        sns.stripplot(x = cate_var[i], y = "y", data = bank, jitter = True, ax = ax[1, i - 3])

# %% [markdown]
# ## Correlation Heatmap
# There are too many variables, it's hard to read.
# More detailed and readable plots will follow.
# **However, from the graph, it still can be seen that most variables are not correlated with other variables.**

# %%
bank_flattened = pd.read_csv("../data/bank_imputed_imp.csv")

plt.figure(figsize = (20, 16))
sns.set(font_scale = 1)
sns.heatmap(bank_flattened.corr(), cmap = "YlGnBu", linewidth = 1)
plt.show()

# %% [markdown]
# 

# %%
