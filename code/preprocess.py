import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

pd.set_option("display.max_columns", None)
bank = pd.read_csv("./data/Casestudy Data.csv")

bank.shape
bank.head
bank.describe()
bank.info()

bank = bank.replace("unknown", value = np.nan)

# drop "duration" column based on the description
bank = bank.drop(column = ["duration"])

bank.to_csv("./data/bank.csv", index = False)

bank_X = bank.drop(columns = ["y"])
bank_y = bank["y"]
cate_features = []

for col in bank_X.columns:
    if bank_X[col].dtype == "object":
        bank_X[col] = bank_X[col].astype("category")
        cate_features.append(col)

bank_X.info()
np.sum(bank_X.isna())

# Generating dummy variables
bank_X_flattened = pd.get_dummies(bank_X, dummy_na = False)
bank_X_flattened.to_csv("./data/bank_X_flattened.csv", index = False)
"""
# label encoding
# not working if we want to do KNN imputation or logistic regression

le = LabelEncoder()
mapping = {}
for col in cate_features:
    le.fit(bank_X[col][bank_X[col].notna()])
    levels = list(le.classes_)
    mapping[col] = {i: le.transform([i]).tolist()[0] for i in levels}

    bank_X.loc[bank_X[col][bank_X[col].notna()].index, col] = \
        pd.Series(le.transform(bank_X[col][bank_X[col].notna()]), index = bank_X[col][bank_X[col].notna()].index)

np.sum(bank_X.isna())
bank_X.describe()
bank_X.info()
cate_features
"""
# KNN imputation
imputer = KNNImputer(weights = "distance")
bank_X_imputed_knn = pd.DataFrame(imputer.fit_transform(bank_X_flattened), columns = bank_X_flattened.columns)
bank_X_imputed_knn.join(bank_y).to_csv(path_or_buf = "./data/bank_imputed_knn.csv", index = False)

# iterative imputation
imp = IterativeImputer()
bank_X_imputed_imp = pd.DataFrame(imp.fit_transform(bank_X_flattened), columns = bank_X_flattened.columns)
bank_X_imputed_imp.join(bank_y).to_csv("./data/bank_imputed_imp.csv", index = False)
