import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
bank = pd.read_csv("./data/Casestudy Data.csv")

bank.describe()
bank.info()
bank_X = bank.drop(columns = ["y"])
bank_y = bank["y"]

cate_features = []

for col in bank_X.columns:
    if bank_X[col].dtype == "object":
        bank_X[col] = bank_X[col].astype("category")
        cate_features.append(col)

bank_X.info()
np.sum(bank_X.isna())

le = LabelEncoder()
mapping = {}
for col in cate_features:
    le.fit(bank_X[col])
    levels = list(le.classes_)
    mapping[col] = {i: le.transform([i]).tolist()[0] for i in levels}

    bank_X.loc[bank_X[col][bank_X[col].notnull()].index, col] = pd.Series(le.transform(bank_X[col][bank_X[col].notnull()]),
    index = bank_X[col][bank_X[col].notnull()].index)


np.sum(bank_X.isna())
bank_X.describe()
bank_X.info()
cate_features

imputer = KNNImputer(weights = "distance")
bank_X_imputed_knn = pd.DataFrame(imputer.fit_transform(bank_X), columns = bank_X.columns)
bank_X_imputed_knn.to_csv("bank_X_imputed_knn.csv")

import json
with open("mapping.json", "w") as file:
    file.write(json.dumps(mapping))
