import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import RFECV, RFE


from sklearn.ensemble import RandomForestClassifier


bank_logit = pd.read_csv("./data/bank_selected.csv")
bank_raw = pd.read_csv("./data/bank_imputed_imp.csv")

bank_logit.head()

bank_X = bank_logit.drop(columns = ["y"])
bank_y = bank_logit["y"]
bank_raw_X = bank_raw.drop(columns = ["y"])
bank_raw_y = bank_raw["y"]

logit = LogisticRegressionCV(penalty = "l1", solver = "liblinear").fit(bank_X, bank_y)
logit.score(bank_X, bank_y)
logit_coef = bank_X.columns[logit.coef_[0] != 0]


rf = RandomForestClassifier(criterion = "entropy").fit(bank_raw_X, bank_raw_y)
rf.score(bank_raw_X, bank_raw_y)


selector = RFE(RandomForestClassifier(criterion = "entropy")).fit(bank_raw_X, bank_raw_y)
rf_coef = bank_raw_X.columns[selector.support_]

rf = RandomForestClassifier(criterion = "entropy").fit(bank_raw_X[rf_coef], bank_raw_y)
rf.score(bank_raw_X[rf_coef], bank_raw_y)
feature_importance = np.sort(np.stack((rf_coef, rf.feature_importances_), axis = 1), axis = 0)[::-1]
feature_importance.sum(axis = 1)
