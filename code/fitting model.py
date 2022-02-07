import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import RFECV, RFE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

bank_logit = pd.read_csv("./data/bank_selected.csv")
bank_raw = pd.read_csv("./data/bank_imputed_imp.csv")

bank_logit.head()

bank_X = bank_logit.drop(columns = ["y"])
bank_y = bank_logit["y"]
bank_raw_X = bank_raw.drop(columns = ["y"])
bank_raw_y = bank_raw["y"]


# Logistic regression
logit = LogisticRegressionCV(penalty = "l1", solver = "liblinear").fit(bank_X, bank_y)  # L1 norm to get sparse solution
logit.score(bank_X, bank_y)
logit_coef = bank_X.columns[logit.coef_[0] != 0]

selector = RFE(LogisticRegression(max_iter = 100))
selector = selector.fit(bank_raw_X, bank_raw_y)
selector.support_
selector.ranking_
# Random forest
rf = RandomForestClassifier(criterion = "entropy").fit(bank_raw_X, bank_raw_y)
rf.score(bank_raw_X, bank_raw_y)

selector = RFE(RandomForestClassifier(criterion = "entropy")).fit(bank_raw_X, bank_raw_y)
rf_coef = bank_raw_X.columns[selector.support_]

rf = RandomForestClassifier(criterion = "entropy").fit(bank_raw_X[rf_coef], bank_raw_y)
rf.score(bank_raw_X[rf_coef], bank_raw_y)
feature_importance = np.sort(np.stack((rf_coef, rf.feature_importances_), axis = 1), axis = 0)[::-1]
np.sum(feature_importance[:, 1])

feature = [i[0] for i in feature_importance]
importance = [i[1] for i in feature_importance]

plt.figure(figsize = (12, 8))
plt.barh(feature, importance, height = 0.5)
plt.gca().invert_yaxis()
plt.gca()
plt.show()
