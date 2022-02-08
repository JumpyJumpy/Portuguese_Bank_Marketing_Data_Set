import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.feature_selection import RFECV, RFE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

bank_logit = pd.read_csv("./data/bank_selected.csv")
bank_raw = pd.read_csv("./data/bank_imputed_imp.csv")

bank_logit.head()

bank_X = bank_logit.drop(columns = ["y"])
bank_y = bank_logit["y"]
bank_raw_X = bank_raw.drop(columns = ["y"])
bank_raw_y = bank_raw["y"]

weight = bank_y.value_counts()[0] / bank_y.value_counts()[1]

# Logistic regression
logit = LogisticRegression(penalty = "none", max_iter = 500, class_weight = {1: weight, 0: 1})
logit = logit.fit(bank_X, bank_y)
logit.score(bank_X, bank_y)

selector = RFE(LogisticRegression(max_iter = 1000, class_weight = {1: weight, 0: 1}))
selector = selector.fit(bank_X, bank_y)
logit_feature = bank_X.columns[selector.support_]

logit_rfe = LogisticRegression(penalty = "none", max_iter = 500, class_weight = {1: weight, 0: 1})
logit_rfe = logit_rfe.fit(bank_X[logit_feature], bank_y)
logit_rfe.score(bank_X[logit_feature], bank_y)
logit_rfe.coef_


# Regularization
logit_l1 = LogisticRegression(penalty = "l2", solver = "saga", max_iter = 500, class_weight = {1: weight, 0: 1})
logit_l1 = logit_l1.fit(bank_X[logit_feature], bank_y)
logit_l1.score(bank_X[logit_feature], bank_y)
logit_l1.coef_

logit_cv = LogisticRegressionCV(penalty = "l2", class_weight = {1: weight, 0: 1})
logit_cv = logit_cv.fit(bank_X[logit_feature], bank_y)
logit_cv.score(bank_X[logit_feature], bank_y)
logit.coef_



# Confusion matrix
models = [logit, logit_cv, logit_sgd]

for model in models:
    pred = model.predict(bank_X[logit_feature])
    cm = confusion_matrix(bank_y, pred, labels = model.classes_)
    print(recall_score(bank_y, pred))
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
    disp.plot()
    plt.title(str(model))
    plt.show()

# Pick logit
logit.coef_
{logit_feature[i]: logit.coef_[0][i] for i in range(len(logit_feature))}

# Random forest
rf = RandomForestClassifier(criterion = "entropy").fit(bank_raw_X, bank_raw_y)
rf.score(bank_raw_X, bank_raw_y)

feature_importance = np.sort(np.stack((bank_raw_X.columns, rf.feature_importances_), axis = 1), axis = 0)[::-1]
np.sum(feature_importance[:, 1])

feature = [i[0] for i in feature_importance]
importance = [i[1] for i in feature_importance]

plt.figure(figsize = (16, 8))
plt.barh(feature[:24], importance[:24], height = 0.75)
plt.title("Top 25 Important Features")
plt.gca().invert_yaxis()
plt.show()
