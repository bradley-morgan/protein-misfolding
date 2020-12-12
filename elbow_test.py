from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import os
import subprocess
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
import tools.general_tools as g_tools
import matplotlib.pyplot as plt
import seaborn as sns


data = g_tools.local_data_load('vs14b-feature-scaled-labelled.joblib')
x_train = data.training_data.drop(['target', 'Correlation_Manders_ER_otsu_V2R_otsu', 'Correlation_Manders_V2R_otsu_ER_otsu'], axis=1).to_numpy()
y_train = data.training_data['target'].to_numpy()
feature_names = data.training_data.drop(['target', 'Correlation_Manders_ER_otsu_V2R_otsu', 'Correlation_Manders_V2R_otsu_ER_otsu'], axis=1).columns.to_numpy()

# # Fit and Validate models then generate confusion matrix
K = 100
step = 5
scores = []
i_s = []
for i in range(1, 100+step,  step):
    i = i-1 if i > 1 else i
    model = BalancedRandomForestClassifier(n_estimators=i, max_depth=16, random_state=1)

    # Train model on all the data no cross validation has been done
    model.fit(x_train, y_train)
    y_preds = model.predict(x_train)
    train_score = matthews_corrcoef(y_train, y_preds)
    print(f'MCC Traning Performance = {train_score} mcc')
    scores.append(train_score)
    i_s.append(i)

fig1, ax2 = plt.subplots(1, 1, figsize=(18,8), dpi= 100)
ax2.plot(scores, marker='.')
ax2.set(xlabel='N Estimators', ylabel='MCC')
ax2.set_xticks(range(0, len(i_s)))
ax2.set_xticklabels(i_s)
ax2.grid(b=True)
plt.title('Best Estimators')
plt.show()


# g_tools.local_model_save(model, file_name='BalancedRandomForest')