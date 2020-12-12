from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
from scipy.stats import sem
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
import tools.general_tools as g_tools


data = g_tools.local_data_load('vs14b-feature-scaled-labelled.joblib')
x_train = data.training_data.drop(['target', 'Correlation_Manders_ER_otsu_V2R_otsu', 'Correlation_Manders_V2R_otsu_ER_otsu'], axis=1).to_numpy()
y_train = data.training_data['target'].to_numpy()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

# # Fit and Validate models then generate confusion matrix
model = DecisionTreeClassifier()

score_func = make_scorer(matthews_corrcoef)
scores = cross_val_score(model, X=x_train, y=y_train,
                         scoring=score_func, cv=cv, n_jobs=-1)


mean_s = mean(scores)
std_s = std(scores)
ste_s = sem(scores)

print('MCC: Mean=%.3f Standard Deviation=%.3f Standard Error=%.3f' % (mean_s, std_s, ste_s))
metrics = {'Mean': mean_s, 'Standard Deviation': std_s, 'Standard Error': ste_s}

# Train model on all the data no cross validation has been done
model.fit(x_train, y_train)
y_preds = model.predict(x_train)
train_score = matthews_corrcoef(y_train, y_preds)
print(f'MCC Traning Performance = {train_score} mcc')

g_tools.local_model_save(model, file_name='BalancedRandomForest')

# save_dir = g_tools.path('./tmp')
# name = 'tree_graph'
# format = 'dot'
#
# dot_out_file = os.path.join(save_dir, f'{name}.{format}')
# tree.export_graphviz(
#     model,
#     out_file=dot_out_file,
#     feature_names=feature_names,
#     class_names=['neg', 'pos'],
#     filled=True,
#     rounded=True,
# )
# # Convert to png
# format = 'png'
# png_out_file = os.path.join(save_dir, f'{name}.{format}')
# out = subprocess.run(['dot', '-Tpng', dot_out_file, '-o', png_out_file])
