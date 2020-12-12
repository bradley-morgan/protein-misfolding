from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import tools.general_tools as g_tools



data = g_tools.local_data_load('vs14b-feature-scaled-not-scaled.joblib')
x_train = data.training_data.drop(['target', 'Correlation_Manders_ER_otsu_V2R_otsu', 'Correlation_Manders_V2R_otsu_ER_otsu'], axis=1).to_numpy()
y_train = data.training_data['target'].to_numpy()
feature_names = data.training_data.drop(['target', 'Correlation_Manders_ER_otsu_V2R_otsu', 'Correlation_Manders_V2R_otsu_ER_otsu'], axis=1).columns.to_numpy()

model = BalancedRandomForestClassifier(n_estimators=75, max_depth=16, random_state=1)

# Train model on all the data no cross validation has been done
model.fit(x_train, y_train)
y_preds = model.predict(x_train)
train_score = matthews_corrcoef(y_train, y_preds)
print(f'MCC Traning Performance = {train_score} mcc')

g_tools.local_model_save(model, file_name='BalancedRandomForest')

