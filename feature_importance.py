import tools.general_tools as g_tools
import tools.model_tools as m_tools
import wandb

run = wandb.init(
    project='protein-misfolding',
    name='Feature Importance'
)

model = g_tools.local_model_load('v2_BalancedRandomForest.joblib')
data = g_tools.local_data_load('vs14b-feature-scaled-not-scaled.joblib')

x_train = data.training_data.drop('target', axis=1).to_numpy()
y_train = data.training_data['target'].to_numpy()
feature_names = data.training_data.drop('target', axis=1).columns.to_numpy()

m_tools.shap_feature_importance(model, x_train, y_train, feature_names, run=run)