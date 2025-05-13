'''Machine Learning Models in comparison'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    y_bin = np.where(y_true == y_true[0], 0, 1)
    p_bin = np.where(y_pred == y_true[0], 0, 1)
    auc = roc_auc_score(y_bin, p_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    gmean = np.sqrt(sens * spec)
    return auc, gmean


data = pd.read_csv('Experiment1/7Ydata.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False),
    "MLP": MLPClassifier()
}

from collections import defaultdict

N_RUNS = 5
aggregated_results = {name: defaultdict(list) for name in models}

for run in range(N_RUNS):
    print(f"\n {run + 1} Iteration：" + "=" * 30)

    random_seed = run
    np.random.seed(random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

    for name, model in models.items():
        if hasattr(model, 'random_state'):
            model.set_params(random_state=random_seed)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        auc, gmean = calculate_metrics(y_test, y_pred)
        aggregated_results[name]['gmean'].append(gmean)
        aggregated_results[name]['auc'].append(auc)

metrics = ['gmean', 'auc']

results_data = []

for model_name, metric_dict in aggregated_results.items():
    model_results = {'Model': model_name}
    for metric in metrics:
        model_results[f'{metric}_mean'] = np.mean(metric_dict[metric])
        model_results[f'{metric}_std'] = np.std(metric_dict[metric])

    results_data.append(model_results)

results_df = pd.DataFrame(results_data)

column_order = ['Model']
for metric in metrics:
    column_order.extend([f'{metric}_mean', f'{metric}_std'])
results_df = results_df[column_order]

print("\n Results (mean ± std):\n")
print(results_df)

output_file = '7YModel_Model_test.xlsx'
with pd.ExcelWriter(output_file) as writer:
    results_df.to_excel(writer, sheet_name='Statistics', index=False)

    raw_data = []
    for model_name, metric_dict in aggregated_results.items():
        for metric in metrics:
            for run, value in enumerate(metric_dict[metric]):
                raw_data.append({
                    'Model': model_name,
                    'Run': run + 1,
                    'Metric': metric,
                    'Value': value
                })

    pd.DataFrame(raw_data).to_excel(writer, sheet_name='Raw_Data', index=False)
