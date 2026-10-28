import sys; sys.path.insert(0, 'src')
import json, numpy as np, pandas as pd, optuna
from sklearn.metrics import roc_auc_score
from cv import get_data, cv_lgb
SP = sys.argv[1]
tr, va, ho, feats, cats = get_data()
full = pd.concat([tr, va], ignore_index=True); y = full.CONVERT_IND
fs = pd.read_csv('src/selected_features.csv').iloc[:, 0].tolist()

def obj(t):
    p = dict(n_estimators=t.suggest_int('n_estimators', 200, 2000, log=True),
             learning_rate=t.suggest_float('learning_rate', 0.003, 0.05, log=True),
             num_leaves=t.suggest_int('num_leaves', 4, 48, log=True),
             min_child_samples=t.suggest_int('min_child_samples', 20, 600, log=True),
             subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
             colsample_bytree=t.suggest_float('colsample_bytree', 0.15, 0.9),
             reg_lambda=t.suggest_float('reg_lambda', 0.01, 100, log=True),
             reg_alpha=t.suggest_float('reg_alpha', 0.001, 10, log=True),
             min_split_gain=t.suggest_float('min_split_gain', 0, 1),
             cat_smooth=t.suggest_float('cat_smooth', 1, 200, log=True),
             min_data_per_group=t.suggest_int('min_data_per_group', 20, 500, log=True),
             max_cat_to_onehot=8, verbose=-1)
    return roc_auc_score(y, cv_lgb(full, fs, p, seed=t.number % 3))

st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
st.enqueue_trial(dict(n_estimators=700, learning_rate=0.01, num_leaves=15, min_child_samples=100, subsample=0.8,
                      colsample_bytree=0.5, reg_lambda=5, reg_alpha=0.001, min_split_gain=0, cat_smooth=50, min_data_per_group=200))
st.optimize(obj, n_trials=30, show_progress_bar=False)
print('BEST', st.best_value, st.best_params)
json.dump(st.best_params, open('src/lgb_params.json', 'w'))
st.trials_dataframe().sort_values('value', ascending=False).head(10).to_csv(f'{SP}/lgb_trials.csv')
