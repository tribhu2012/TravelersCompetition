import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from features import load_all, DROP

tr, va, ho = load_all()
feats = [c for c in tr.columns if c not in DROP]
cats = [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]
def prep(df):
    X = df[feats].copy()
    for c in cats: X[c] = pd.Categorical(X[c], categories=sorted(set(tr[c]) | set(va[c]) | set(ho[c])))
    return X
Xtr, Xva = prep(tr), prep(va); ytr, yva = tr.CONVERT_IND, va.CONVERT_IND

def p10(y, p):
    k = int(len(p) * 0.1); return y.values[np.argsort(-p)[:k]].mean()

m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.01, num_leaves=15, min_child_samples=100,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                       cat_smooth=50, min_data_per_group=200, max_cat_to_onehot=8, verbose=-1)
m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc', callbacks=[lgb.early_stopping(300, verbose=False)])
pl = m.predict_proba(Xva)[:, 1]
print('LGB', m.best_iteration_, roc_auc_score(yva, pl), p10(yva, pl))
imp = pd.Series(m.booster_.feature_importance('gain'), feats).sort_values(ascending=False)
print(imp.head(30).round(0))

cb = CatBoostClassifier(iterations=5000, learning_rate=0.03, depth=6, l2_leaf_reg=5, eval_metric='AUC',
                        od_type='Iter', od_wait=400, verbose=0, cat_features=cats, thread_count=-1)
cb.fit(tr[feats], ytr, eval_set=(va[feats], yva))
pc = cb.predict_proba(va[feats])[:, 1]
print('CB', cb.best_iteration_, roc_auc_score(yva, pc), p10(yva, pc))
print(pd.Series(cb.get_feature_importance(), feats).sort_values(ascending=False).head(20).round(2))
