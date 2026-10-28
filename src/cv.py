"""5-fold stratified CV on train+validate combined (more stable than one 7k validation split),
plus the official train->validate score."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from features import load_all, DROP

def p10(y, p):
    y = np.asarray(y); k = int(len(p) * 0.1); return y[np.argsort(-p)[:k]].mean()

def get_data(exclude=()):
    tr, va, ho = load_all()
    feats = [c for c in tr.columns if c not in DROP and c not in exclude]
    cats = [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]
    for c in cats:
        levels = sorted(set(tr[c]) | set(va[c]) | set(ho[c]))
        for df in (tr, va, ho): df[c] = pd.Categorical(df[c], categories=levels)
    return tr, va, ho, feats, cats

LGB_PARAMS = dict(n_estimators=700, learning_rate=0.01, num_leaves=15, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                  cat_smooth=50, min_data_per_group=200, max_cat_to_onehot=8, verbose=-1)

def cv_lgb(full, feats, params=LGB_PARAMS, seed=0):
    oof = np.zeros(len(full)); y = full.CONVERT_IND.values
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(full, y):
        m = lgb.LGBMClassifier(**params, random_state=seed).fit(full[feats].iloc[trn], y[trn])
        oof[val] = m.predict_proba(full[feats].iloc[val])[:, 1]
    return oof

if __name__ == "__main__":
    tr, va, ho, feats, cats = get_data()
    full = pd.concat([tr, va], ignore_index=True)
    oof = cv_lgb(full, feats)
    print(f"CV AUC {roc_auc_score(full.CONVERT_IND, oof):.4f}  P@10 {p10(full.CONVERT_IND, oof):.4f}")
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(tr[feats], tr.CONVERT_IND)
    pv = m.predict_proba(va[feats])[:, 1]
    print(f"VAL AUC {roc_auc_score(va.CONVERT_IND, pv):.4f}  P@10 {p10(va.CONVERT_IND, pv):.4f}")
    print(pd.Series(m.booster_.feature_importance('gain'), feats).sort_values(ascending=False).head(25).round(0))
