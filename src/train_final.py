"""Final pipeline: LightGBM + CatBoost + XGBoost, seed-bagged, blended on out-of-fold predictions.

Stage 1 (honest estimate): fit on train, score validation -> metrics + charts.
Stage 2 (submission):      refit on train+validate, predict holdout.
"""
import sys; sys.path.insert(0, 'src')
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv import get_data, p10

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)

tr, va, ho, _, _ = get_data()
FEATS = pd.read_csv(ROOT / 'src/selected_features.csv').iloc[:, 0].tolist()
CATS = [c for c in FEATS if isinstance(tr[c].dtype, pd.CategoricalDtype)]
LGB_P = dict(json.load(open(ROOT / 'src/lgb_params.json')), subsample_freq=1, max_cat_to_onehot=8, verbose=-1)
XGB_P = dict(n_estimators=800, learning_rate=0.01, max_depth=4, min_child_weight=10, subsample=0.8,
             colsample_bytree=0.5, reg_lambda=10, gamma=0.5, enable_categorical=True,
             max_cat_to_onehot=8, tree_method='hist', eval_metric='auc')
CB_P = dict(iterations=1200, learning_rate=0.03, depth=5, l2_leaf_reg=10, random_strength=1,
            verbose=0, thread_count=-1)


def fit_predict(name, Xa, ya, Xb):
    if name == 'ebm':  # bags internally, so one fit is enough
        Xa_, Xb_ = Xa.copy(), Xb.copy()
        for c in CATS: Xa_[c] = Xa_[c].astype(str); Xb_[c] = Xb_[c].astype(str)
        m = ExplainableBoostingClassifier(interactions=30, outer_bags=8, n_jobs=-1, random_state=0).fit(Xa_, ya)
        return m.predict_proba(Xb_)[:, 1]
    preds = []
    for s in SEEDS:
        if name == 'lgb':
            m = lgb.LGBMClassifier(**LGB_P, random_state=s).fit(Xa, ya)
        elif name == 'xgb':
            m = xgb.XGBClassifier(**XGB_P, random_state=s).fit(Xa, ya)
        else:
            Xa_, Xb_ = Xa.copy(), Xb.copy()
            for c in CATS: Xa_[c] = Xa_[c].astype(str); Xb_[c] = Xb_[c].astype(str)
            m = CatBoostClassifier(**CB_P, random_seed=s, cat_features=CATS).fit(Xa_, ya)
            preds.append(m.predict_proba(Xb_)[:, 1]); continue
        preds.append(m.predict_proba(Xb)[:, 1])
    return np.mean(preds, axis=0)


MODELS = ('lgb', 'cb', 'xgb', 'ebm')


def oof_preds(df):
    y = df.CONVERT_IND.values; oof = {k: np.zeros(len(df)) for k in MODELS}
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=42).split(df, y):
        for k in MODELS:
            oof[k][val] = fit_predict(k, df[FEATS].iloc[trn], y[trn], df[FEATS].iloc[val])
    return oof


def blend_weights(oof, y):
    P = np.column_stack([oof[k] for k in MODELS])
    f = lambda w: -roc_auc_score(y, P @ (np.abs(w) / np.abs(w).sum()))
    w = np.abs(minimize(f, np.ones(len(MODELS)) / len(MODELS), method='Nelder-Mead').x)
    return w / w.sum()


def report_charts(y, p, tag):
    df = pd.DataFrame({'y': y, 'p': p})
    df['decile'] = pd.qcut(df.p.rank(method='first', ascending=False), 10, labels=range(1, 11))
    t = df.groupby('decile', observed=True).agg(n=('y', 'size'), actual=('y', 'mean'), predicted=('p', 'mean'))
    t['lift'] = t.actual / df.y.mean()
    t['cum_capture'] = (t.actual * t.n).cumsum() / df.y.sum()
    t.to_csv(OUT / f'decile_table_{tag}.csv')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(t.index.astype(int), t.lift, color='#2a6fdb')
    ax[0].axhline(1, color='gray', ls='--', lw=1)
    ax[0].set(title='Decile lift (1 = highest score)', xlabel='Score decile', ylabel='Lift vs. average')
    ax[0].set_xticks(range(1, 11))
    for i, v in zip(t.index.astype(int), t.lift): ax[0].text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
    ax[1].plot(t.predicted, t.actual, 'o-', color='#2a6fdb', label='Model')
    lim = [0, max(t.predicted.max(), t.actual.max()) * 1.1]
    ax[1].plot(lim, lim, '--', color='gray', label='Perfect calibration')
    ax[1].set(title='Calibration by score decile', xlabel='Mean predicted conversion', ylabel='Observed conversion',
              xlim=lim, ylim=lim)
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / f'lift_calibration_{tag}.png', dpi=150); plt.close(fig)
    return t


if __name__ == '__main__':
    # ---- Stage 1: blend weights from OOF on train, honest score on validation
    oof = oof_preds(tr)
    for k in MODELS: print(f'OOF(train) {k}: AUC {roc_auc_score(tr.CONVERT_IND, oof[k]):.4f}')
    w = blend_weights(oof, tr.CONVERT_IND.values)
    print('blend weights', dict(zip(MODELS, w.round(3))))

    vp = {k: fit_predict(k, tr[FEATS], tr.CONVERT_IND, va[FEATS]) for k in MODELS}
    yv = va.CONVERT_IND.values
    for k in MODELS: print(f'VALIDATION {k}: AUC {roc_auc_score(yv, vp[k]):.4f}  P@10 {p10(yv, vp[k]):.4f}')
    pv = sum(w[i] * vp[k] for i, k in enumerate(MODELS))
    print(f'VALIDATION BLEND: AUC {roc_auc_score(yv, pv):.4f}  P@10 {p10(yv, pv):.4f}')
    print(report_charts(yv, pv, 'validation').round(3))

    # ---- Stage 2: refit on train+validate, predict holdout
    full = pd.concat([tr, va], ignore_index=True)
    hp = sum(w[i] * fit_predict(k, full[FEATS], full.CONVERT_IND, ho[FEATS]) for i, k in enumerate(MODELS))
    sub = pd.DataFrame({'POLICY_ID': ho.POLICY_ID.values, 'CONVERT_IND_PRED': np.clip(hp, 1e-6, 1 - 1e-6)})
    assert len(sub) == 7375 and sub.POLICY_ID.is_unique
    sub.to_csv(OUT / 'submission.csv', index=False)
    print('wrote', OUT / 'submission.csv', sub.CONVERT_IND_PRED.describe().round(4).to_dict())

    # feature importance for the deck
    m = lgb.LGBMClassifier(**LGB_P, random_state=0).fit(full[FEATS], full.CONVERT_IND)
    pd.Series(m.booster_.feature_importance('gain'), FEATS).sort_values(ascending=False).to_csv(OUT / 'feature_importance.csv')
