"""Load the three relational tables and build one policy-level feature matrix."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "train": ROOT / "2026-travelers-umc-training/training",
    "validate": ROOT / "2026-travelers-umc-validation/validation",
    "holdout": ROOT / "2026-travelers-umc-holdout/holdout",
}

CAT_COLS = ["STATE_CD", "PRIOR_CARRIER_GRP", "COV_PKG_TYPE", "PRIMARY_PARKING", "QUOTE_CHANNEL"]


def load_split(split):
    d = PATHS[split]
    if split == "train":
        pol = pd.read_csv(d / "policies_train.csv")
    elif split == "validate":
        pol = pd.read_csv(d / "policies_validate_features.csv").merge(
            pd.read_csv(d / "policies_validate_target.csv"), on="POLICY_ID")
    else:
        pol = pd.read_csv(d / "policies_holdout_features.csv")
    suffix = {"train": "train", "validate": "validate", "holdout": "holdout"}[split]
    drv = pd.read_csv(d / f"drivers_{suffix}.csv")
    veh = pd.read_csv(d / f"vehicles_{suffix}.csv")
    return pol, drv, veh


def driver_features(drv):
    d = drv.copy()
    d["IS_MALE"] = (d.GENDER == "M").astype(float)
    for s in ["own", "rent", "dependent"]:
        d[f"LIV_{s}"] = (d.LIVING_STATUS == s).astype(float)
    for s in ["married", "single", "divorced", "widowed"]:
        d[f"MAR_{s}"] = (d.MARITAL_STATUS == s).astype(float)
    d["YOUNG"] = (d.DRIVER_AGE < 25).astype(float)
    d["SENIOR"] = (d.DRIVER_AGE >= 70).astype(float)
    num = ["DRIVER_AGE", "SAFETY_RATING", "COMMUTE_MILES_DAILY", "ACCIDENT_FREE_YRS"]
    flags = ["IS_MALE", "HIGH_EDU_IND", "TELEMATICS_OPT_IN", "YOUNG", "SENIOR"] + \
        [c for c in d.columns if c.startswith(("LIV_", "MAR_"))]
    g = d.groupby("POLICY_ID")
    out = pd.concat([
        g[num].agg(["mean", "min", "max", "std"]),
        g[flags].agg(["mean", "max"]),
    ], axis=1)
    out.columns = ["DRV_" + "_".join(c) for c in out.columns]
    # primary driver approximated as the first listed row
    first = g[["DRIVER_AGE", "SAFETY_RATING", "LIVING_STATUS", "MARITAL_STATUS"]].first()
    first.columns = ["P1_" + c for c in first.columns]
    out = out.join(first)
    out["DRV_CNT"] = g.size()

    # household structure: rows 0-1 are adults (policyholder/partner), later rows are mostly teens
    d["RK"] = d.groupby("POLICY_ID").cumcount()
    d["TEEN"] = (d.DRIVER_AGE < 21).astype(float)
    adults = d[d.TEEN == 0].groupby("POLICY_ID")
    teens = d[d.TEEN == 1].groupby("POLICY_ID")
    out["N_TEEN"] = g.TEEN.sum()
    out["N_ADULT"] = out.DRV_CNT - out.N_TEEN
    out["ADULT_AGE_MEAN"] = adults.DRIVER_AGE.mean()
    out["ADULT_AGE_MIN"] = adults.DRIVER_AGE.min()
    out["ADULT_AGE_MAX"] = adults.DRIVER_AGE.max()
    out["ADULT_EDU_MEAN"] = adults.HIGH_EDU_IND.mean()
    out["ADULT_EDU_SUM"] = adults.HIGH_EDU_IND.sum()
    out["TEEN_EDU_SUM"] = teens.HIGH_EDU_IND.sum()
    out["ADULT_SAFETY_MEAN"] = adults.SAFETY_RATING.mean()
    out["ADULT_AFY_MEAN"] = adults.ACCIDENT_FREE_YRS.mean()
    out["ADULT_3550"] = adults.DRIVER_AGE.apply(lambda a: a.between(35, 50).mean())
    out[["N_TEEN", "N_ADULT", "TEEN_EDU_SUM", "ADULT_EDU_SUM"]] = out[["N_TEEN", "N_ADULT", "TEEN_EDU_SUM", "ADULT_EDU_SUM"]].fillna(0)
    for r in (0, 1):
        rr = d[d.RK == r].set_index("POLICY_ID")
        for c in ["DRIVER_AGE", "HIGH_EDU_IND", "SAFETY_RATING", "ACCIDENT_FREE_YRS", "TELEMATICS_OPT_IN", "COMMUTE_MILES_DAILY"]:
            out[f"D{r}_{c}"] = rr[c]
        out[f"D{r}_MALE"] = (rr.GENDER == "M").astype(float)
    out["D01_AGE_GAP"] = (out.D0_DRIVER_AGE - out.D1_DRIVER_AGE).abs()
    return out


def vehicle_features(veh):
    v = veh.copy()
    v["MAKE"] = v.MAKE_MODEL.str.split(" : ").str[0]
    num = ["VEHICLE_AGE", "ANNUAL_MILEAGE", "SAFETY_FEATURE_SCORE", "DEPRECIATION_FACTOR"]
    flags = ["ANTI_THEFT_IND", "GARAGE_IND"]
    g = v.groupby("POLICY_ID")
    out = pd.concat([g[num].agg(["mean", "min", "max", "sum"]), g[flags].mean()], axis=1)
    out.columns = ["VEH_" + ("_".join(c) if isinstance(c, tuple) else c) for c in out.columns]
    out["VEH_CNT"] = g.size()
    out["VEH_N_MAKES"] = g.MAKE.nunique()
    prim = v.sort_values("CAR_NO").groupby("POLICY_ID")[["MAKE", "MAKE_MODEL", "COLOR", "VEHICLE_AGE"]].first()
    prim.columns = ["V1_" + c for c in prim.columns]
    return out.join(prim)


def build(split):
    pol, drv, veh = load_split(split)
    df = pol.drop(columns=["SPLIT_FLAG", "CALL_NOTE", "CREDIT_SCORE_SCALED", "MONTHLY_PREMIUM"], errors="ignore")
    df = df.join(driver_features(drv), on="POLICY_ID").join(vehicle_features(veh), on="POLICY_ID")

    dt = pd.to_datetime(df.QUOTE_DT)
    df["Q_YEAR"] = dt.dt.year
    df["Q_MONTH"] = dt.dt.month
    df["Q_DOW"] = dt.dt.dayofweek
    df["Q_T"] = (dt - pd.Timestamp("2015-01-01")).dt.days
    df["DISCOUNT_IND"] = (df.DISCOUNT_IND == "Yes").astype(int)
    df["HOME_POLICY_IND"] = (df.HOME_POLICY_IND == "Y").astype(int)
    df["COV_ORD"] = df.COV_PKG_TYPE.map({"Low": 0, "Medium": 1, "High": 2})
    df["ZIP3"] = (df.ZIP_CD // 100)
    df["PRIOR_CARRIER_MISSING"] = df.PRIOR_CARRIER_GRP.isna().astype(int)

    # pricing ratios — premium relative to exposure
    df["LOG_QUOTE"] = np.log1p(df.QUOTED_AMT)
    df["AMT_PER_VEH"] = df.QUOTED_AMT / df.TOTAL_NUM_VEH
    df["AMT_PER_DRV"] = df.QUOTED_AMT / df.NUM_DRIVERS
    df["AMT_PER_UNIT"] = df.QUOTED_AMT / (df.TOTAL_NUM_VEH + df.NUM_DRIVERS)
    df["VEH_PER_DRV"] = df.TOTAL_NUM_VEH / df.NUM_DRIVERS
    df["LEASE_LOAN_SHARE"] = (df.NUM_LOANED_VEH + df.NUM_LEASED_VEH) / df.TOTAL_NUM_VEH
    df["CREDIT_X_AMT"] = df.CREDIT_SCORE / df.LOG_QUOTE
    df["BUNDLE_DISC"] = df.HOME_POLICY_IND * 2 + df.DISCOUNT_IND

    for c in CAT_COLS + ["P1_LIVING_STATUS", "P1_MARITAL_STATUS", "V1_MAKE", "V1_MAKE_MODEL", "V1_COLOR"]:
        df[c] = df[c].fillna("NA").astype(str)
    return df


def add_relative_price(frames):
    """Premium relative to similar quotes (state x coverage x vehicle count), computed on all
    splits combined — features only, no target, so no leakage."""
    allx = pd.concat(frames, keys=range(len(frames)))
    grp_sets = {
        "REL_AMT_STATE_COV": ["STATE_CD", "COV_PKG_TYPE"],
        "REL_AMT_STATE_COV_VEH": ["STATE_CD", "COV_PKG_TYPE", "TOTAL_NUM_VEH"],
        "REL_AMT_STATE_COV_DRV": ["STATE_CD", "COV_PKG_TYPE", "NUM_DRIVERS"],
        "REL_AMT_ZIP3": ["ZIP3"],
    }
    for name, keys in grp_sets.items():
        med = allx.groupby(keys, dropna=False).LOG_QUOTE.transform("median")
        allx[name] = allx.LOG_QUOTE - med
    allx["ZIP3_CNT"] = allx.groupby("ZIP3").POLICY_ID.transform("size")
    allx["ZIP_CNT"] = allx.groupby("ZIP_CD").POLICY_ID.transform("size")
    return [allx.xs(i).copy() for i in range(len(frames))]


REGIME_FEATS = ["POST17", "EDU_POST", "EDU_PRE", "AMT_PRE", "AMT_POST", "CR_PRE", "CR_POST",
                "DISC_16", "COV_16", "RELAMT_PRE"]


def add_regime(d):
    """Conversion drivers switch over time: price/credit matter in 2015-16, education from 2017,
    discount and coverage package mostly in 2016. Period-specific copies let trees split on each regime."""
    post = (d.Q_YEAR >= 2017).astype(int)
    d["POST17"] = post
    d["EDU_POST"] = d.DRV_HIGH_EDU_IND_mean * post
    d["EDU_PRE"] = d.DRV_HIGH_EDU_IND_mean * (1 - post)
    d["AMT_PRE"] = np.where(post == 0, d.LOG_QUOTE, np.nan)
    d["AMT_POST"] = np.where(post == 1, d.LOG_QUOTE, np.nan)
    d["CR_PRE"] = np.where(post == 0, d.CREDIT_SCORE, np.nan)
    d["CR_POST"] = np.where(post == 1, d.CREDIT_SCORE, np.nan)
    d["DISC_16"] = d.DISCOUNT_IND * (d.Q_YEAR == 2016)
    d["COV_16"] = d.COV_ORD * (d.Q_YEAR == 2016)
    d["RELAMT_PRE"] = np.where(post == 0, d.REL_AMT_STATE_COV, np.nan)
    return d


def load_all():
    tr, va, ho = add_relative_price([build("train"), build("validate"), build("holdout")])
    return add_regime(tr), add_regime(va), add_regime(ho)


DROP = ["POLICY_ID", "QUOTE_DT", "CONVERT_IND"]
