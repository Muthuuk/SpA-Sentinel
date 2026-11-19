"""The four prediction models of SpA-Sentinel (paper, Table 4).

M1  Flare forecaster          - gradient-boosted trees (XGBoost) on weekly monitoring features
M2  Flare-or-infection scorer - calibrated logistic regression, threshold tuned for infection sensitivity
M3  Organ-spread risk         - Cox proportional hazards models (lifelines) for new IBD and kidney involvement
M4  Drug response             - XGBoost with SHAP explanations for 12-week TNF-inhibitor response

Every model is evaluated on a held-out split, grouped by patient where people appear more than once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

M1_FEATURES = ["crp", "crp_ratio_baseline", "crp_slope", "basdai", "basdai_slope", "sleep_h", "steps",
               "weeks_since_flare", "on_biologic", "low_crp_phenotype"]
M2_FEATURES = ["crp_vs_flare_typical", "crp_vs_remission", "temp_c", "heart_rate", "wbc", "urine_nitrites",
               "urine_leukocytes", "basdai_rise", "inflammatory_back_pain", "on_biologic",
               "site_urinary", "site_respiratory", "site_skin", "site_gut"]
M3_IBD = ["faecal_calprotectin_log", "gut_symptoms", "antibiotic_courses_2y", "on_il17i", "hla_b27"]
M3_KID = ["urine_pcr_log", "nsaid_years", "egfr"]
M4_FEATURES = ["age", "sex_male", "bmi", "crp_log", "basdai", "basdai_q2_spinal_pain", "basfi",
               "disease_duration_y", "patient_global"]


def _split_by_patient(df, test_size=0.25, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["patient_id"]))
    return df.iloc[tr], df.iloc[te]


def calibration_table(y, p, bins=5):
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"mean_predicted": pd.Series(p).groupby(q, observed=True).mean().values,
                         "observed_rate": pd.Series(y).groupby(q, observed=True).mean().values})


# ---------------- M1 ----------------
def train_m1(weekly: pd.DataFrame, seed=0):
    tr, te = _split_by_patient(weekly, seed=seed)
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="logloss", random_state=seed)
    model.fit(tr[M1_FEATURES], tr["flare_next_14d"])
    p = model.predict_proba(te[M1_FEATURES])[:, 1]
    metrics = dict(auroc=roc_auc_score(te["flare_next_14d"], p), brier=brier_score_loss(te["flare_next_14d"], p),
                   prevalence=te["flare_next_14d"].mean(), n_test=len(te))
    return model, metrics, (te["flare_next_14d"].values, p)


# ---------------- M2 ----------------
def prep_m2(ep: pd.DataFrame) -> pd.DataFrame:
    X = ep.copy()
    for s in ["urinary", "respiratory", "skin", "gut"]:
        X[f"site_{s}"] = (X["site"] == s).astype(int)
    return X


def _threshold_for(p_val, y_val, target):
    inf_p = np.sort(p_val[y_val == 1])
    return float(inf_p[int(np.floor((1 - target) * len(inf_p)))]) if len(inf_p) else 0.5


def train_m2(episodes: pd.DataFrame, target_sensitivity=0.98, seed=0, high_sensitivity=0.90):
    """Returns model, safety threshold, metrics, test predictions.

    Two thresholds are set on a validation split:
      * the safety threshold catches `target_sensitivity` of infections (default 98%);
      * the 'infection likely' threshold catches `high_sensitivity` of them (default 90%).
    Between the two, the agent reports 'uncertain - cultures first'.
    """
    X = prep_m2(episodes)
    tr, te = _split_by_patient(X, seed=seed)
    tr, va = _split_by_patient(tr, test_size=0.25, seed=seed + 1)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5).fit(tr[M2_FEATURES], tr["infection"])
    pv = model.predict_proba(va[M2_FEATURES])[:, 1]
    threshold = _threshold_for(pv, va["infection"].values, target_sensitivity)
    model.high_threshold_ = _threshold_for(pv, va["infection"].values, high_sensitivity)
    p = model.predict_proba(te[M2_FEATURES])[:, 1]
    flagged = (p >= threshold).astype(int)
    metrics = dict(auroc=roc_auc_score(te["infection"], p), threshold=threshold, high_threshold=model.high_threshold_,
                   sensitivity_infection=recall_score(te["infection"], flagged),
                   specificity=recall_score(1 - te["infection"], 1 - flagged), n_test=len(te))
    return model, threshold, metrics, (te["infection"].values, p)


# ---------------- M3 ----------------
def prep_m3(org: pd.DataFrame) -> pd.DataFrame:
    X = org.copy()
    X["faecal_calprotectin_log"] = np.log(X["faecal_calprotectin"])
    X["urine_pcr_log"] = np.log(X["urine_pcr"])
    return X


def train_m3(organ: pd.DataFrame, seed=0):
    from lifelines import CoxPHFitter
    X = prep_m3(organ)
    out = {}
    for name, feats, tcol, ecol in [("ibd", M3_IBD, "ibd_time_y", "ibd_event"),
                                    ("kidney", M3_KID, "kidney_time_y", "kidney_event")]:
        d = X[feats + [tcol, ecol]].dropna()
        tr, te = train_test_split(d, test_size=0.25, random_state=seed)
        cph = CoxPHFitter(penalizer=0.01).fit(tr, duration_col=tcol, event_col=ecol)
        c_index = cph.score(te, scoring_method="concordance_index")
        out[name] = dict(model=cph, features=feats, c_index=c_index, events=int(d[ecol].sum()), n=len(d))
    return out


def m3_risk_12m(m3: dict, patient_row: dict) -> dict:
    X = prep_m3(pd.DataFrame([patient_row]))
    risks = {}
    for name, info in m3.items():
        sf = info["model"].predict_survival_function(X[info["features"]], times=[1.0])
        risks[name] = float(1 - sf.values[0, 0])
    return risks


# ---------------- M4 ----------------
def prep_m4(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["crp_log"] = np.log(X["crp"])
    return X


def train_m4(tnfi: pd.DataFrame, seed=0):
    X = prep_m4(tnfi)
    tr, te = train_test_split(X, test_size=0.25, random_state=seed, stratify=X["response_12w"])
    model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9,
                          eval_metric="logloss", random_state=seed)
    model.fit(tr[M4_FEATURES], tr["response_12w"])
    p = model.predict_proba(te[M4_FEATURES])[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[M4_FEATURES], tr["response_12w"])
    metrics = dict(auroc_xgb=roc_auc_score(te["response_12w"], p),
                   auroc_logistic=roc_auc_score(te["response_12w"], lr.predict_proba(te[M4_FEATURES])[:, 1]),
                   response_rate=te["response_12w"].mean(), n_test=len(te))
    return model, metrics, te


def m4_explain(model, row: dict, top=4):
    """Return probability and the top SHAP contributions (feature, contribution in log-odds)."""
    import shap
    X = prep_m4(pd.DataFrame([row]))[M4_FEATURES]
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)[0]
    p = float(model.predict_proba(X)[0, 1])
    contrib = sorted(zip(M4_FEATURES, sv, X.iloc[0].values), key=lambda t: -abs(t[1]))[:top]
    return p, contrib


def train_all(tables: dict, seed=0):
    m1, m1m, m1_eval = train_m1(tables["weekly"], seed)
    m2, thr, m2m, m2_eval = train_m2(tables["crp_episodes"], seed=seed)
    m3 = train_m3(tables["organ_spread"], seed)
    m4, m4m, m4_test = train_m4(tables["tnfi_starts"], seed)
    return dict(m1=m1, m1_metrics=m1m, m1_eval=m1_eval, m2=m2, m2_threshold=thr, m2_metrics=m2m, m2_eval=m2_eval,
                m3=m3, m4=m4, m4_metrics=m4m, m4_test=m4_test)
