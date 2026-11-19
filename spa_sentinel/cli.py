"""Command-line tool for SpA-Sentinel.

Examples
  python -m spa_sentinel.cli demo                      # Patient A from the paper, then the sepsis case
  python -m spa_sentinel.cli assess --crp 36 --remission 6 --flare 32 --temp 37.1 --hr 82 --wbc 8.6 \
         --site urinary --urine-blood --biologic TNFi
  python -m spa_sentinel.cli metrics                   # train and print test metrics
  python -m spa_sentinel.cli roi --trusts 3            # ROI model with changed assumptions
  python -m spa_sentinel.cli card --remission 6 --flare 32 --biologic TNFi --out card.png

Trained models are cached in models/trained.pkl (delete the file to retrain).
RESEARCH DEMONSTRATOR ONLY - synthetic data - not a medical device.
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "models", "trained.pkl")
NAMES = ["patients", "weekly", "crp_episodes", "organ_spread", "tnfi_starts"]

PATIENT_A = dict(patient_id="Patient A", age=41, biologic="TNFi", crp_remission=6, crp_flare_typical=32,
                 ibd=0, uveitis_history=0, egfr=88, tb_screened=True)
EPISODE_A = dict(crp=36, temp_c=37.1, heart_rate=82, wbc=8.6, resp_rate=16, systolic_bp=124, site="urinary",
                 urine_nitrites=0, urine_leukocytes=0, urine_blood=1, basdai_rise=1.6, inflammatory_back_pain=1)
ORGAN_A = dict(faecal_calprotectin=70, gut_symptoms=0, antibiotic_courses_2y=3, on_il17i=0, hla_b27=1,
               urine_pcr=34, nsaid_years=8, egfr=88)
TNFI_A = dict(age=41, sex_male=1, bmi=29, crp=36, basdai=5.8, basdai_q2_spinal_pain=6.5, basfi=4.2,
              disease_duration_y=9, patient_global=6.0)


def load_tables():
    from .data_gen import generate_all
    if not all(os.path.exists(os.path.join(DATA, f"{n}.csv")) for n in NAMES):
        return generate_all(DATA)
    return {n: pd.read_csv(os.path.join(DATA, f"{n}.csv")) for n in NAMES}


def load_models(retrain=False):
    if os.path.exists(CACHE) and not retrain:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from .models import train_all
    trained = train_all(load_tables())
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(trained, f)
    return trained


def metrics_table(T) -> pd.DataFrame:
    return pd.DataFrame([
        ["M1 flare in next 14 days", "AUROC", T["m1_metrics"]["auroc"]],
        ["M2 infection vs flare", "AUROC", T["m2_metrics"]["auroc"]],
        ["M2 at 98% safety threshold", "sensitivity (infection)", T["m2_metrics"]["sensitivity_infection"]],
        ["M2 at 98% safety threshold", "specificity", T["m2_metrics"]["specificity"]],
        ["M3 new IBD (3 years)", "C-index", T["m3"]["ibd"]["c_index"]],
        ["M3 kidney involvement (3 years)", "C-index", T["m3"]["kidney"]["c_index"]],
        ["M4 TNFi response (XGBoost)", "AUROC", T["m4_metrics"]["auroc_xgb"]],
    ], columns=["model", "metric", "value"]).round(3)


def cmd_demo(_):
    from .agent import SpASentinel
    agent = SpASentinel(load_models())
    rep = agent.assess_raised_crp(PATIENT_A, EPISODE_A, ORGAN_A)
    print(agent.summary(rep, agent.drug_options(PATIENT_A, TNFI_A)))
    print("\n" + "=" * 70 + "\nSame patient with signs of sepsis (temp 38.9 C, heart rate 118, WBC 15.8):\n")
    sep = dict(EPISODE_A, temp_c=38.9, heart_rate=118, wbc=15.8)
    print(agent.summary(agent.assess_raised_crp(PATIENT_A, sep)))


def cmd_assess(a):
    from .agent import SpASentinel
    agent = SpASentinel(load_models())
    patient = dict(patient_id=a.id, biologic=a.biologic, crp_remission=a.remission, crp_flare_typical=a.flare,
                   ibd=int(a.ibd), egfr=a.egfr)
    ep = dict(crp=a.crp, temp_c=a.temp, heart_rate=a.hr, wbc=a.wbc, site=a.site, urine_nitrites=int(a.urine_nitrites),
              urine_leukocytes=int(a.urine_leukocytes), urine_blood=int(a.urine_blood), basdai_rise=a.basdai_rise,
              inflammatory_back_pain=int(a.back_pain), systolic_bp=a.sbp)
    print(agent.summary(agent.assess_raised_crp(patient, ep), agent.drug_options(patient)))


def cmd_metrics(a):
    print(metrics_table(load_models(retrain=a.retrain)).to_string(index=False))


def cmd_roi(a):
    from .roi import run, Assumptions
    r = run(Assumptions(patients=a.patients, trusts_sharing=a.trusts, admit_avoided=a.admit_avoided))
    for k, v in r["items"].items():
        print(f"  {k:38s} £{v:>10,.0f}")
    print(f"  {'annual NHS benefit':38s} £{r['annual_benefit']:>10,.0f}")
    print("  cumulative net by year: " + ", ".join(f"Y{i+1} £{v:,.0f}" for i, v in enumerate(r["cumulative_net"])))
    print(f"  5-year ROI {r['roi']:.0%}; break-even year {r['breakeven_year']}")


def cmd_card(a):
    from .card import make_card
    print(make_card(dict(crp_remission=a.remission, crp_flare_typical=a.flare, biologic=a.biologic), a.contact, a.out))


def main(argv=None):
    from . import DISCLAIMER
    p = argparse.ArgumentParser(prog="spa-sentinel", description=DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(f=cmd_demo)
    s = sub.add_parser("assess")
    s.add_argument("--id", default="patient"); s.add_argument("--crp", type=float, required=True)
    s.add_argument("--remission", type=float, required=True); s.add_argument("--flare", type=float, required=True)
    s.add_argument("--temp", type=float, default=37.0); s.add_argument("--hr", type=float, default=80)
    s.add_argument("--wbc", type=float, default=8.0); s.add_argument("--sbp", type=float, default=120)
    s.add_argument("--site", default="back", choices=["back", "joint", "urinary", "respiratory", "skin", "gut"])
    s.add_argument("--urine-nitrites", action="store_true"); s.add_argument("--urine-leukocytes", action="store_true")
    s.add_argument("--urine-blood", action="store_true"); s.add_argument("--back-pain", action="store_true")
    s.add_argument("--basdai-rise", type=float, default=1.0); s.add_argument("--biologic", default="none")
    s.add_argument("--ibd", action="store_true"); s.add_argument("--egfr", type=float, default=90)
    s.set_defaults(f=cmd_assess)
    m = sub.add_parser("metrics"); m.add_argument("--retrain", action="store_true"); m.set_defaults(f=cmd_metrics)
    r = sub.add_parser("roi"); r.add_argument("--patients", type=int, default=1200); r.add_argument("--trusts", type=int, default=1)
    r.add_argument("--admit-avoided", type=float, default=0.20); r.set_defaults(f=cmd_roi)
    c = sub.add_parser("card"); c.add_argument("--remission", type=float, required=True); c.add_argument("--flare", type=float, required=True)
    c.add_argument("--biologic", default="none"); c.add_argument("--contact", default="advice line (demo)")
    c.add_argument("--out", default="alert_card.png"); c.set_defaults(f=cmd_card)
    args = p.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
