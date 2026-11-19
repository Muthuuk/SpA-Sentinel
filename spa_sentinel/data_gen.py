"""Synthetic data generator for the SpA-Sentinel demonstrator.

No real patient data are used. Each table is simulated from a small set of rules chosen so
that the relationships are clinically plausible and match the direction of effects reported
in the literature cited in the paper, for example:
  * CRP is a general marker: both flares and infections raise it (Pepys and Hirschfield, 2003).
  * Many patients have normal CRP during active disease; values fluctuate (Landewe et al., 2018).
  * Higher CRP raises and higher BMI lowers the chance of TNF-inhibitor response (Wang et al., 2022).
  * Repeated antibiotic courses raise IBD risk (Nguyen et al., 2020); calprotectin marks gut risk
    (Cypers et al., 2016).
The effect sizes are illustrative, not estimates from any dataset.

Tables produced (all keyed by patient_id):
  patients.csv          one row per person (baseline characteristics and personal CRP baselines)
  weekly.csv            52 weeks of monitoring per person, label = flare in the next 14 days (M1)
  crp_episodes.csv      episodes of raised CRP, label = infection (1) or inflammatory flare (0) (M2)
  organ_spread.csv      time to new IBD and to kidney involvement over 3 years (M3)
  tnfi_starts.csv       people starting a TNF inhibitor, label = response at 12 weeks (M4)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_patients(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], size=n, p=[0.64, 0.36])
    age = np.clip(rng.normal(42, 12, n), 18, 80).round()
    duration = np.clip(rng.gamma(2.2, 4.5, n), 0.5, 45).round(1)
    hla_b27 = rng.random(n) < np.where(sex == "M", 0.82, 0.72)
    bmi = np.clip(rng.normal(27.5, 4.8, n), 17, 48).round(1)
    ibd = rng.random(n) < 0.068                       # pooled prevalence (Stolwijk et al., 2015)
    uveitis = rng.random(n) < (0.18 + 0.12 * hla_b27)  # about 26% overall
    psoriasis = rng.random(n) < 0.093
    egfr = np.clip(rng.normal(92, 16, n) - 0.35 * (age - 42), 25, 130).round()
    crp_remission = np.exp(rng.normal(np.log(5), 0.45, n)).round(1)   # personal baseline, mg/L
    flare_mult = np.exp(rng.normal(np.log(5.5), 0.35, n))              # flare CRP / remission CRP
    crp_flare = (crp_remission * flare_mult).round(1)
    low_crp_phenotype = rng.random(n) < 0.35          # people whose CRP barely moves in flares
    crp_flare = np.where(low_crp_phenotype, crp_remission * rng.uniform(1.1, 1.8, n), crp_flare).round(1)
    biologic = np.where(ibd, rng.random(n) < 0.6, rng.random(n) < 0.38)
    bio_type = np.where(~biologic, "none",
                        np.where(psoriasis & (rng.random(n) < 0.6), "IL17i",
                                 np.where(rng.random(n) < 0.12, "JAKi", "TNFi")))
    bio_type = np.where(ibd & (bio_type == "IL17i"), "TNFi", bio_type)
    return pd.DataFrame(dict(
        patient_id=[f"P{i:04d}" for i in range(n)], sex=sex, age=age, disease_duration_y=duration,
        hla_b27=hla_b27.astype(int), bmi=bmi, ibd=ibd.astype(int), uveitis_history=uveitis.astype(int),
        psoriasis=psoriasis.astype(int), egfr=egfr, crp_remission=crp_remission, crp_flare_typical=crp_flare,
        low_crp_phenotype=low_crp_phenotype.astype(int), biologic=bio_type,
    ))


def make_weekly(patients: pd.DataFrame, weeks: int = 52, seed: int = 11) -> pd.DataFrame:
    """Weekly monitoring with a hidden disease-activity process; label = flare within 14 days."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in patients.itertuples():
        act = rng.normal(0, 1)
        flare_left = 0
        hist = []
        for w in range(weeks):
            treated = p.biologic != "none"
            act = 0.82 * act + rng.normal(0, 0.55) - (0.08 if treated else 0)
            start_flare = flare_left == 0 and rng.random() < _sigmoid(act * 1.4 - 3.0)
            if start_flare:
                flare_left = int(rng.integers(1, 4))
            in_flare = flare_left > 0
            flare_left = max(0, flare_left - 1)
            crp_mean = p.crp_flare_typical if in_flare else p.crp_remission * np.exp(0.15 * act)
            crp = max(0.3, crp_mean * np.exp(rng.normal(0, 0.3)))
            basdai = float(np.clip(3.0 + 1.1 * act + (2.6 if in_flare else 0) + rng.normal(0, 0.7), 0, 10))
            sleep = float(np.clip(7.0 - 0.35 * act - (0.9 if in_flare else 0) + rng.normal(0, 0.5), 3, 10))
            steps = float(np.clip(7000 - 900 * act - (2200 if in_flare else 0) + rng.normal(0, 900), 500, 20000))
            hist.append(dict(week=w, in_flare=int(in_flare), crp=round(crp, 1), basdai=round(basdai, 1),
                             sleep_h=round(sleep, 1), steps=int(steps), act=act))
        for w in range(2, weeks - 2):
            cur, prev, prev2 = hist[w], hist[w - 1], hist[w - 2]
            weeks_since_flare = next((w - k for k in range(w, -1, -1) if hist[k]["in_flare"]), 99)
            label = int(hist[w + 1]["in_flare"] or hist[w + 2]["in_flare"])
            if cur["in_flare"]:
                continue  # forecast only from weeks that are not already in a flare
            rows.append(dict(
                patient_id=p.patient_id, week=w,
                crp=cur["crp"], crp_ratio_baseline=round(cur["crp"] / p.crp_remission, 2),
                crp_slope=round(cur["crp"] - prev2["crp"], 2), basdai=cur["basdai"],
                basdai_slope=round(cur["basdai"] - prev2["basdai"], 2), sleep_h=cur["sleep_h"],
                steps=cur["steps"], weeks_since_flare=min(weeks_since_flare, 52),
                on_biologic=int(p.biologic != "none"), low_crp_phenotype=p.low_crp_phenotype,
                flare_next_14d=label))
    return pd.DataFrame(rows)


def make_crp_episodes(patients: pd.DataFrame, n: int = 6000, seed: int = 13) -> pd.DataFrame:
    """Episodes where CRP is raised; true cause is infection (1) or inflammatory flare (0)."""
    rng = np.random.default_rng(seed)
    pts = patients.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    on_bio = (pts.biologic != "none").to_numpy()
    p_inf = np.where(on_bio, 0.33, 0.24)
    infection = rng.random(n) < p_inf
    sites = ["urinary", "respiratory", "skin", "gut", "back", "joint"]
    # Overlap on purpose: infections can present as back or joint pain (e.g. discitis, septic joint),
    # and flares can coincide with chest or skin symptoms (e.g. a cold, a psoriasis flare).
    site = np.where(infection, rng.choice(sites, n, p=[0.38, 0.30, 0.10, 0.09, 0.08, 0.05]),
                    rng.choice(sites, n, p=[0.14, 0.06, 0.05, 0.12, 0.52, 0.11]))
    flare_crp = pts.crp_flare_typical.to_numpy() * np.exp(rng.normal(0, 0.5, n))
    inf_crp = np.maximum(pts.crp_remission.to_numpy() * np.exp(rng.normal(np.log(5), 1.0, n)), 6)
    crp = np.where(infection, inf_crp, np.maximum(flare_crp, pts.crp_remission.to_numpy() * 1.3))
    temp = np.where(infection, rng.normal(37.5, 0.8, n), rng.normal(37.15, 0.5, n))
    hr = np.where(infection, rng.normal(89, 14, n), rng.normal(81, 11, n))
    wbc = np.where(infection, rng.normal(10.6, 3.3, n), rng.normal(8.9, 2.3, n))
    urinary = site == "urinary"
    nitrites = np.where(infection & urinary, rng.random(n) < 0.45, rng.random(n) < 0.05)
    leuk = np.where(infection & urinary, rng.random(n) < 0.7, rng.random(n) < 0.2)
    haematuria = np.where(infection & urinary, rng.random(n) < 0.5, rng.random(n) < 0.06)
    basdai_rise = np.where(infection, rng.normal(0.9, 1.3, n), rng.normal(1.8, 1.3, n))
    back_pain_pattern = np.where(infection, rng.random(n) < 0.35, rng.random(n) < 0.7)
    return pd.DataFrame(dict(
        patient_id=pts.patient_id, crp=crp.round(1),
        crp_vs_flare_typical=(crp / pts.crp_flare_typical.to_numpy()).round(2),
        crp_vs_remission=(crp / pts.crp_remission.to_numpy()).round(2),
        temp_c=temp.round(1), heart_rate=hr.round(), wbc=wbc.round(1),
        urine_nitrites=nitrites.astype(int), urine_leukocytes=leuk.astype(int), urine_blood=haematuria.astype(int),
        basdai_rise=basdai_rise.round(1), inflammatory_back_pain=back_pain_pattern.astype(int),
        site=site, on_biologic=on_bio.astype(int), infection=infection.astype(int)))


def make_organ_spread(patients: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    """Time to new IBD and to kidney involvement over 3 years (exponential hazards, censored)."""
    rng = np.random.default_rng(seed)
    n = len(patients)
    p = patients
    calpro = np.exp(rng.normal(np.log(60), 0.8, n)).round()          # faecal calprotectin, ug/g
    gut_sx = (rng.random(n) < 0.18 + 0.002 * np.minimum(calpro, 400)).astype(int)
    abx = rng.poisson(0.9, n) + (rng.random(n) < 0.25) * rng.poisson(2, n)  # courses in last 2 years
    nsaid_years = np.clip(rng.gamma(2, 2, n), 0, 30).round(1)
    urine_pcr = np.exp(rng.normal(np.log(12), 0.7, n)).round(1)       # protein:creatinine, mg/mmol
    il17 = (p.biologic == "IL17i").astype(int).to_numpy()
    lp_ibd = (-4.2 + 0.9 * np.log(calpro / 50) + 0.6 * gut_sx + 0.17 * abx + 0.35 * il17
              - 0.3 * p.hla_b27.to_numpy())
    lp_kid = (-5.0 + 0.9 * np.log(urine_pcr / 15) + 0.05 * nsaid_years - 0.025 * (p.egfr.to_numpy() - 90)
              + 0.4 * (rng.random(n) < 0.05))
    t_ibd = rng.exponential(1 / np.exp(lp_ibd))
    t_kid = rng.exponential(1 / np.exp(lp_kid))
    horizon = 3.0
    ibd_at_start = p.ibd.to_numpy() == 1
    df = pd.DataFrame(dict(
        patient_id=p.patient_id, faecal_calprotectin=calpro, gut_symptoms=gut_sx, antibiotic_courses_2y=abx,
        hla_b27=p.hla_b27, on_il17i=il17, nsaid_years=nsaid_years, urine_pcr=urine_pcr, egfr=p.egfr,
        ibd_time_y=np.minimum(t_ibd, horizon).round(2), ibd_event=(t_ibd <= horizon).astype(int),
        kidney_time_y=np.minimum(t_kid, horizon).round(2), kidney_event=(t_kid <= horizon).astype(int)))
    df.loc[ibd_at_start, ["ibd_time_y", "ibd_event"]] = np.nan   # already have IBD: not at risk of new IBD
    return df


def make_tnfi_starts(n: int = 1500, seed: int = 19) -> pd.DataFrame:
    """People starting a TNF inhibitor; response direction follows Wang et al. (2022)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(39, 11, n), 18, 75).round()
    sex = rng.choice(["M", "F"], n, p=[0.75, 0.25])
    bmi = np.clip(rng.normal(27, 5, n), 17, 48).round(1)
    crp = np.exp(rng.normal(np.log(12), 0.9, n)).round(1)
    basdai = np.clip(rng.normal(6.3, 1.4, n), 2, 10).round(1)
    basdai_q2 = np.clip(basdai + rng.normal(0.3, 1.2, n), 0, 10).round(1)
    basfi = np.clip(rng.normal(5.4, 2.0, n), 0, 10).round(1)
    duration = np.clip(rng.gamma(2, 5, n), 0.5, 40).round(1)
    pga = np.clip(basdai + rng.normal(0, 1.2, n), 0, 10).round(1)
    lp = (-0.4 + 0.55 * np.log(crp / 10) + 0.12 * (basdai_q2 - 6) + 0.10 * (pga - 6) - 0.06 * (bmi - 27)
          - 0.18 * (basfi - 5) - 0.02 * (age - 40) - 0.015 * (duration - 10) + rng.normal(0, 0.6, n))
    response = (rng.random(n) < _sigmoid(lp)).astype(int)
    return pd.DataFrame(dict(age=age, sex_male=(sex == "M").astype(int), bmi=bmi, crp=crp, basdai=basdai,
                             basdai_q2_spinal_pain=basdai_q2, basfi=basfi, disease_duration_y=duration,
                             patient_global=pga, response_12w=response))


def generate_all(out_dir: str = "data", seed: int = 7) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    patients = make_patients(seed=seed)
    tables = dict(patients=patients, weekly=make_weekly(patients), crp_episodes=make_crp_episodes(patients),
                  organ_spread=make_organ_spread(patients), tnfi_starts=make_tnfi_starts())
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    return tables


if __name__ == "__main__":
    t = generate_all()
    for k, v in t.items():
        print(f"{k:14s} {v.shape}")
