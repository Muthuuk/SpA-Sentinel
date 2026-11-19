"""SpA-Sentinel orchestrator (paper, Section 4.2 and Figure 2).

The agent receives an event (for example "raised CRP at a GP appointment"), calls the relevant
models, applies the hard safety rules, and writes a plain-English summary for the clinician.
It never prescribes: any change to treatment needs sign-off by the rheumatology team.

Explanations are written from templates by default, so the demo runs with no API key. If the
environment variable ANTHROPIC_API_KEY is set and the `anthropic` package is installed, the
summary can optionally be rewritten by a language model (`explain_with_llm`), which only rewords
the structured findings and cannot change the rule outcomes.
"""
from __future__ import annotations

import os
import textwrap
import numpy as np
import pandas as pd

from . import DISCLAIMER
from .models import M1_FEATURES, M2_FEATURES, prep_m2, m3_risk_12m, m4_explain
from .rules import check_raised_crp, check_drug_options

PLAIN = {"crp_log": "CRP", "bmi": "body mass index", "basfi": "function score (BASFI)", "age": "age",
         "basdai": "disease activity (BASDAI)", "basdai_q2_spinal_pain": "spinal pain score",
         "patient_global": "patient global score", "disease_duration_y": "disease duration", "sex_male": "sex"}


class SpASentinel:
    def __init__(self, trained: dict):
        self.t = trained

    # ---- event: raised CRP ----
    def assess_raised_crp(self, patient: dict, episode: dict, organ: dict | None = None) -> dict:
        ep = dict(episode)
        ep["crp_vs_flare_typical"] = ep["crp"] / patient["crp_flare_typical"]
        ep["crp_vs_remission"] = ep["crp"] / patient["crp_remission"]
        ep["on_biologic"] = int(patient.get("biologic", "none") != "none")
        X = prep_m2(pd.DataFrame([ep]))
        p_inf = float(self.t["m2"].predict_proba(X[M2_FEATURES])[:, 1][0])
        thr = self.t["m2_threshold"]
        rules = check_raised_crp(ep, patient)
        risks = m3_risk_12m(self.t["m3"], organ) if organ else {}
        in_flare_range = patient["crp_remission"] * 1.5 <= ep["crp"] <= patient["crp_flare_typical"] * 1.4
        hi = getattr(self.t["m2"], "high_threshold_", thr)
        if rules.stop:
            verdict = "POSSIBLE SEPSIS - follow sepsis pathway (rules override the model)."
        elif p_inf >= hi:
            verdict = "INFECTION LIKELY - find the source and treat infection if confirmed."
        elif p_inf >= thr:
            verdict = ("UNCERTAIN - " + ("CRP is in this patient's usual flare range, but " if in_flare_range else "")
                       + "infection is not excluded: take cultures before any antibiotic.")
        else:
            verdict = "INFLAMMATORY FLARE MORE LIKELY - still take cultures before any antibiotic."
        return dict(p_infection=p_inf, threshold=thr, verdict=verdict, in_personal_flare_range=in_flare_range,
                    rules=rules, organ_risk_12m=risks, crp=ep["crp"], patient=patient)

    # ---- event: weekly monitoring ----
    def flare_outlook(self, week_row: dict) -> float:
        return float(self.t["m1"].predict_proba(pd.DataFrame([week_row])[M1_FEATURES])[:, 1][0])

    # ---- event: treatment review ----
    def drug_options(self, patient: dict, tnfi_row: dict | None = None) -> dict:
        rules = check_drug_options(patient)
        out = dict(rules=rules, tnfi=None)
        if tnfi_row is not None:
            p, contrib = m4_explain(self.t["m4"], tnfi_row)
            out["tnfi"] = dict(p_response=p, drivers=contrib)
        return out

    # ---- text ----
    @staticmethod
    def summary(rep: dict, drug: dict | None = None) -> str:
        p = rep["patient"]
        lines = [DISCLAIMER, "",
                 f"SpA-Sentinel summary for {p.get('patient_id', 'patient')} (CRP {rep['crp']:.0f} mg/L)",
                 f"Usual CRP: {p['crp_remission']:.0f} mg/L in remission, about {p['crp_flare_typical']:.0f} mg/L in flares.",
                 "This CRP is " + ("within" if rep["in_personal_flare_range"] else "outside") + " this patient's usual flare range.",
                 f"Model M2: infection probability {rep['p_infection']:.2f} (safety threshold {rep['threshold']:.2f}).",
                 f"Overall: {rep['verdict']}", "", "Safety rules:"] + ["  " + l for l in rep["rules"].lines()]
        if rep["organ_risk_12m"]:
            lines += ["", "12-month risk (M3): " + ", ".join(f"{k} {v:.1%}" for k, v in rep["organ_risk_12m"].items())]
        if drug:
            lines += ["", "Drug choice checks:"] + ["  " + l for l in drug["rules"].lines()]
            if drug["tnfi"]:
                d = drug["tnfi"]
                lines.append(f"  M4: probability of TNF-inhibitor response at 12 weeks {d['p_response']:.2f}; main reasons:")
                for f, s, v in d["drivers"]:
                    if f == "crp_log":
                        v = float(np.exp(v))
                    lines.append(f"    {PLAIN.get(f, f)} = {v:.1f} " + ("raises" if s > 0 else "lowers") + " the estimate")
        lines += ["", "Any change to treatment needs sign-off by the rheumatology team."]
        return "\n".join(lines)


def explain_with_llm(summary_text: str, model: str | None = None) -> str:
    """Optional: reword the structured summary in plain language with Claude. Rules are not re-decided."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "(No ANTHROPIC_API_KEY set - template summary shown above is used.)"
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or os.environ.get("SPA_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=500,
        system=("Rewrite this clinical decision-support summary in short, plain English for a GP. "
                "Do not change any number, rule outcome or recommendation, and keep every safety warning."),
        messages=[{"role": "user", "content": summary_text}])
    return "\n".join(textwrap.wrap(msg.content[0].text, 100))
