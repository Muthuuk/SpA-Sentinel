"""Hard safety rules for SpA-Sentinel.

These rules are deliberately written as plain, deterministic code, separate from any machine
learning model or language model, so that they are predictable and auditable and cannot be
overridden by a model's output (paper, Section 4.2). Thresholds are illustrative and would be
set and signed off by the deploying organisation's clinical safety officer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuleResult:
    level: str          # "STOP" (overrides everything), "WARN", or "INFO"
    code: str
    message: str


@dataclass
class RuleReport:
    results: list = field(default_factory=list)

    def add(self, level, code, message):
        self.results.append(RuleResult(level, code, message))

    @property
    def stop(self) -> bool:
        return any(r.level == "STOP" for r in self.results)

    def lines(self):
        order = {"STOP": 0, "WARN": 1, "INFO": 2}
        return [f"[{r.level}] {r.message}" for r in sorted(self.results, key=lambda r: order[r.level])]


def sepsis_screen(episode: dict) -> int:
    """Count simple warning signs (SIRS-style, illustrative). Two or more = possible sepsis."""
    t, hr, wbc = episode.get("temp_c"), episode.get("heart_rate"), episode.get("wbc")
    rr = episode.get("resp_rate")
    signs = 0
    signs += t is not None and (t >= 38.0 or t < 36.0)
    signs += hr is not None and hr > 90
    signs += rr is not None and rr > 20
    signs += wbc is not None and (wbc > 12 or wbc < 4)
    return int(signs)


def check_raised_crp(episode: dict, patient: dict) -> RuleReport:
    """Rules applied whenever a raised CRP is assessed."""
    rep = RuleReport()
    on_bio = patient.get("biologic", "none") != "none"
    if sepsis_screen(episode) >= 2 or episode.get("systolic_bp", 120) < 100:
        rep.add("STOP", "SEPSIS", "Possible sepsis: follow the local sepsis pathway now. Do not delay antibiotics.")
    if episode.get("urine_nitrites") and episode.get("site") == "urinary":
        rep.add("WARN", "UTI_SIGNS", "Positive urine nitrites with urinary symptoms: urinary infection is likely; "
                                     "send urine culture and treat as infection unless the clinician judges otherwise.")
    if on_bio:
        rep.add("WARN", "BIOLOGIC", f"Patient takes a biologic or targeted drug ({patient.get('biologic')}): "
                                    "higher infection risk; take cultures today and review the same day.")
    if episode.get("urine_blood") or episode.get("urine_pcr", 0) >= 30:
        rep.add("WARN", "KIDNEY", "Blood or protein in the urine: check urine protein:creatinine ratio and eGFR; "
                                  "consider kidney involvement (IgA nephropathy, amyloidosis, drug effect).")
    if not rep.stop:
        rep.add("INFO", "SOURCE_FIRST", "If no sepsis signs, look for a source (urine or blood culture, chest "
                                    "examination) before starting antibiotics, as stewardship guidance recommends.")
    return rep


def check_drug_options(patient: dict) -> RuleReport:
    """Rules that constrain drug suggestions (ASAS-EULAR 2022 phenotype guidance, simplified)."""
    rep = RuleReport()
    if patient.get("ibd"):
        rep.add("WARN", "IBD_IL17", "IBD present: IL-17 inhibitors are not preferred; a monoclonal TNF "
                                    "inhibitor is the usual choice. Etanercept is less suitable for IBD.")
    if patient.get("uveitis_history"):
        rep.add("INFO", "UVEITIS", "Recurrent uveitis: monoclonal TNF inhibitors are preferred.")
    if patient.get("egfr", 90) < 60:
        rep.add("WARN", "NSAID_KIDNEY", f"eGFR {patient.get('egfr')}: limit NSAIDs and involve the kidney team.")
    if not patient.get("tb_screened", True):
        rep.add("STOP", "TB_SCREEN", "No tuberculosis or hepatitis B screening recorded: screen before any biologic.")
    if patient.get("age", 0) >= 65 or patient.get("cv_risk_high"):
        rep.add("WARN", "JAKI_RISK", "Age 65 or over or high cardiovascular risk: JAK inhibitors need careful "
                                     "consideration of heart, clot and cancer risk.")
    return rep
