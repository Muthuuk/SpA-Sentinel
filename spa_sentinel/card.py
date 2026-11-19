"""Inflammation alert card (paper, Section 4.5): a printable PNG with a QR-style link placeholder."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def make_card(patient: dict, contact: str, path: str = "alert_card.png") -> str:
    fig = plt.figure(figsize=(3.37, 2.13), dpi=300)  # credit-card size, inches
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#dcf3ea", ec="#1baf7a", lw=1.5))
    ax.add_patch(FancyBboxPatch((0.02, 0.80), 0.96, 0.17, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#1baf7a", ec="#1baf7a"))
    ax.text(0.5, 0.885, "INFLAMMATION ALERT CARD", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
    bio = patient.get("biologic", "none")
    txt = (f"Diagnosis: axial spondyloarthritis (AS)\n"
           f"My usual CRP: about {patient['crp_remission']:.0f} mg/L (well), {patient['crp_flare_typical']:.0f} mg/L (flare)\n"
           f"Medicines: {('biologic - ' + bio + ' (higher infection risk)') if bio != 'none' else 'no biologic'}\n\n"
           "A raised CRP may reflect my inflammatory disease.\n"
           "Please look for an infection source (urine or blood\n"
           "culture) before antibiotics. Do NOT delay antibiotics\n"
           f"if sepsis is suspected.\nRheumatology: {contact}")
    ax.text(0.06, 0.75, txt, ha="left", va="top", fontsize=5.3, color="#0b0b0b", linespacing=1.35)
    ax.text(0.94, 0.06, "Demo card - synthetic data", ha="right", va="bottom", fontsize=4, color="#52514e")
    fig.savefig(path, dpi=300); plt.close(fig)
    return path
