"""Make the screenshots in docs/screenshots from real runs of SpA-Sentinel (tool developed for this study).

Every image is produced by running the package on the synthetic data: terminal images show the
actual text printed by the command-line tool; charts are drawn from the trained models' test results.

Usage:  python tools/make_screenshots.py
"""
import contextlib
import io
import os
import sys
import textwrap
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from spa_sentinel import cli
from spa_sentinel.card import make_card
from spa_sentinel.models import calibration_table, prep_m4, M4_FEATURES, train_m2
from spa_sentinel.roi import run, Assumptions

import argparse
_ap = argparse.ArgumentParser(description="Regenerate README screenshots from live output.")
_ap.add_argument("--dpi", type=int, default=200, help="image resolution (default 200)")
_ap.add_argument("--out", default=os.path.join(ROOT, "docs", "screenshots"), help="output folder")
_args = _ap.parse_args()
DPI = _args.dpi
OUT = _args.out
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def terminal(text, title, fname, width=100):
    lines = []
    for raw in text.rstrip().split("\n"):
        lines += textwrap.wrap(raw, width, subsequent_indent="    ") or [""]
    h = 0.62 + 0.165 * len(lines)
    fig = plt.figure(figsize=(9.2, h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, h)
    ax.add_patch(plt.Rectangle((0, 0), 1, h, color="#1e1f24"))
    ax.add_patch(plt.Rectangle((0, h - 0.34), 1, 0.34, color="#34363d"))
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        ax.add_patch(Ellipse((0.025 + i * 0.024, h - 0.17), 0.055 / 9.2 * 2, 0.11, color=c))
    ax.text(0.5, h - 0.17, title, color="#c9c9c9", ha="center", va="center", fontsize=8.5, family="DejaVu Sans")
    for i, l in enumerate(lines):
        col = "#e6e6e6"
        if "[STOP]" in l or "SEPSIS" in l: col = "#ff7b72"
        elif "[WARN]" in l: col = "#f2cc60"
        elif "[INFO]" in l: col = "#7ee787"
        elif l.startswith("$"): col = "#79c0ff"
        ax.text(0.015, h - 0.5 - i * 0.165, l, color=col, ha="left", va="center", fontsize=7.4, family="DejaVu Sans Mono")
    fig.savefig(os.path.join(OUT, fname), dpi=DPI); plt.close(fig)


def capture(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()


T = cli.load_models()

# 1-2. Agent run: Patient A and the sepsis override
out = capture(cli.cmd_demo, None)
a_part, s_part = out.split("=" * 70)
terminal("$ python -m spa_sentinel.cli demo\n\n" + a_part, "SpA-Sentinel - worked example (Patient A)", "01_agent_patient_a.png")
terminal("$ python -m spa_sentinel.cli demo   (continued)\n" + s_part, "SpA-Sentinel - safety rule overrides the model", "02_sepsis_override.png")

# 3. Model metrics
terminal("$ python -m spa_sentinel.cli metrics\n\n" + cli.metrics_table(T).to_string(index=False),
         "SpA-Sentinel - test-set metrics on synthetic data", "03_model_metrics.png")

# 4. ROC and calibration
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=DPI)
for name, (y, pr), c in [("M1 flare (AUROC %.2f)" % T["m1_metrics"]["auroc"], T["m1_eval"], BLUE),
                         ("M2 infection (AUROC %.2f)" % T["m2_metrics"]["auroc"], T["m2_eval"], ORANGE)]:
    fpr, tpr, _ = roc_curve(y, pr); axes[0].plot(fpr, tpr, color=c, lw=2, label=name)
axes[0].plot([0, 1], [0, 1], ls="--", c="#b5b4ae"); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[0].legend(frameon=False, fontsize=8); axes[0].set_title("ROC curves (held-out patients)", fontsize=10)
y, pr = T["m1_eval"]; ct = calibration_table(y, pr, bins=8)
axes[1].plot(ct["mean_predicted"], ct["observed_rate"], "o-", color=BLUE, lw=2, label="M1 flare forecaster")
lim = max(ct.max()) * 1.1; axes[1].plot([0, lim], [0, lim], ls="--", c="#b5b4ae", label="perfect calibration")
axes[1].set_xlabel("predicted probability"); axes[1].set_ylabel("observed rate"); axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("Calibration of M1", fontsize=10)
for ax in axes:
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "04_roc_calibration.png")); plt.close(fig)

# 5. M2 safety trade-off
rows = []
for tgt in (0.90, 0.95, 0.98):
    _, thr, m, _ = train_m2(cli.load_tables()["crp_episodes"], target_sensitivity=tgt)
    rows.append((tgt, m["sensitivity_infection"], m["specificity"]))
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=DPI)
x = np.arange(len(rows)); w = 0.36
ax.bar(x - w / 2, [r[1] for r in rows], w, color=ORANGE, label="infections caught (sensitivity)")
ax.bar(x + w / 2, [r[2] for r in rows], w, color=BLUE, label="flares cleared (specificity)")
for i, r in enumerate(rows):
    ax.text(i - w / 2, r[1] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=8); ax.text(i + w / 2, r[2] + 0.02, f"{r[2]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"target {int(r[0]*100)}%" for r in rows]); ax.set_ylim(0, 1.3)
ax.set_title("M2: the safety trade-off when choosing a threshold", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "05_m2_tradeoff.png")); plt.close(fig)

# 6. SHAP summary for M4
import shap
X4 = prep_m4(T["m4_test"])[M4_FEATURES]
sv = shap.TreeExplainer(T["m4"]).shap_values(X4)
plt.figure(dpi=DPI); shap.summary_plot(sv, X4, show=False, plot_size=(7, 3.8))
plt.title("M4: what drives predicted TNF-inhibitor response (SHAP)", fontsize=10); plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_shap_m4.png")); plt.close()

# 7. Cox hazard ratios for new IBD
s = T["m3"]["ibd"]["model"].summary
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=DPI)
yy = np.arange(len(s))[::-1]
ax.errorbar(s["exp(coef)"], yy, xerr=[s["exp(coef)"] - s["exp(coef) lower 95%"], s["exp(coef) upper 95%"] - s["exp(coef)"]],
            fmt="o", color=BLUE, ecolor="#9fc3ee", capsize=3)
ax.axvline(1, color="#b5b4ae", ls="--"); ax.set_yticks(yy); ax.set_yticklabels(s.index, fontsize=8); ax.set_xscale("log")
ax.set_xlabel("hazard ratio (log scale)"); ax.set_title("M3: Cox model for new IBD (95% intervals)", fontsize=10)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "07_cox_ibd.png")); plt.close(fig)

# 8. Alert card
make_card(cli.PATIENT_A, "advice line 0113 000 0000 (demo)", os.path.join(OUT, "08_alert_card.png"), dpi=max(DPI, 300))

# 9. ROI tool
terminal("$ python -m spa_sentinel.cli roi\n" + capture(cli.cmd_roi, type("A", (), dict(patients=1200, trusts=1, admit_avoided=0.2))) +
         "\n$ python -m spa_sentinel.cli roi --trusts 3\n" + capture(cli.cmd_roi, type("A", (), dict(patients=1200, trusts=3, admit_avoided=0.2))),
         "SpA-Sentinel - ROI calculator", "09_roi_tool.png")
print("screenshots written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
