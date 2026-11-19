"""Paper diagram generator (tool developed for this study).

Draws the five diagrams of the ABH 2027 paper (Figures 1 to 5). Captions are not drawn inside the
images; they go below the figures in the paper.

Usage:  python tools/paper_figures.py [--dpi 600] [--out docs/figures]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spa_sentinel.roi import run as run_roi, Assumptions
_ap = argparse.ArgumentParser(); _ap.add_argument("--dpi", type=int, default=300); _ap.add_argument("--out", default="docs/figures")
_args = _ap.parse_args()
DPI = _args.dpi
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
INK = "#0b0b0b"; INK2 = "#52514e"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
FILL = {"blue": "#e3eefb", "orange": "#fde6dc", "aqua": "#dcf3ea", "yellow": "#fdf0d2", "violet": "#e7e4f8", "grey": "#f0efec"}
EDGE = {"blue": BLUE, "orange": ORANGE, "aqua": AQUA, "yellow": YELLOW, "violet": VIOLET, "grey": "#8f8e88"}
OUT = _args.out.rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, c="grey", fs=7.5, bold_first=False, wrap=None, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=FILL[c], ec=EDGE[c], lw=0.9))
    lines = text.split("\n")
    if wrap:
        lines = sum([textwrap.wrap(l, wrap) or [""] for l in lines], [])
    if bold_first and lines:
        ax.text(x + w / 2, y + h - 0.012, lines[0], ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)
        ax.text(x + w / 2 if align == "center" else x + 0.008, y + h - 0.012 - fs * 0.0042 * 1.6,
                "\n".join(lines[1:]), ha=align, va="top", fontsize=fs - 0.5, color=INK, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(ax, p, q, c=INK2, lw=1.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=c, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=1, shrinkB=1))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


# ---------- Figure 1: the CRP confusion loop ----------
fig, ax = canvas(6.6, 3.3)
bw, bh = 0.22, 0.2
A = (0.03, 0.62); B = (0.39, 0.62); C = (0.75, 0.62); D = (0.75, 0.12); E = (0.39, 0.12)
box(ax, *A, bw, bh, "1  Inflammatory flare\nBack pain, stiffness, fatigue;\nmay include gut or\nbladder symptoms", "orange", bold_first=True)
box(ax, *B, bw, bh, "2  CRP raised\nSeen outside rheumatology\n(GP, A&E, urology)", "orange", bold_first=True)
box(ax, *C, bw, bh, "3  Read as infection\nAntibiotics started without\na confirmed source", "orange", bold_first=True)
box(ax, *D, bw, bh, "4  Gut flora disturbed\nRepeated antibiotic courses\nlinked to higher IBD risk", "orange", bold_first=True)
box(ax, *E, bw, bh, "5  Inflammation goes on\nNew gut, eye, bladder or\nkidney problems; trust falls", "orange", bold_first=True)
arrow(ax, (A[0] + bw, A[1] + bh / 2), (B[0], B[1] + bh / 2), lw=1.2)
arrow(ax, (B[0] + bw, B[1] + bh / 2), (C[0], C[1] + bh / 2), lw=1.2)
arrow(ax, (C[0] + bw / 2, C[1]), (D[0] + bw / 2, D[1] + bh), lw=1.2)
arrow(ax, (D[0], D[1] + bh / 2), (E[0] + bw, E[1] + bh / 2), lw=1.2)
arrow(ax, (E[0] + 0.02, E[1] + bh), (A[0] + bw - 0.02, A[1]), lw=1.2)
# agent break points
ga = dict(c="aqua", fs=6.5)
box(ax, 0.39, 0.885, 0.22, 0.1, "Agent: shows this patient's own\nCRP baseline and flare history", **ga)
arrow(ax, (0.5, 0.885), (0.5, 0.82), c=AQUA, lw=1.1)
box(ax, 0.75, 0.885, 0.22, 0.1, "Agent: find an infection source\n(culture, urine test) first", **ga)
arrow(ax, (0.86, 0.885), (0.86, 0.82), c=AQUA, lw=1.1)
box(ax, 0.39, 0.385, 0.22, 0.12, "Agent: early warning checks\n(calprotectin, urine tests,\neGFR); fast referral", **ga)
arrow(ax, (0.5, 0.385), (0.5, 0.32), c=AQUA, lw=1.1)
box(ax, 0.03, 0.12, 0.22, 0.2, "Inflammation alert card\nCarried by the patient; tells\nany clinician that raised CRP\nmay be disease activity, and\nhow to reach rheumatology", "aqua", bold_first=True, fs=6.8)
arrow(ax, (0.25, 0.22), (0.39, 0.22), c=AQUA, lw=1.1, ls="--")
ax.text(0.03, 0.02, "Orange: today's loop.   Green: where the proposed agent and alert card break it.", fontsize=6.8, color=INK2)
fig.savefig(OUT + "fig1_loop.png", dpi=DPI); plt.close(fig)

# ---------- Figure 2: agent architecture ----------
fig, ax = canvas(6.6, 3.7)
cols = [0.01, 0.205, 0.405, 0.615, 0.815]; cw = 0.175
heads = ["Data sources", "Perception layer", "Prediction models", "Agent orchestrator", "Outputs to people"]
hc = ["blue", "blue", "violet", "aqua", "yellow"]
for x, h, c in zip(cols, heads, hc):
    ax.text(x + cw / 2, 0.965, h, ha="center", va="center", fontsize=8, fontweight="bold", color=EDGE[c])
src = ["Hospital and GP\nrecords (diagnoses,\nletters, medicines)", "Blood and urine tests\n(CRP, ESR, calprotectin,\neGFR, protein, blood)", "Patient app check-ins\n(BASDAI, pain, sleep,\ngut, bladder)", "Wearables\n(activity, sleep,\nheart rate)", "Imaging reports\n(MRI, X-ray text)"]
for i, s in enumerate(src):
    box(ax, cols[0], 0.79 - i * 0.165, cw, 0.14, s, "blue", fs=5.9)
per = ["Data linking and\nstandard codes\n(FHIR, SNOMED CT)", "Personal baselines\n(each patient's usual\nCRP and symptoms)", "Text reading (NLP)\nof letters and\nreports", "Data quality and\nconsent checks"]
for i, s in enumerate(per):
    box(ax, cols[1], 0.76 - i * 0.205, cw, 0.16, s, "blue", fs=5.9)
mod = ["M1 Flare forecaster\n(gradient-boosted\ntrees + baseline)", "M2 Flare-or-infection\nscorer (calibrated\nlogistic or GBT)", "M3 Organ-spread risk\n(survival models:\ngut, eye, kidney,\nbladder)", "M4 Drug response\nand side effects\n(random forest\n+ SHAP)"]
for i, s in enumerate(mod):
    box(ax, cols[2], 0.76 - i * 0.205, cw, 0.16, s, "violet", fs=5.9)
box(ax, cols[3], 0.40, cw, 0.52, "Reasoning agent\nPlans the next step,\ncalls models and\ntools, explains advice\nin plain words\n\nKnowledge tools:\nASAS-EULAR, NICE,\ndrug safety data,\nlocal formulary\n\nHard safety rules:\nnever delay antibiotics\nif sepsis signs; flag\nIL-17 inhibitor if IBD", "aqua", bold_first=True, fs=6.3)
box(ax, cols[3], 0.145, cw, 0.2, "Human sign-off\nRheumatology team\napproves any change\nto treatment", "aqua", bold_first=True, fs=6.3)
outs = ["Clinician alert in\nthe record (GP,\nA&E, urology)", "Digital and printed\ninflammation\nalert card", "Patient app: flare\nwarnings and\nwhat to do", "Fast-track referral\nand test\nsuggestions", "Drug choice support\nwith side-effect\nchecks"]
for i, s in enumerate(outs):
    box(ax, cols[4], 0.79 - i * 0.165, cw, 0.14, s, "yellow", fs=5.9)
for i in range(5):
    arrow(ax, (cols[0] + cw, 0.86 - i * 0.165), (cols[1], 0.62), c="#9fc3ee", lw=0.6)
for i in range(4):
    arrow(ax, (cols[1] + cw, 0.84 - i * 0.205), (cols[2], 0.84 - i * 0.205), c=INK2, lw=0.8)
    arrow(ax, (cols[2] + cw, 0.84 - i * 0.205), (cols[3], 0.66), c=INK2, lw=0.8)
arrow(ax, (cols[3] + cw / 2, 0.40), (cols[3] + cw / 2, 0.345), c=AQUA, lw=1.0, style="<|-|>")
for i in range(5):
    arrow(ax, (cols[3] + cw, 0.66), (cols[4], 0.86 - i * 0.165), c="#f2c766", lw=0.7)
ax.add_patch(FancyBboxPatch((0.01, 0.015), 0.98, 0.075, boxstyle="round,pad=0.004,rounding_size=0.01", fc=FILL["grey"], ec=EDGE["grey"], lw=0.8))
ax.text(0.5, 0.052, "Governance: V-EGRCSbyD checks, clinical safety case (DCB0129/0160), audit log, bias and drift monitoring, consent and opt-out",
        ha="center", va="center", fontsize=6.0, color=INK)
arrow(ax, (cols[4] + cw / 2, 0.09), (cols[4] + cw / 2, 0.09), lw=0.1)
ax.annotate("", xy=(0.12, 0.09), xytext=(0.9, 0.09))
fig.savefig(OUT + "fig2_arch.png", dpi=DPI); plt.close(fig)

# ---------- Figure 3: drug choice support pathway ----------
fig, ax = canvas(6.6, 3.5)
box(ax, 0.01, 0.78, 0.2, 0.17, "Step 1  Confirm\naxSpA confirmed by\nrheumatologist; ASDAS\nand BASDAI recorded", "blue", bold_first=True, fs=6.8)
box(ax, 0.255, 0.78, 0.22, 0.17, "Step 2  Safety screen\nKidney function, blood\npressure, stomach and heart\nhistory; TB and hepatitis B", "blue", bold_first=True, fs=6.8)
box(ax, 0.52, 0.78, 0.22, 0.17, "Step 3  First-line care\nExercise and physiotherapy;\nNSAID trial (at least two\nNSAIDs over about 4 weeks)", "blue", bold_first=True, fs=6.8)
box(ax, 0.785, 0.78, 0.205, 0.17, "Step 4  Still active?\nASDAS 2.1 or more and\nrheumatologist agrees\nmore treatment is needed", "yellow", bold_first=True, fs=6.8)
for x1, x2 in [(0.21, 0.255), (0.475, 0.52), (0.74, 0.785)]:
    arrow(ax, (x1, 0.865), (x2, 0.865), lw=1.1)
ax.text(0.875, 0.735, "Yes: choose by phenotype", ha="right", fontsize=6.8, color=INK2)
arrow(ax, (0.887, 0.78), (0.887, 0.70), lw=1.1)
ph = [(0.01, "IBD (Crohn's or colitis)\nor recurrent uveitis", "Monoclonal anti-TNF\n(adalimumab, infliximab,\ngolimumab, certolizumab)", "Avoid IL-17 inhibitors\nin active IBD; etanercept\nless suitable for IBD"),
      (0.255, "Significant psoriasis", "IL-17 inhibitor\n(secukinumab, ixekizumab)\nor anti-TNF", "Watch for new gut\nsymptoms; check\ncalprotectin if they appear"),
      (0.52, "No extra-spinal disease", "Anti-TNF, IL-17 inhibitor\nor JAK inhibitor (shared\ndecision with patient)", "JAK inhibitors: check heart,\nclot and cancer risk\nbefore starting"),
      (0.785, "Kidney or urine findings\n(protein, blood, low eGFR)", "Limit NSAIDs; refer to\nkidney team; biologic\nchoice as phenotype", "Treat infection only\nwhen confirmed; avoid\nblind antibiotics")]
for x, a, b, c in ph:
    box(ax, x, 0.53, 0.205 if x > 0.7 else 0.22 if x > 0.2 else 0.2, 0.14, a, "orange", fs=6.8)
    box(ax, x, 0.31, 0.205 if x > 0.7 else 0.22 if x > 0.2 else 0.2, 0.18, b, "aqua", fs=6.6)
    box(ax, x, 0.11, 0.205 if x > 0.7 else 0.22 if x > 0.2 else 0.2, 0.16, c, "grey", fs=6.4)
    xm = x + (0.1025 if x > 0.7 else 0.11 if x > 0.2 else 0.1)
    arrow(ax, (xm, 0.53), (xm, 0.49), lw=0.9); arrow(ax, (xm, 0.31), (xm, 0.27), lw=0.9)
ax.plot([0.11, 0.887], [0.70, 0.70], color=INK2, lw=0.9)
for xm in [0.11, 0.365, 0.63]:
    arrow(ax, (xm, 0.70), (xm, 0.67), lw=0.9)
arrow(ax, (0.887, 0.70), (0.887, 0.67), lw=0.9)
ax.text(0.5, 0.055, "Step 5  Review at about 12 weeks: continue if responding; if not, re-check the diagnosis and other conditions, then switch.", ha="center", fontsize=6.4, color=INK)
ax.text(0.5, 0.012, "Orange: patient feature    Green: options    Grey: cautions", ha="center", fontsize=6.2, color=INK2)
fig.savefig(OUT + "fig3_drugs.png", dpi=DPI); plt.close(fig)

# ---------- Figure 4: evaluation and adoption roadmap ----------
fig, ax = canvas(6.6, 2.3)
ph = [("Phase 1  Retrospective\nvalidation (about 6 months)", "Test models on past records\nfrom two sites; check\nfairness by sex, age, ethnicity", "Gate: AUROC 0.75 or more,\ngood calibration, no group\nmuch worse than others"),
      ("Phase 2  Silent mode\n(3 to 6 months)", "Agent runs in the background;\nclinicians do not see alerts;\nalerts compared with outcomes", "Gate: alert precision agreed\nby clinicians; zero missed\nserious infections"),
      ("Phase 3  Supervised pilot\n(about 12 months)", "One rheumatology clinic, its\nGP cluster and A&E; alert card\nissued; human sign-off", "Gate: fewer blind antibiotic\ncourses and unplanned visits;\nToothbrush Test passed"),
      ("Phase 4  Scale and\nmonitor (ongoing)", "Roll out across the network;\nmonitor drift monthly; re-\nvalidate every year", "Gate: benefits sustained;\nROI reviewed; safety case\nupdated")]
w = 0.215
for i, (a, b, c) in enumerate(ph):
    x = 0.01 + i * 0.25
    box(ax, x, 0.66, w, 0.27, a, "blue", fs=7)
    box(ax, x, 0.34, w, 0.28, b, "grey", fs=6.2)
    box(ax, x, 0.03, w, 0.27, c, "aqua", fs=6.2)
    if i < 3:
        arrow(ax, (x + w + 0.003, 0.795), (x + 0.25 - 0.003, 0.795), lw=1.3)
fig.savefig(OUT + "fig4_roadmap.png", dpi=DPI); plt.close(fig)

# ---------- Figure 5: ROI chart ----------
single = [v / 1000 for v in run_roi(Assumptions())["cumulative_net"]]
network = [v / 1000 for v in run_roi(Assumptions(trusts_sharing=3))["cumulative_net"]]
yrs = [1, 2, 3, 4, 5]
fig = plt.figure(figsize=(3.1, 2.3), dpi=DPI)
ax = fig.add_axes([0.2, 0.17, 0.76, 0.76])
ax.axhline(0, color="#b5b4ae", lw=0.8)
ax.plot(yrs, single, color=BLUE, lw=2, marker="o", ms=4.5, label="Single trust")
ax.plot(yrs, network, color=ORANGE, lw=2, marker="o", ms=4.5, label="Three-trust network")
ax.set_xticks(yrs); ax.set_xlabel("Year", fontsize=7, color=INK2)
ax.set_ylabel("Cumulative net benefit (£ thousand)", fontsize=7, color=INK2)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#b5b4ae")
ax.tick_params(labelsize=6.5, colors=INK2)
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax.text(5.18, single[-1], f"{single[-1]:+.0f}", fontsize=6.5, color=INK, va="center")
ax.text(5.18, network[-1], f"{network[-1]:+.0f}", fontsize=6.5, color=INK, va="center")
ax.set_xlim(0.8, 5.75)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
fig.savefig(OUT + "fig5_roi.png", dpi=DPI); plt.close(fig)
print("figures written to", OUT, "at", DPI, "dpi")
