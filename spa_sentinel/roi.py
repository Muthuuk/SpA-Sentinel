"""Illustrative ROI model (paper, Section 5.3 and Table 5). Change the assumptions to local figures."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Assumptions:
    patients: int = 1200
    abx_courses_per_pt: float = 0.5; abx_avoided: float = 0.30; abx_cost: float = 50
    admit_rate: float = 0.06; admit_avoided: float = 0.20; admit_cost: float = 3200
    ae_rate: float = 0.15; ae_avoided: float = 0.20; ae_cost: float = 300
    bio_starts: int = 70; bio_avoided: float = 0.10; bio_cost: float = 2500
    minutes_saved_per_pt: float = 12; clinician_hourly: float = 110
    build: float = 90000; integration: float = 60000; governance_training: float = 30000; running: float = 55000
    years: int = 5; ramp_year1: float = 0.5; trusts_sharing: int = 1


def run(a: Assumptions = Assumptions()) -> dict:
    items = {
        "antibiotic courses avoided": a.patients * a.abx_courses_per_pt * a.abx_avoided * a.abx_cost,
        "admissions avoided": a.patients * a.admit_rate * a.admit_avoided * a.admit_cost,
        "A&E visits avoided": a.patients * a.ae_rate * a.ae_avoided * a.ae_cost,
        "ineffective biologic courses avoided": a.bio_starts * a.bio_avoided * a.bio_cost,
        "clinician time released": a.patients * a.minutes_saved_per_pt / 60 * a.clinician_hourly,
    }
    benefit = sum(items.values())
    one_off = (a.build + a.integration + a.governance_training) / a.trusts_sharing
    costs = [one_off + a.running] + [a.running] * (a.years - 1)
    bens = [benefit * a.ramp_year1] + [benefit] * (a.years - 1)
    cum, net, breakeven = 0, [], None
    for y in range(a.years):
        cum += bens[y] - costs[y]; net.append(cum)
        if breakeven is None and cum >= 0:
            breakeven = y + 1
    return dict(items=items, annual_benefit=benefit, costs=costs, benefits=bens, cumulative_net=net,
                roi=net[-1] / sum(costs), breakeven_year=breakeven, assumptions=asdict(a))
