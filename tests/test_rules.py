"""Safety-rule tests: the rules must behave the same whatever the models say."""
from spa_sentinel.rules import check_raised_crp, check_drug_options, sepsis_screen
from spa_sentinel.roi import run, Assumptions

PT = dict(patient_id="T1", biologic="none", crp_remission=5, crp_flare_typical=30, ibd=0, egfr=90)


def test_sepsis_overrides():
    ep = dict(crp=120, temp_c=38.9, heart_rate=118, wbc=16, site="respiratory")
    assert sepsis_screen(ep) >= 2
    assert check_raised_crp(ep, PT).stop


def test_no_sepsis_flag_for_quiet_flare():
    ep = dict(crp=30, temp_c=36.9, heart_rate=76, wbc=8, site="back")
    assert not check_raised_crp(ep, PT).stop


def test_biologic_warning_and_kidney_check():
    rep = check_raised_crp(dict(crp=35, temp_c=37.1, heart_rate=80, wbc=8, site="urinary", urine_blood=1),
                           dict(PT, biologic="TNFi"))
    codes = {r.code for r in rep.results}
    assert {"BIOLOGIC", "KIDNEY", "SOURCE_FIRST"} <= codes


def test_ibd_blocks_il17_preference():
    codes = {r.code for r in check_drug_options(dict(PT, ibd=1)).results}
    assert "IBD_IL17" in codes


def test_low_egfr_limits_nsaids():
    codes = {r.code for r in check_drug_options(dict(PT, egfr=45)).results}
    assert "NSAID_KIDNEY" in codes


def test_roi_matches_paper():
    single = run()
    assert round(single["annual_benefit"]) == 109780
    assert single["breakeven_year"] == 5
    assert run(Assumptions(trusts_sharing=3))["breakeven_year"] == 3


def test_sepsis_suppresses_source_first_advice():
    ep = dict(crp=120, temp_c=38.9, heart_rate=118, wbc=16, site="respiratory")
    codes = {r.code for r in check_raised_crp(ep, PT).results}
    assert "SEPSIS" in codes and "SOURCE_FIRST" not in codes
