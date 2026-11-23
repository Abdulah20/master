"""Independent QA of a speed-treatment decision on frozen CMU steady-flight means."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/m100_reaudit"
RELEASE = RESULTS / "cruise_power_decision"
KNOWN_IDS = {20, 127, 154, 251, 258}
MODES = {"fit", "evaluation"}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def gazebo_cruise_w(payload, target_agl_m):
    """Compute the nominal pack proposal independently of the fitted treatment."""
    kelvin_at_datum = 288.15
    kelvin = kelvin_at_datum - .0065 * target_agl_m
    pressure = 101325. * (kelvin / kelvin_at_datum)**(9.80665 / (287.05287 * .0065))
    density = pressure / (287.05287 * kelvin)
    return 68.9 * (3.715 + payload)**1.5 / math.sqrt(density) + 16.8


def legacy_delta_w(command):
    return -18.369385003709404 - 4.357496259094581 * (command - 6.)


@pytest.fixture(scope="module")
def original():
    """Inputs come from the archived flight-level recalculation of signed V*I."""
    result = []
    for r in csv_rows(RESULTS / "gong_cruise/flights.csv"):
        flight = int(r["flight_id"])
        speed, payload, h, duration, measured = (float(r[k]) for k in (
            "speed_mps", "payload_kg", "altitude_m", "duration_s", "measured_W"))
        assert duration >= 20. and measured * duration / 3600. == pytest.approx(float(r["measured_Wh"]), abs=1e-10)
        baseline = gazebo_cruise_w(payload, h)
        result.append(dict(flight_id=flight, split=r["split"], date=r["date"], payload_kg=payload,
                           altitude_m=h, speed_mps=speed, measured_W=measured, baseline_W=baseline,
                           legacy_W=baseline+legacy_delta_w(speed), field_rodrigues_W=float(r["rodrigues_W"])))
    return result


def test_locked_cohorts_are_flight_disjoint_but_share_all_dates(original):
    original_ids = {r["flight_id"] for r in original}
    assert len(original_ids) == len(original) == 94
    cohorts = {k: [r for r in original if r["split"] == k] for k in MODES}
    assert len(cohorts["fit"]) == 61 and len(cohorts["evaluation"]) == 33
    fit_dates = {r["date"] for r in cohorts["fit"]}
    assert len(fit_dates) == 14
    assert {r["date"] for r in cohorts["evaluation"]} == fit_dates
    assert {r["speed_mps"] for r in original} == {4., 6., 8.}
    previously_inspected = csv_rows(RESULTS / "speed_pack_link/field_per_flight_errors.csv")
    previous = {int(r["field_flight_id"]) for r in previously_inspected if r["window"] == "locked_stable"}
    exact = {r["flight_id"] for r in cohorts["evaluation"] if r["payload_kg"] == .25 and r["altitude_m"] == 50}
    assert previous == exact == KNOWN_IDS
    field_context=json.loads((RESULTS/"external/validation.json").read_bytes())
    raw_bag=json.loads((RESULTS/"cmu_component_evidence/raw_channels.json").read_bytes())
    assert field_context["ambient_wind_ground_truth"] is False and "KAGC METAR" in field_context["density"]
    assert set(raw_bag["battery_definition"]["absent_from_this_schema"]) >= {"temperature","cell_temperature"}


def test_supervisor_originals_distinguish_steady_fit_and_airborne_training_eligibility():
    folder = RESULTS / "supervisor_rerun"
    rows = csv_rows(folder / "M100_69_experiments_rerun.csv")
    assert len(rows) == 69 and {float(r["cruise_speed_mps"]) for r in rows} == {0., 5., 10., 15., 20.}
    assert sum(r["training_eligible"] == "True" for r in rows) == 6
    with zipfile.ZipFile(folder / "recorder_files.zip") as records:
        for r in rows:
            path = f"{r['case_id']}/attempt_{r['attempt']}/{r['run_id']}"
            doc = json.loads(records.read(path + ".provenance.json"))
            native_csv = f"{r['case_id']}/attempt_{r['attempt']}/training/{r['run_id']}.csv"
            native, = csv.DictReader(io.StringIO(records.read(native_csv).decode("utf-8")))
            assert doc["energy_summary"]["stable_cruise_speed_applied"] is False
            assert doc["stable_cruise_speed_model"] is None
            assert float(r["estimated_airborne_energy_Wh"]) == pytest.approx(float(native["estimated_airborne_energy_Wh"]), abs=1e-10)
            assert float(r["measurement_energy_Wh"]) == pytest.approx(doc["energy_summary"]["measurement_window_energy_wh"], abs=1e-10)
            assert float(r["airborne_mean_power_W"]) == pytest.approx(
                float(r["estimated_airborne_energy_Wh"]) * 3600. / float(r["airborne_duration_s"]), abs=1e-8)
    support = {int(r["command_mps"]): r for r in csv_rows(RESULTS / "speed_pack_link/field_speed_support.csv")}
    for cmd in (5, 10, 15, 20):
        assert all(int(support[cmd][f"locked_stable_{fold}_flights"]) == 0 for fold in MODES)
    assert (int(support[10]["whole_phase_fit_flights"]), int(support[10]["whole_phase_evaluation_flights"])) == (26, 10)
    assert all(int(support[v]["whole_phase_evaluation_flights"]) == 0 for v in (5, 15, 20))


@pytest.mark.parametrize("replace", [{"split": "evaluation"}, {"speed_mps": 5.}, {"speed_mps": 10.},
                                     {"speed_mps": 15.}, {"speed_mps": 20.}, {"payload_kg": .75},
                                     {"measured_W": float("nan")}, {"baseline_W": float("inf")}])
def test_fit_rejects_out_of_cohort_or_unusable_input(original, replace):
    from tools.m100_reaudit.cruise_power_decision import fit_policy

    fit = [r.copy() for r in original if r["split"] == "fit"]
    fit[0].update(replace)
    with pytest.raises(ValueError):
        fit_policy(fit)


def test_fit_rejects_duplicated_flights(original):
    from tools.m100_reaudit.cruise_power_decision import fit_policy

    fit = [r for r in original if r["split"] == "fit"]
    with pytest.raises(ValueError):
        fit_policy([*fit, dict(fit[0])])


def test_prediction_does_not_use_holdout_labels_or_retrospective_density(original):
    from tools.m100_reaudit.cruise_power_decision import fit_policy, predict

    policy = fit_policy([r for r in original if r["split"] == "fit"])
    for row in (r for r in original if r["split"] == "evaluation"):
        inputs = {k: row[k] for k in ("payload_kg", "altitude_m", "speed_mps", "baseline_W")}
        expected = predict(policy, inputs)
        changed = dict(row, measured_W=99999., field_rodrigues_W=1., legacy_W=0., flight_id=-1, date="2099-01-01")
        assert math.isfinite(expected) and predict(policy, changed) == pytest.approx(expected, abs=1e-10)


def test_no_speed_fit_reaches_supervisor_or_environmental_extrapolation(original):
    from tools.m100_reaudit.cruise_power_decision import predict

    # Deliberately nonzero if selection slips through a domain gate.
    policy={"enabled_payloads_kg": [0., .25, .5], "coefficients":{"offset_W": -10.}}
    row=next(r for r in original if r["payload_kg"]==.25 and r["speed_mps"]==4.)
    assert predict(policy,row)==pytest.approx(row["baseline_W"]-10.)
    for change in ([dict(speed_mps=v) for v in (0., 5., 10., 12., 15., 20.)]
                   + [dict(altitude_m=200.),dict(payload_kg=1.),dict(wind_speed_mps=3.),dict(temperature_c=25.)]):
        assert predict(policy,{**row,**change}) == row["baseline_W"]


def test_archived_native_grid_reproduces_the_physical_power_base_and_domain():
    from tools.m100_reaudit.cruise_native_inputs import reference_grid

    for (payload,h,speed),inputs in reference_grid().items():
        base=gazebo_cruise_w(payload,h)
        assert inputs["baseline_W"] == pytest.approx(base, abs=1e-9, rel=0.)
        if speed in {4., 6., 8.}:
            assert inputs["legacy_W"] == pytest.approx(base+legacy_delta_w(speed), abs=1e-9, rel=0.)
        else:
            assert inputs["legacy_adjustment_W"] is None and inputs["legacy_W"]==inputs["baseline_W"]
    short=reference_grid(altitudes=(50.,),speeds=(8.,),route_length_m=150.)
    assert all(r["legacy_adjustment_W"] is None for r in short.values())
    observations={r["case_id"]:r for r in csv_rows(RESULTS/"speed_pack_link/native_flights.csv")}
    traces=json.loads((RELEASE/"native_reference_checks.json").read_bytes())["traces"]
    assert len(traces)==6 and {t["case_id"] for t in traces}==observations.keys()
    for trace in traces:
        native=observations[trace["case_id"]]
        assert trace["archive_sha256"]==native["raw_archive_sha256"] and trace["run_id"]==native["run_id"]
        assert trace["terminal_packets"]>0 and max(trace["maximum_absolute_residual_W"].values())<1e-7
        assert float(native["stable_pack_W"])==pytest.approx(trace["expected_W"],abs=1e-7,rel=0.)


def raw_scores(rows):
    result={}
    for model in ("baseline","legacy","selected"):
        errors=[float(r[model+"_W"])-float(r["measured_W"]) for r in rows]
        result[model]=dict(mae_W=statistics.mean(abs(e) for e in errors),
                           rmse_W=math.sqrt(statistics.mean(e*e for e in errors)),
                           bias_W=statistics.mean(errors))
    return result


def test_frozen_release_rows_metrics_gates_and_digests_agree_with_sources(original):
    from tools.m100_reaudit.cruise_power_decision import fit_policy, predict

    proto_path=RELEASE/"protocol.json"
    protocol=json.loads(proto_path.read_bytes())
    decision=json.loads((RELEASE/"decision.json").read_bytes())
    assert decision["protocol_sha256"]==hashlib.sha256(proto_path.read_bytes()).hexdigest()
    for name,sha in protocol["fixed_sources_sha256"].items():
        assert hashlib.sha256((RESULTS/name).read_bytes()).hexdigest()==sha
    assert set(protocol["previously_inspected_matching_flights"])==KNOWN_IDS
    expected={r["flight_id"]:r for r in original}
    train=[r for r in original if r["split"]=="fit"]
    policy=fit_policy(train)
    for key in ("model","enabled_payloads_kg"):
        assert policy[key]==decision["policy"][key]
    assert policy["coefficients"]==pytest.approx(decision["policy"]["coefficients"],rel=1e-10,abs=1e-9)
    if policy["model"]=="native_speed":
        # Closed-form two-parameter check against field V*I minus simulated base,
        # independent of the candidate-selection and least-squares implementation.
        line=statistics.linear_regression([r["speed_mps"]-6. for r in train],
                                          [r["measured_W"]-r["baseline_W"] for r in train])
        assert policy["coefficients"] == pytest.approx(
            {"offset_W": line.intercept, "slope_W_per_mps": line.slope}, abs=1e-9, rel=0.)
    rows=csv_rows(RELEASE/"flights.csv")
    assert {int(r["flight_id"]) for r in rows} == expected.keys() and len(rows)==len(expected)
    for r in rows:
        original_row=expected[int(r["flight_id"])]
        assert r["split"]==original_row["split"] and r["date"]==original_row["date"]
        for key in ("measured_W","baseline_W","legacy_W","field_rodrigues_W","speed_mps","payload_kg","altitude_m"):
            assert float(r[key])==pytest.approx(original_row[key],abs=1e-8,rel=0.)
        assert float(r["selected_W"])==pytest.approx(predict(policy,original_row),abs=1e-8,rel=0.)
    held=[r for r in rows if r["split"]=="evaluation"]
    old=[r for r in held if int(r["flight_id"]) in KNOWN_IDS]
    other=[r for r in held if int(r["flight_id"]) not in KNOWN_IDS]
    assert sorted(decision["held_out_flight_ids"])==sorted(int(r["flight_id"]) for r in held)
    assert sorted(decision["fit_flight_ids"])==sorted(r["flight_id"] for r in train)
    scores={key:raw_scores(selected) for key,selected in (
        ("heldout", held),("previously_inspected_five",old),("previously_unmatched_28",other),
        ("nested_development",csv_rows(RELEASE/"cv_flights.csv")))}
    for name,score in scores.items():
        for model,metrics in score.items():
            assert decision[name][model]==pytest.approx(metrics,abs=1e-9,rel=0.)
    s,f,c=(scores[name] for name in ("heldout","previously_inspected_five","nested_development"))
    gates=dict(held_mae_le_baseline=s["selected"]["mae_W"]<=s["baseline"]["mae_W"],
               held_mae_lt_legacy=s["selected"]["mae_W"]<s["legacy"]["mae_W"],
               held_rmse_le_legacy=s["selected"]["rmse_W"]<=s["legacy"]["rmse_W"],
               outer_date_mae_le_refit_legacy=c["selected"]["mae_W"]<=c["legacy"]["mae_W"],
               prior_five_mae_le_legacy=f["selected"]["mae_W"]<=f["legacy"]["mae_W"])
    assert decision["gates"]==gates and decision["deploy"] is all(gates.values())


def test_outer_predictions_refit_everything_without_the_omitted_date(original):
    from tools.m100_reaudit.cruise_power_decision import fit_policy, predict

    fit=[r for r in original if r["split"]=="fit"]
    cv=csv_rows(RELEASE/"cv_flights.csv")
    by_id={int(r["flight_id"]):r for r in cv}
    assert len(by_id)==len(cv)==len(fit) and by_id.keys()=={r["flight_id"] for r in fit}
    for date in sorted({r["date"] for r in fit}):
        train=[r for r in fit if r["date"]!=date]
        policy=fit_policy(train)
        # Legacy's training residual uses retrospective field density. Its
        # out-of-fold evaluation still applies that residual to the Gazebo base.
        legacy=statistics.linear_regression([r["speed_mps"]-6. for r in train],
                                            [r["measured_W"]-r["field_rodrigues_W"] for r in train])
        for r in (r for r in fit if r["date"]==date):
            stored=by_id[r["flight_id"]]
            assert stored["date"]==date and stored["outer_selected_model"]==policy["model"]
            assert (stored["outer_payload_enabled"]=="True")== (r["payload_kg"] in policy["enabled_payloads_kg"])
            assert float(stored["legacy_W"])==pytest.approx(
                r["baseline_W"]+legacy.intercept+legacy.slope*(r["speed_mps"]-6.),abs=1e-8,rel=0.)
            assert float(stored["selected_W"])==pytest.approx(predict(policy,r),abs=1e-8,rel=0.)
