"""Describe frozen cruise-power predictions on the original CMU evaluation flights."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .evaluate_gong_cruise import digest, write_csv


ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "results/m100_reaudit/gong_cruise/flights.csv"
EARLIER_FIVE = {20, 127, 154, 251, 258}
MODELS = ("baseline", "legacy", "selected")
PAIRS = (("legacy", "baseline"), ("selected", "baseline"), ("selected", "legacy"))
LEVELS = {"payload_kg": (0.0, 0.25, 0.50), "altitude_m": (25, 50, 75, 100), "speed_mps": (4, 6, 8)}
DRAWS = 10000
SEED = 20261123


def checked_evaluation(rows):
    with LOCKED.open(newline="") as f:
        source = {int(r["flight_id"]): r for r in csv.DictReader(f)}
    held, seen = [], set()
    for r in rows:
        fid = int(r["flight_id"])
        old = source.get(fid)
        if (old is None or fid in seen or r["split"] != old["split"] or str(r["date"]) != old["date"] or
                any(float(r[k]) != float(old[k]) for k in LEVELS) or
                not np.isclose(float(r["measured_W"]), float(old["measured_W"]), atol=1e-7, rtol=1e-9)):
            raise ValueError(f"Drift in frozen field identity, split, conditions or power for {fid}")
        seen.add(fid)
        if r["split"] == "evaluation":
            item = dict(flight_id=fid, date=old["date"], measured_W=float(r["measured_W"]),
                        **{k: float(r[k]) for k in LEVELS}, **{m: float(r[m+"_W"]) for m in MODELS})
            if not all(np.isfinite(item[k]) for k in MODELS):
                raise ValueError(f"Missing or invalid power prediction for {fid}")
            held.append(item)
    if {r["flight_id"] for r in held} != {fid for fid, r in source.items() if r["split"] == "evaluation"}:
        raise ValueError("Scoring requires the entire original evaluation partition")
    if {r["flight_id"] for r in held if r["payload_kg"] == 0.25 and r["altitude_m"] == 50} != EARLIER_FIVE:
        raise ValueError("Previously examined payload/altitude subgroup changed")
    return sorted(held, key=lambda r: r["flight_id"])


def subsets(held):
    groups = (("evaluation", held),
              ("earlier_five", [r for r in held if r["flight_id"] in EARLIER_FIVE]),
              ("other_evaluation", [r for r in held if r["flight_id"] not in EARLIER_FIVE]))
    for cohort, rows in groups:
        yield cohort, {}, "flight", rows
        yield cohort, {}, "equal_speed", rows
        dimensions = ((k,) for k in ("speed_mps",)) if cohort != "evaluation" else (
            dims for n in range(1, 4) for dims in itertools.combinations(LEVELS, n))
        for dims in dimensions:
            for values in itertools.product(*(LEVELS[k] for k in dims)):
                filters = dict(zip(dims, values, strict=True))
                yield cohort, filters, "flight", [r for r in rows if all(r[k] == v for k, v in filters.items())]


def date_bootstrap_weights(rows, weighting):
    """Rows of cluster multiplicities implement a flight-weighted estimand, preserving pairs."""
    n = len(rows)
    if n == 0:
        return np.empty(0), np.empty((0, 0))
    dates, date_index = np.unique([r["date"] for r in rows], return_inverse=True)
    groups = ([np.ones(n, dtype=bool)] if weighting == "flight" else
              [np.array([r["speed_mps"] == s for r in rows]) for s in LEVELS["speed_mps"]])
    if any(not g.any() for g in groups):
        raise ValueError("Equal-speed comparison needs observations at all three registered speeds")
    point = sum((g / g.sum() for g in groups)) / len(groups)
    if len(dates) < 2:
        return point, np.empty((0, n))
    rng = np.random.default_rng(SEED)
    multiplicity = rng.multinomial(len(dates), np.full(len(dates), 1/len(dates)), size=DRAWS)[:, date_index]
    denominators = np.column_stack([multiplicity[:, mask].sum(axis=1) for mask in groups])
    # Equal-speed resamples that lose a speed cannot estimate the full fixed-speed contrast.
    valid = np.all(denominators > 0, axis=1)
    weights = np.zeros((valid.sum(), n))
    for j, mask in enumerate(groups):
        weights[:, mask] = multiplicity[valid][:, mask] / denominators[valid, j, None] / len(groups)
    return point, weights


def export_scores(rows, out):
    held = checked_evaluation(rows)
    with LOCKED.open(newline="") as f:
        development_dates = {r["date"] for r in csv.DictReader(f) if r["split"] == "fit"}
    evaluation_dates = {r["date"] for r in held}
    metrics, changes = [], []
    for cohort, filters, weighting, sample in subsets(held):
        n, dates = len(sample), len({r["date"] for r in sample})
        group_info = dict(cohort=cohort, grouping="_x_".join(filters) if filters else "all", weighting=weighting,
                          **{k: filters.get(k) for k in LEVELS}, n_flights=n, n_dates=dates,
                          coverage=("EMPTY" if n == 0 else "ONE_FLIGHT" if n == 1 else
                                    "TWO_TO_FOUR_FLIGHTS" if n < 5 else "FIVE_OR_MORE_FLIGHTS"),
                          date_coverage=("NO_DATE" if dates == 0 else "ONE_DATE" if dates == 1 else
                                         "TWO_TO_FOUR_DATES" if dates < 5 else "FIVE_OR_MORE_DATES"))
        weights, draws = date_bootstrap_weights(sample, weighting)
        errors = {}
        for model in MODELS:
            errors[model] = signed = np.array([r[model] - r["measured_W"] for r in sample])
            metrics.append(dict(**group_info, model=model, bias_W=float(weights @ signed) if n else None,
                                MAE_W=float(weights @ np.abs(signed)) if n else None,
                                RMSE_W=float(np.sqrt(weights @ np.square(signed))) if n else None))
        for model, reference in PAIRS:
            diff = np.abs(errors[model]) - np.abs(errors[reference])
            # Matrix-vector dot in a single-threaded reduction also keeps the small cell analyses fast.
            low, high = (np.quantile(np.einsum("ij,j->i", draws, diff), [0.025, 0.975])
                         if len(draws) else (None, None))
            changes.append(dict(**group_info, model=model, reference=reference,
                                MAE_change_W=float(weights @ diff) if n else None,
                                date_bootstrap_95_low_W=low, date_bootstrap_95_high_W=high,
                                valid_draws=len(draws)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "paired_changes.csv", changes)
    exact = [r for r in metrics if r["grouping"] == "_x_".join(LEVELS) and r["model"] == "selected"]
    receipt = dict(
        locked_input=str(LOCKED.relative_to(ROOT)), locked_input_sha256=digest(LOCKED),
        evaluation_ids=[r["flight_id"] for r in held], dates=sorted(evaluation_dates),
        development_dates=sorted(development_dates), shared_dates=sorted(evaluation_dates & development_dates),
        earlier_five_ids=sorted(EARLIER_FIVE), earlier_five_condition={"payload_kg": 0.25, "altitude_m": 50},
        other_evaluation_ids=[r["flight_id"] for r in held if r["flight_id"] not in EARLIER_FIVE],
        registered_levels=LEVELS, exact_cells=len(exact), occupied_cells=sum(r["n_flights"] > 0 for r in exact),
        measurement="One time-weighted measured pack-power mean per flight on its frozen eligible intervals.",
        metrics="Bias is prediction minus measurement. MAE averages absolute errors. RMSE is the square root "
                "of the mean squared error. Primary weights every flight equally. The equal-speed summary gives "
                "one third weight to each commanded speed and equal flight weights within speed.",
        paired_change="Difference of absolute errors of two predictions for each identical field flight. "
                      "Negative mean change indicates a smaller MAE for the named model.",
        date_cluster_bootstrap=dict(seed=SEED, draws=DRAWS, interval="percentile 2.5% and 97.5%",
            method="Sample the observed flight dates with replacement within each subgroup. Retain all flights "
                   "on each sampled date and share date multiplicities across all model predictions. For equal-speed "
                   "weighting, omit replicates missing any registered speed and report the number retained. "
                   "At least two distinct dates are required to calculate an interval. Intervals with few dates are "
                   "descriptive. These intervals quantify pointwise variation across the observed dates only."),
        scope="The original speed evaluation flights belong to the same CMU field collection as the fit set. "
              "Five flights had already been examined in the native paired check. The other 28 are a sensitivity "
              "subgroup of the previously evaluated set. Holdout scores are descriptive for these field conditions.",
    )
    (out / "score_protocol.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt
