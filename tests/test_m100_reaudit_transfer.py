import math
from types import SimpleNamespace

import pytest

from tools.m100_reaudit import transfer


def atmosphere(_temperature, height):
    return SimpleNamespace(density_kg_m3=1.2 - .0001*height)


def test_fit_uses_only_the_commanded_mass_and_reference_height():
    at = [transfer.Commands(p, speed, h) for p, speed, h in
          [(0., 4., 25.), (.25, 6., 50.), (.5, 8., 100.)]]
    get_x = lambda c: transfer.feature(c, atmosphere, 3.715)
    fit = transfer.fit_simulator([(c, 12. + 73. * get_x(c)) for c in at], atmosphere, 3.715)
    assert fit.design_rank == 2
    assert (fit.intercept_W, fit.mass_density_slope_W) == pytest.approx((12., 73.))
    other_speed = transfer.Commands(at[0].payload_kg, 15., at[0].height_m)
    assert get_x(at[0]) == get_x(other_speed)
    with pytest.raises(ValueError, match='Rank-deficient'):
        transfer.fit_simulator([(at[0], 440.), (other_speed, 445.)], atmosphere, 3.715)


def test_errors_and_joint_support_are_flight_weighted_and_signed():
    test = transfer.Commands(.25, 6., 25.)
    training = {test}
    assert transfer.support(test, training) == 'exact_payload_speed_altitude'
    assert transfer.support(transfer.Commands(.25, 6., 50.), training) == 'payload_speed_at_other_altitude'
    assert transfer.support(transfer.Commands(.25, 8., 25.), training) == 'payload_speed_pair_unseen'
    e = transfer.errors([(100., 130.), (110., 102.)])
    assert e['flights'] == 2
    assert (e['mae_W'], e['rmse_W'], e['bias_W']) == pytest.approx((19., math.sqrt(482.), -11.))


def test_nontraining_and_field_targets_cannot_change_predictions(tmp_path):
    original = transfer.run(transfer.SIMULATOR, transfer.FIELD, transfer.FIELD_MANIFEST, tmp_path/'base')
    simulated = transfer.read_rows(transfer.SIMULATOR)
    for row in simulated:
        if (row['split'] != 'train' or row['target_phase'] != 'CRUISE' or
                float(row['wind_speed_mps']) != 0. or float(row['temperature_c']) != 15.):
            row[transfer.LABEL] = 'invalid target outside training'
        row['route_for_cruise_m'] = 'ignored'
    field = transfer.read_rows(transfer.FIELD)
    for r in field:
        r[transfer.OBSERVED] = str(100. + float(r[transfer.OBSERVED])) if r['split']=='test' else 'ignored'
    sim_source, field_source = tmp_path/'sim.csv', tmp_path/'field.csv'
    transfer.write_rows(sim_source, simulated)
    transfer.write_rows(field_source, field)
    updated = transfer.run(sim_source, field_source, transfer.FIELD_MANIFEST, tmp_path/'altered')
    assert updated['fitted'] == original['fitted']
    base = transfer.read_rows(tmp_path/'base/field_predictions.csv')
    changed = transfer.read_rows(tmp_path/'altered/field_predictions.csv')
    assert [(r['flight_id'],r['simulated_fit_prediction_W']) for r in changed] == [
        (r['flight_id'],r['simulated_fit_prediction_W']) for r in base]
    assert updated['field_comparison']['bias_W'] == pytest.approx(original['field_comparison']['bias_W']-100.)
    assert original['simulator_existing_train_cases'] == 95 and original['training_cruise_cases'] == 27
    assert original['fitted']['design_rank'] == 2 and original['field_comparison']['flights'] == 60
    assert sorted(original['training_case_ids']) == sorted(r['case_id'] for r in simulated if
        r['split']=='train' and r['target_phase']=='CRUISE' and float(r['wind_speed_mps'])==0 and float(r['temperature_c'])==15)


def test_field_requires_each_original_test_flight_and_original_commands(tmp_path):
    field = transfer.read_rows(transfer.FIELD)
    test = [r for r in field if r['split']=='test']
    test[0]['commanded_speed_mps']='0.2'
    p = tmp_path/'changed.csv'
    transfer.write_rows(p, test)
    with pytest.raises(ValueError, match='disagrees'):
        transfer.load_field(p, transfer.FIELD_MANIFEST)
    test[0] = test[1].copy()
    transfer.write_rows(p, test)
    with pytest.raises(ValueError, match='60 existing'):
        transfer.load_field(p, transfer.FIELD_MANIFEST)


def test_repeated_simulator_variants_or_split_leaks_fail_closed(tmp_path):
    rows = transfer.read_rows(transfer.SIMULATOR)
    p = tmp_path/'rows.csv'
    transfer.write_rows(p, [*rows,rows[0]])
    with pytest.raises(ValueError, match='Repeated simulator case'):
        transfer.load_training(p)
    copy = dict(rows[0], case_id='new_id', split='test' if rows[0]['split']!='test' else 'train')
    transfer.write_rows(p,[*rows,copy])
    with pytest.raises(ValueError, match='multiple splits'):
        transfer.load_training(p)
