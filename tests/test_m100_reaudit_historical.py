from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/m100_reaudit'))
import historical


def sample(t, p, counter, d=None, phase='HOVER'):
    return dict(energy_timestamp_s=str(t), electrical_power_W=str(p),
                measurement_energy_Wh=str(counter), measurement_distance_m=d, flight_phase=phase)


def test_timestamp_weighted_power_and_counter_are_independent():
    rows = [sample(5, 0, 12), sample(5.5, 90, 14), sample(7, 30, 17)]
    result = historical.sampled_window(rows, 'HOVER')
    assert result['aligned_mean_power_W'] == 56.25
    assert result['aligned_1Hz_energy_Wh'] == 112.5 / 3600
    assert result['aligned_high_rate_counter_change_Wh'] == 5
    assert result['aligned_outbound_distance_m'] is None


def test_displacement_uses_first_and_last_observed_sample():
    result = historical.sampled_window([sample(3, 2, 1, 10.2, 'CRUISE'),
                                       sample(4, 2, 1.1, 18.2, 'CRUISE')], 'CRUISE')
    assert result['aligned_outbound_distance_m'] == pytest.approx(8)


def test_interrupted_measurement_rejected_without_bridging_gap():
    with pytest.raises(ValueError, match='interrupted'):
        historical.sampled_window([sample(3, 3, 0), sample(4, 4, 0, phase='CLIMB'),
                                   sample(5, 3, 0)], 'HOVER')
    with pytest.raises(ValueError, match='Non-increasing'):
        historical.trapezoid([1, 1], [3, 3])


def test_outputs_cannot_be_under_original_inputs(tmp_path):
    with pytest.raises(ValueError, match='overlaps'):
        historical.safe_destination(tmp_path / 'original' / 'new', [tmp_path / 'original'])
