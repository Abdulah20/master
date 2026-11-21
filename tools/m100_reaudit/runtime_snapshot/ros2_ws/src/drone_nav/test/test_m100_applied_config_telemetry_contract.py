from pathlib import Path


MANAGER = Path(__file__).parents[1] / "drone_nav/m100_experiment_manager.py"


def test_status_exposes_the_applied_factor_configuration() -> None:
    source = MANAGER.read_text(encoding="utf-8")

    required_mappings = (
        '"temperature_c": config.temperature_c if config else None',
        '"wind_speed_command_mps":',
        'config.wind_speed_mps if config else None',
        '"wind_direction_command_deg":',
        'config.wind_direction_deg if config else None',
        '"commanded_speed_mps":',
        'config.cruise_speed_mps if config else None',
        '"commanded_altitude_m": config.altitude_m if config else None',
        '"route_length_command_m":',
        'config.route_length_m if config else None',
        '"hover_duration_command_s":',
        'config.hover_duration_s if config else None',
    )
    for mapping in required_mappings:
        assert mapping in source


def test_temperature_claim_is_gated_by_the_verified_derived_runtime_model() -> None:
    source = MANAGER.read_text(encoding="utf-8")

    assert '"temperature_physical": temperature_physical' in source
    assert "generated_temperature_model is not None" in source
    assert (
        "self._active_runtime_model == generated_temperature_model.model_name"
        in source
    )
    assert '"battery_temperature_physical": False' in source
    assert 'TEMPERATURE_EFFECT_SCOPE = "ROTOR_DENSITY_FIXED_ALTITUDE_APPROXIMATION"' in source


def test_dynamics_uses_case_reference_and_energy_uses_current_height() -> None:
    source = MANAGER.read_text(encoding="utf-8")

    assert "compare_density_to_15c_reference(" in source
    assert "density_ratio=atmosphere.density_ratio" in source
    assert "self._applied_density_ratio = selection.density_ratio" in source
    assert "air_density_kg_m3=power_density" in source
    assert "self._air_state_for_power(sample" in source
    assert "self._propulsion.thrust_constant_n_per_rad_s2" in source
    assert "* self._applied_density_ratio" in source
    assert "STATIC_THRUST_RESERVE_DENSITY_STATUS" in source
    assert "STATIC_THRUST_RESERVE_DENSITY_BASIS" in source


def test_15c_is_a_literal_canonical_identity_branch() -> None:
    source = MANAGER.read_text(encoding="utf-8")
    selection = source.split("def _temperature_runtime_selection", 1)[1].split(
        "def _external_px4_pids", 1
    )[0]

    assert "if is_reference_temperature_c(config.temperature_c):" in selection
    assert "px4_target=config.payload.target" in selection
    assert "gazebo_model=config.payload.model" in selection
    assert "generated_model=None" in selection
