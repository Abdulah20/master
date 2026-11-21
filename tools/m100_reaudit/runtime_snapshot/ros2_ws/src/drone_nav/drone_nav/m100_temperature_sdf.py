"""Deterministic derived-SDF generation for M100 temperature experiments.

This module does not calculate atmospheric density.  It accepts the already
validated density ratio from :mod:`m100_atmosphere` and applies that ratio only
to the V2 thrust coefficient (``motorConstant``).  The V2 rotor-drag and
rolling-moment terms are deliberately disabled and must remain exactly zero.
Canonical PX4 model files are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET

from .m100_atmosphere import is_reference_temperature_c


SCHEMA = "M100_CMU_V2_TEMPERATURE_DERIVED_SDF_V2"
REFERENCE_TEMPERATURE_C = 15.0
MOTOR_PLUGIN_NAME = "gz::sim::systems::MulticopterMotorModel"
BASE_MODEL_NAME = "m100_cmu_v2"
PAYLOAD_MODEL_NAMES = (
    BASE_MODEL_NAME,
    "m100_cmu_v2_payload_025",
    "m100_cmu_v2_payload_050",
    "m100_cmu_v2_payload_075",
    "m100_cmu_v2_payload_100",
    "m100_cmu_v2_payload_125",
)
SCALED_COEFFICIENT_TAGS = ("motorConstant",)
DISABLED_ZERO_COEFFICIENT_TAGS = (
    "rotorDragCoefficient",
    "rollingMomentCoefficient",
)


@dataclass(frozen=True)
class GeneratedTemperatureModel:
    """Immutable identity and paths for one derived Gazebo model."""

    model_name: str
    payload_model_name: str
    case_sha256: str
    sdf_sha256: str
    scaled_base_sdf_sha256: str
    model_directory: Path
    model_sdf_path: Path
    scaled_base_model_name: str
    scaled_base_model_directory: Path
    scaled_base_sdf_path: Path
    manifest_path: Path
    launcher_alias_path: Path
    scaled_base_launcher_alias_path: Path | None
    temperature_c: float
    reference_altitude_m: float
    density_ratio: float
    source_coefficients: Mapping[str, float]
    effective_coefficients: Mapping[str, float]


def _finite_positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _safe_model_name(value: str) -> str:
    name = str(value)
    if Path(name).name != name or name not in PAYLOAD_MODEL_NAMES:
        allowed = ", ".join(PAYLOAD_MODEL_NAMES)
        raise ValueError(f"payload_model_name must be one of: {allowed}")
    return name


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_xml(data: bytes, source: Path) -> ET.Element:
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        return ET.fromstring(data, parser=parser)
    except ET.ParseError as error:
        raise ValueError(f"invalid SDF XML: {source}: {error}") from error


def _model_element(root: ET.Element, source: Path) -> ET.Element:
    model = root.find("./model")
    if model is None:
        raise ValueError(f"SDF has no top-level model: {source}")
    return model


def _motor_plugins(model: ET.Element, source: Path) -> list[ET.Element]:
    plugins = [
        element for element in model.findall("./plugin")
        if element.get("name") == MOTOR_PLUGIN_NAME
    ]
    if len(plugins) != 4:
        raise ValueError(
            f"expected exactly four {MOTOR_PLUGIN_NAME} plugins in {source}; "
            f"found {len(plugins)}"
        )
    numbers = sorted(
        int(plugin.findtext("motorNumber", "-1")) for plugin in plugins
    )
    if numbers != [0, 1, 2, 3]:
        raise ValueError(f"invalid motorNumber mapping in {source}: {numbers}")
    return plugins


def _read_source_coefficients(
    plugins: list[ET.Element], source: Path
) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    for tag in SCALED_COEFFICIENT_TAGS:
        values: list[float] = []
        for plugin in plugins:
            element = plugin.find(tag)
            if element is None or element.text is None:
                raise ValueError(f"motor plugin lacks {tag}: {source}")
            values.append(_finite_positive(float(element.text), tag))
        reference = values[0]
        if any(not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-18)
               for value in values[1:]):
            raise ValueError(f"motor plugins disagree on {tag}: {source}")
        coefficients[tag] = reference
    return coefficients


def _read_disabled_zero_coefficients(
    plugins: list[ET.Element], source: Path
) -> dict[str, float]:
    """Fail closed unless every explicitly disabled V2 term remains zero."""

    coefficients: dict[str, float] = {}
    for tag in DISABLED_ZERO_COEFFICIENT_TAGS:
        values: list[float] = []
        for plugin in plugins:
            element = plugin.find(tag)
            if element is None or element.text is None:
                raise ValueError(f"motor plugin lacks {tag}: {source}")
            values.append(_finite(float(element.text), tag))
        if any(value != 0.0 for value in values):
            raise ValueError(
                f"{tag} must remain exactly zero in the active V2 model: {source}"
            )
        coefficients[tag] = 0.0
    return coefficients


def _xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _install_launcher_alias(alias: Path, model_directory: Path) -> None:
    """Expose an immutable output model to PX4's fixed model-root launcher."""

    target = model_directory.resolve()
    if alias.is_symlink():
        if alias.resolve() != target:
            raise RuntimeError(f"generated-model alias points elsewhere: {alias}")
        return
    if alias.exists():
        raise RuntimeError(f"refusing to replace existing PX4 model path: {alias}")
    try:
        alias.symlink_to(target, target_is_directory=True)
    except FileExistsError:
        if not alias.is_symlink() or alias.resolve() != target:
            raise RuntimeError(f"generated-model alias race at {alias}")


def _model_config_bytes(model_name: str) -> bytes:
    return (
        "<?xml version=\"1.0\"?>\n"
        "<model>\n"
        f"  <name>{model_name}</name>\n"
        "  <version>1.0</version>\n"
        "  <sdf version=\"1.9\">model.sdf</sdf>\n"
        "</model>\n"
    ).encode("utf-8")


def generate_temperature_model(
    *,
    px4_models_root: str | Path,
    output_root: str | Path,
    payload_model_name: str,
    temperature_c: float,
    reference_altitude_m: float,
    density_ratio: float,
) -> GeneratedTemperatureModel:
    """Generate and validate one non-reference-temperature M100 model.

    The directory identity is a SHA-256 of the immutable inputs and canonical
    source hashes.  Repeating the same case is idempotent.  A 15 degree C case
    is rejected so the caller must use the canonical model byte-for-byte.
    """

    models_root = Path(px4_models_root).resolve()
    generated_root = Path(output_root).resolve()
    payload_name = _safe_model_name(payload_model_name)
    temperature = _finite(temperature_c, "temperature_c")
    altitude = _finite(reference_altitude_m, "reference_altitude_m")
    ratio = _finite_positive(density_ratio, "density_ratio")
    if altitude < 0.0:
        raise ValueError("reference_altitude_m must be non-negative")
    if is_reference_temperature_c(temperature):
        raise ValueError("15 C must use the canonical M100 model without generation")

    source_base_path = models_root / BASE_MODEL_NAME / "model.sdf"
    source_payload_path = models_root / payload_name / "model.sdf"
    if not source_base_path.is_file():
        raise FileNotFoundError(source_base_path)
    if not source_payload_path.is_file():
        raise FileNotFoundError(source_payload_path)

    source_base = source_base_path.read_bytes()
    source_payload = source_payload_path.read_bytes()
    base_root = _parse_xml(source_base, source_base_path)
    base_model = _model_element(base_root, source_base_path)
    plugins = _motor_plugins(base_model, source_base_path)
    source_coefficients = _read_source_coefficients(plugins, source_base_path)
    disabled_zero_coefficients = _read_disabled_zero_coefficients(
        plugins, source_base_path
    )
    effective_coefficients = {
        tag: value * ratio for tag, value in source_coefficients.items()
    }

    case_record = {
        "schema": SCHEMA,
        "payload_model_name": payload_name,
        "temperature_c": temperature,
        "reference_temperature_c": REFERENCE_TEMPERATURE_C,
        "reference_altitude_m": altitude,
        "density_ratio": ratio,
        "source_base_sdf_sha256": _sha256(source_base),
        "source_payload_sdf_sha256": _sha256(source_payload),
        "source_coefficients": source_coefficients,
        "effective_coefficients": effective_coefficients,
        "disabled_zero_coefficients": disabled_zero_coefficients,
    }
    canonical_record = json.dumps(
        case_record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    case_sha256 = _sha256(canonical_record)
    model_name = f"m100_cmu_v2_temp_{case_sha256[:24]}"
    model_directory = generated_root / model_name
    model_sdf_path = model_directory / "model.sdf"
    scaled_base_model_name = f"{model_name}_base"
    scaled_base_model_directory = generated_root / scaled_base_model_name
    scaled_base_sdf_path = scaled_base_model_directory / "model.sdf"
    manifest_path = model_directory / "manifest.json"
    launcher_alias_path = models_root / model_name
    scaled_base_launcher_alias_path: Path | None = (
        models_root / scaled_base_model_name
    )

    for plugin in plugins:
        for tag, effective in effective_coefficients.items():
            element = plugin.find(tag)
            assert element is not None
            element.text = format(effective, ".17g")
    base_model.set("name", scaled_base_model_name)
    scaled_base = _xml_bytes(base_root)

    if payload_name == BASE_MODEL_NAME:
        base_model.set("name", model_name)
        final_sdf = _xml_bytes(base_root)
        scaled_base = final_sdf
        scaled_base_model_name = model_name
        scaled_base_model_directory = model_directory
        scaled_base_sdf_path = model_sdf_path
        scaled_base_launcher_alias_path = None
    else:
        payload_root = _parse_xml(source_payload, source_payload_path)
        payload_model = _model_element(payload_root, source_payload_path)
        includes = payload_model.findall("./include")
        matching = [
            include for include in includes
            if include.findtext("uri", "").strip()
            == f"model://{BASE_MODEL_NAME}"
        ]
        if len(matching) != 1:
            raise ValueError(
                f"expected one model://{BASE_MODEL_NAME} include in "
                f"{source_payload_path}"
            )
        uri = matching[0].find("uri")
        assert uri is not None
        # Resolve through the deterministic PX4 model root rather than an
        # output-root-specific file URI.  This keeps the wrapper SDF bytes and
        # case hash portable while remaining compatible with sdformat's model
        # URI resolver.
        uri.text = f"model://{scaled_base_model_name}"
        payload_model.set("name", model_name)
        final_sdf = _xml_bytes(payload_root)

    final_sha256 = _sha256(final_sdf)
    scaled_base_sha256 = _sha256(scaled_base)
    manifest = {
        **case_record,
        "case_sha256": case_sha256,
        "model_name": model_name,
        "model_sdf_sha256": final_sha256,
        "scaled_base_sdf_sha256": scaled_base_sha256,
        "model_sdf_path": str(model_sdf_path),
        "scaled_base_sdf_path": str(scaled_base_sdf_path),
        "scaled_base_model_name": scaled_base_model_name,
        "scaled_base_model_directory": str(scaled_base_model_directory),
        "launcher_alias_path": str(launcher_alias_path),
        "scaled_base_launcher_alias_path": (
            str(scaled_base_launcher_alias_path)
            if scaled_base_launcher_alias_path is not None else None
        ),
        "temperature_effect_scope": "ROTOR_DENSITY_FIXED_ALTITUDE_APPROXIMATION",
        "scaled_coefficient_tags": list(SCALED_COEFFICIENT_TAGS),
        "disabled_zero_coefficient_tags": list(
            DISABLED_ZERO_COEFFICIENT_TAGS
        ),
        "battery_temperature_physical": False,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    model_directory.mkdir(parents=True, exist_ok=True)
    if scaled_base_sdf_path != model_sdf_path:
        _atomic_write(scaled_base_sdf_path, scaled_base)
        _atomic_write(
            scaled_base_model_directory / "model.config",
            _model_config_bytes(scaled_base_model_name),
        )
    _atomic_write(model_sdf_path, final_sdf)
    _atomic_write(manifest_path, manifest_bytes)
    _install_launcher_alias(launcher_alias_path, model_directory)
    if scaled_base_launcher_alias_path is not None:
        _install_launcher_alias(
            scaled_base_launcher_alias_path,
            scaled_base_model_directory,
        )

    # Read back from disk so READY can depend on the actual launch artifact,
    # not merely on the in-memory tree used to create it.
    if _sha256(model_sdf_path.read_bytes()) != final_sha256:
        raise RuntimeError("generated temperature SDF hash verification failed")
    if _sha256(scaled_base_sdf_path.read_bytes()) != scaled_base_sha256:
        raise RuntimeError("scaled base temperature SDF hash verification failed")
    parsed_base = _parse_xml(scaled_base_sdf_path.read_bytes(), scaled_base_sdf_path)
    verified_plugins = _motor_plugins(
        _model_element(parsed_base, scaled_base_sdf_path), scaled_base_sdf_path
    )
    verified = _read_source_coefficients(verified_plugins, scaled_base_sdf_path)
    _read_disabled_zero_coefficients(verified_plugins, scaled_base_sdf_path)
    for tag, expected in effective_coefficients.items():
        if not math.isclose(verified[tag], expected, rel_tol=1e-14, abs_tol=0.0):
            raise RuntimeError(f"generated temperature coefficient mismatch: {tag}")

    return GeneratedTemperatureModel(
        model_name=model_name,
        payload_model_name=payload_name,
        case_sha256=case_sha256,
        sdf_sha256=final_sha256,
        scaled_base_sdf_sha256=scaled_base_sha256,
        model_directory=model_directory,
        model_sdf_path=model_sdf_path,
        scaled_base_model_name=scaled_base_model_name,
        scaled_base_model_directory=scaled_base_model_directory,
        scaled_base_sdf_path=scaled_base_sdf_path,
        manifest_path=manifest_path,
        launcher_alias_path=launcher_alias_path,
        scaled_base_launcher_alias_path=scaled_base_launcher_alias_path,
        temperature_c=temperature,
        reference_altitude_m=altitude,
        density_ratio=ratio,
        source_coefficients=dict(source_coefficients),
        effective_coefficients=dict(effective_coefficients),
    )
