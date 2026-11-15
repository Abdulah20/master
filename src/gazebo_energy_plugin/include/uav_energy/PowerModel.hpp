#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace uav_energy
{
struct PowerResult
{
  std::vector<double> rotorShaftPowerW;
  double totalShaftPowerW{0.0};
  double electricalLoadW{0.0};
};

inline PowerResult ComputePower(
    const std::vector<double> &_physicalRotorSpeedRadS,
    const double _motorConstant,
    const double _momentConstant,
    const double _propulsionEfficiency,
    const double _auxiliaryPowerW,
    const double _maximumElectricalPowerW)
{
  if (_physicalRotorSpeedRadS.empty())
  {
    throw std::invalid_argument("At least one rotor speed is required");
  }
  if (!std::isfinite(_motorConstant) || _motorConstant <= 0.0 ||
      !std::isfinite(_momentConstant) || _momentConstant <= 0.0)
  {
    throw std::invalid_argument("Motor and moment constants must be finite and positive");
  }
  if (!std::isfinite(_propulsionEfficiency) || _propulsionEfficiency <= 0.0 ||
      _propulsionEfficiency > 1.0)
  {
    throw std::invalid_argument("Propulsion efficiency must be in (0, 1]");
  }
  if (!std::isfinite(_auxiliaryPowerW) || _auxiliaryPowerW < 0.0 ||
      !std::isfinite(_maximumElectricalPowerW) ||
      _maximumElectricalPowerW < _auxiliaryPowerW)
  {
    throw std::invalid_argument("Electrical power limits are invalid");
  }

  PowerResult result;
  result.rotorShaftPowerW.reserve(_physicalRotorSpeedRadS.size());
  for (const double speed : _physicalRotorSpeedRadS)
  {
    if (!std::isfinite(speed))
    {
      throw std::invalid_argument("Rotor speed must be finite");
    }
    const double magnitude = std::abs(speed);
    const double shaftPower =
        _momentConstant * _motorConstant * magnitude * magnitude * magnitude;
    result.rotorShaftPowerW.push_back(shaftPower);
    result.totalShaftPowerW += shaftPower;
  }

  const double unconstrainedLoad =
      _auxiliaryPowerW + result.totalShaftPowerW / _propulsionEfficiency;
  result.electricalLoadW = std::clamp(
      unconstrainedLoad, _auxiliaryPowerW, _maximumElectricalPowerW);
  return result;
}
}  // namespace uav_energy
