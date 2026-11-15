#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "uav_energy/PowerModel.hpp"

namespace
{
bool Near(const double _left, const double _right, const double _tolerance = 1e-9)
{
  return std::abs(_left - _right) <= _tolerance;
}

void Require(const bool _condition, const std::string &_message)
{
  if (!_condition)
  {
    throw std::runtime_error(_message);
  }
}
}  // namespace

int main()
{
  constexpr double motorConstant = 8.54858e-06;
  constexpr double momentConstant = 0.016;

  const auto stopped = uav_energy::ComputePower(
      {0.0, 0.0, 0.0, 0.0}, motorConstant, momentConstant, 0.70, 10.0, 2000.0);
  Require(Near(stopped.totalShaftPowerW, 0.0), "Stopped shaft power is not zero");
  Require(Near(stopped.electricalLoadW, 10.0), "Stopped load is not auxiliary power");

  const auto low = uav_energy::ComputePower(
      {200.0, -200.0, 200.0, -200.0},
      motorConstant, momentConstant, 0.70, 10.0, 2000.0);
  const auto high = uav_energy::ComputePower(
      {400.0, -400.0, 400.0, -400.0},
      motorConstant, momentConstant, 0.70, 10.0, 2000.0);
  Require(high.totalShaftPowerW > low.totalShaftPowerW, "Power did not increase");
  Require(
      Near(high.totalShaftPowerW / low.totalShaftPowerW, 8.0, 1e-12),
      "Cubic speed-to-power scaling failed");
  Require(high.electricalLoadW > low.electricalLoadW, "Electrical load did not increase");

  const auto clamped = uav_energy::ComputePower(
      {2000.0, 2000.0, 2000.0, 2000.0},
      motorConstant, momentConstant, 0.70, 10.0, 500.0);
  Require(Near(clamped.electricalLoadW, 500.0), "Maximum power clamp failed");

  bool rejected = false;
  try
  {
    static_cast<void>(uav_energy::ComputePower(
        {100.0}, motorConstant, momentConstant, 0.0, 10.0, 2000.0));
  }
  catch (const std::invalid_argument &)
  {
    rejected = true;
  }
  Require(rejected, "Invalid efficiency was accepted");
  return 0;
}
