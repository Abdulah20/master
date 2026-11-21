// Whole-aircraft pack power from the frozen Rodrigues / CMU field observations.
// The two estimators mirror transient_cmu.py and transient_pooled.py. Phase,
// arming, actuator delivery, and the airborne energy clock belong to the caller.
#pragma once

#include "M100TransientModelId.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <QCryptographicHash>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace m100::transient
{
struct Prediction
{
  std::optional<double> power, acceleration, dwell;
  const char *source{"unavailable"};
  const char *regime{""};
  const char *localReason{"INSUFFICIENT_CAUSAL_HISTORY"};
  const char *reason{"INSUFFICIENT_CAUSAL_HISTORY"};
  int supportFlights{0};
  std::string flags;
};

namespace detail
{
using Features = std::array<double, 6>;  // speed, signed d(speed)/dt, up, low-speed dwell, AGL, tilt
using Range = std::array<double, 2>;
using Bands = std::vector<Range>;

template <std::size_t N>
std::array<double, N> Numbers(const QJsonValue &value)
{
  const auto array = value.toArray();
  if (array.size() != static_cast<int>(N))
    throw std::runtime_error("Invalid dimension in CMU transient calibration");
  std::array<double, N> result;
  for (std::size_t i = 0; i < N; ++i)
  {
    result[i] = array.at(static_cast<int>(i)).toDouble(std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(result[i]))
      throw std::runtime_error("Invalid number in CMU transient calibration");
  }
  return result;
}

struct Donor
{
  Features features;
  double power, weight;
  int flight;
};

struct Payload
{
  double kg, intercept, dwellLimit;
  std::array<Range, 7> ranges;
  Bands bands;
  std::array<std::vector<Donor>, 5> groups;
  std::size_t flightSlots{0};
};

struct Policy
{
  double gap, lag, lowSpeed, maxDwell, smallAccel, settlingSpeed, maxSpeed, maxAccel, maxUp;
  Range altitude;
  int minimumFlights;
  double minimumWeight, tiltSlack, dwellSlack, altitudeSlack;
  Features widths;
};

struct Release
{
  Policy policy;
  std::array<double, 6> coefficients;
  std::vector<Payload> payloads;
  double baseMass, gravity;

  static Release Read()
  {
    QFile file(":/m100/calibration/cmu_transient_v1.qz");
    if (!file.open(QIODevice::ReadOnly))
      throw std::runtime_error("Bundled CMU transient calibration is unavailable");
    const QByteArray bytes = qUncompress(file.readAll());
    if (bytes.isEmpty() || QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex() !=
        kCalibrationContentSha256)
      throw std::runtime_error("Bundled CMU transient calibration checksum mismatch");
    const auto doc = QJsonDocument::fromJson(bytes).object();
    if (doc["id"].toString() != kCalibrationId)
      throw std::runtime_error("Unexpected bundled CMU transient release");
    const auto p = doc["policy"].toObject();
    Release r;
    r.policy = {p["max_gap_s"].toDouble(), p["causal_accel_lag_s"].toDouble(),
        p["low_speed_mps"].toDouble(), p["max_low_speed_dwell_s"].toDouble(),
        p["small_accel_mps2"].toDouble(), p["settling_speed_mps"].toDouble(),
        p["max_speed_mps"].toDouble(), p["max_abs_accel_mps2"].toDouble(),
        p["max_abs_vertical_up_mps"].toDouble(), Numbers<2>(p["absolute_agl_m"]),
        p["minimum_support_flights"].toInt(), p["minimum_kernel_weight_s_per_flight"].toDouble(),
        p["local_tilt_edge_tolerance_deg"].toDouble(), p["local_dwell_edge_tolerance_s"].toDouble(),
        p["local_agl_edge_tolerance_m"].toDouble(), Numbers<6>(p["kernel_half_widths"])};
    r.coefficients = Numbers<6>(doc["coefficients"]);
    r.baseMass = doc["base_mass_kg"].toDouble();
    r.gravity = doc["gravity_mps2"].toDouble();
    for (const auto &entry : doc["payloads"].toArray())
    {
      const auto item = entry.toObject();
      Payload payload;
      payload.kg = item["kg"].toDouble();
      payload.intercept = item["intercept_w"].toDouble();
      payload.dwellLimit = item["dwell_limit_s"].toDouble();
      const auto ranges = item["ranges"].toArray();
      for (std::size_t i = 0; i < payload.ranges.size(); ++i)
        payload.ranges[i] = Numbers<2>(ranges.at(static_cast<int>(i)));
      for (const auto &band : item["altitude_bands_m"].toArray())
        payload.bands.push_back(Numbers<2>(band));
      const auto groups = item["groups"].toArray();
      for (std::size_t g = 0; g < payload.groups.size(); ++g)
      {
        const auto rows = groups.at(static_cast<int>(g)).toArray();
        auto &group = payload.groups[g];
        group.reserve(rows.size());
        for (const auto &row : rows)
        {
          const auto v = Numbers<9>(row);
          const int id = static_cast<int>(v[8]);
          group.push_back({{v[0], v[1], v[2], v[3], v[4], v[5]}, v[6], v[7], id});
          payload.flightSlots = std::max(payload.flightSlots, static_cast<std::size_t>(id + 1));
        }
      }
      r.payloads.push_back(std::move(payload));
    }
    return r;
  }
};

inline std::size_t Regime(double speed, double acceleration, const Policy &p)
{
  if (acceleration >= p.smallAccel)
    return 0;
  if (acceleration <= -p.smallAccel)
    return 1;
  if (speed < p.lowSpeed)
    return 2;
  return speed < p.settlingSpeed ? 3 : 4;
}

struct LocalStats
{
  static constexpr double inf = std::numeric_limits<double>::infinity();
  double weight{0}, weightedPower{0}, minTilt{inf}, maxTilt{-inf};
  double minHeight{inf}, maxHeight{-inf}, maxDwell{-inf};
};

inline const char *LocalPower(const Features &x, std::size_t regime, const Payload &payload,
                              const Policy &p, Prediction &out)
{
  if (std::abs(x[2]) > p.maxUp)
    return "VERTICAL_OUTSIDE_NEAR_LEVEL";
  if (x[0] > p.maxSpeed || std::abs(x[1]) > p.maxAccel)
    return "AGGRESSIVE_OR_OUTSIDE_SPEED_ACCEL";
  if (!std::isfinite(x[3]) || x[3] < 0 || x[3] > p.maxDwell)
    return "LONG_OR_UNKNOWN_LOW_SPEED_DWELL";
  if (!std::isfinite(x[4]) || x[4] < p.altitude[0] || x[4] > p.altitude[1])
    return "ALTITUDE_OUTSIDE_FIELD_SUPPORT";
  if (!std::isfinite(x[5]) || x[5] < 0 || x[5] >= 90)
    return "TILT_UNAVAILABLE_OR_INVALID";
  // Donors are already restricted to the trained payload and regime, then sorted by speed.
  const auto &group = payload.groups[regime];
  const auto first = std::lower_bound(group.begin(), group.end(), x[0] - p.widths[0],
      [](const Donor &donor, double speed) { return donor.features[0] < speed; });
  std::vector<LocalStats> flights(payload.flightSlots);
  for (auto it = first; it != group.end() && it->features[0] < x[0] + p.widths[0]; ++it)
  {
    double kernel = 1.0;
    for (std::size_t j = 0; j < x.size(); ++j)
    {
      const double u = (it->features[j] - x[j]) / p.widths[j];
      if (std::abs(u) >= 1.0)
      {
        kernel = 0;
        break;
      }
      kernel *= 1.0 - u*u;
    }
    if (kernel == 0)
      continue;
    auto &s = flights[it->flight];
    const double weight = kernel * it->weight;
    s.weight += weight;
    s.weightedPower += weight * it->power;
    s.minTilt = std::min(s.minTilt, it->features[5]);
    s.maxTilt = std::max(s.maxTilt, it->features[5]);
    s.minHeight = std::min(s.minHeight, it->features[4]);
    s.maxHeight = std::max(s.maxHeight, it->features[4]);
    s.maxDwell = std::max(s.maxDwell, it->features[3]);
  }
  LocalStats limits;
  double sum = 0;
  for (const auto &s : flights)
  {
    if (s.weight < p.minimumWeight)
      continue;
    ++out.supportFlights;
    sum += s.weightedPower / s.weight;  // Each qualifying flight contributes equally.
    limits.minTilt = std::min(limits.minTilt, s.minTilt);
    limits.maxTilt = std::max(limits.maxTilt, s.maxTilt);
    limits.minHeight = std::min(limits.minHeight, s.minHeight);
    limits.maxHeight = std::max(limits.maxHeight, s.maxHeight);
    limits.maxDwell = std::max(limits.maxDwell, s.maxDwell);
  }
  if (out.supportFlights < p.minimumFlights)
    return "INSUFFICIENT_LOCAL_FLIGHTS";
  if (x[5] < limits.minTilt - p.tiltSlack || x[5] > limits.maxTilt + p.tiltSlack)
    return "OUTSIDE_LOCAL_TILT_RANGE";
  if (x[4] < limits.minHeight - p.altitudeSlack || x[4] > limits.maxHeight + p.altitudeSlack)
    return "ALTITUDE_OUTSIDE_FIELD_SUPPORT";
  if (x[0] < p.lowSpeed && x[3] > limits.maxDwell + p.dwellSlack)
    return "DWELL_EXCEEDS_LOCAL_FIELD";
  out.power = sum / out.supportFlights;
  out.source = "cmu_local";
  return "SUPPORTED_LOCAL_FIELD";
}

inline void AddFlag(std::string &flags, const std::string &name)
{
  if (!flags.empty())
    flags += '|';
  flags += name;
}

inline double Bound(double value, const Range &range, const char *name, std::string &flags)
{
  if (value < range[0])
    AddFlag(flags, std::string(name) + "_CLAMPED_LOW");
  else if (value > range[1])
    AddFlag(flags, std::string(name) + "_CLAMPED_HIGH");
  return std::clamp(value, range[0], range[1]);
}

inline void PooledPower(const Features &x, const Payload &payload, const Release &r, Prediction &out)
{
  if (!(std::isfinite(x[4]) && x[4] > 0 && std::isfinite(x[5]) && x[5] >= 0 && x[5] < 90))
  {
    out.reason = "INVALID_OR_MISSING_KINEMATICS";
    return;
  }
  const auto &bounds = payload.ranges;
  const double s = Bound(x[0], bounds[0], "HORIZONTAL_SPEED", out.flags);
  const double a = Bound(x[1], bounds[1], "ACCELERATION", out.flags);
  const double up = Bound(x[2], bounds[2], "VERTICAL_SPEED", out.flags);
  const double tilt = Bound(x[5], bounds[3], "TILT", out.flags);
  if (x[0] < r.policy.lowSpeed && std::abs(x[2]) <= r.policy.maxUp)
  {
    if (!std::isfinite(x[3]))
      AddFlag(out.flags, "UNKNOWN_LOW_SPEED_DWELL_TRANSFER");
    else if (x[3] > payload.dwellLimit)
      AddFlag(out.flags, "LONG_LOW_SPEED_DWELL_TRANSFER");
  }
  const bool inBand = std::any_of(payload.bands.begin(), payload.bands.end(),
      [&](const Range &band) { return band[0] <= x[4] && x[4] <= band[1]; });
  if (!inBand)
  {
    // The source altitude bands may overlap; compare the outer limits independently.
    const auto low = std::min_element(payload.bands.begin(), payload.bands.end(),
        [](const Range &aBand, const Range &bBand) { return aBand[0] < bBand[0]; });
    const auto high = std::max_element(payload.bands.begin(), payload.bands.end(),
        [](const Range &aBand, const Range &bBand) { return aBand[1] < bBand[1]; });
    AddFlag(out.flags, x[4] >= (*low)[0] && x[4] <= (*high)[1]
        ? "ALTITUDE_BETWEEN_OBSERVED_PLATEAUS" : "ALTITUDE_OUTSIDE_TRAINING_RANGE");
  }
  const double m = r.baseMass + payload.kg;
  const double work = m * s * a;
  const double positive = Bound(std::max(work, 0.0), bounds[4], "ACCELERATION_WORK", out.flags);
  const double braking = Bound(std::max(-work, 0.0), bounds[5], "BRAKING_WORK", out.flags);
  const auto &c = r.coefficients;
  const double watts = payload.intercept + c[0]*s + c[1]*s*s + c[2]*m*r.gravity*up +
      c[3]*positive + c[4]*braking + c[5]*(1.0/std::cos(tilt*(std::acos(-1.0)/180.0)) - 1.0);
  if (!(std::isfinite(watts) && watts > 0))
  {
    out.reason = "NONPHYSICAL_REGRESSION_OUTPUT";
    return;
  }
  out.power = Bound(watts, bounds[6], "OUTPUT_POWER", out.flags);
  out.source = "cmu_pooled";
  out.reason = "POOLED_FIELD_ESTIMATE";
}
}  // namespace detail

class Predictor
{
  public: void Load()
  {
    static const auto release = detail::Release::Read();
    this->model = &release;
    this->Reset();
  }

  public: void Reset()
  {
    this->history.clear();
    this->wasLow.reset();
    this->slowSince.reset();
    this->sample.reset();
  }

  // Call through the entire airborne period, including phases with another power source.
  // Duplicate and older stamps give no new sample. Missing motion breaks derivative history.
  public: bool Observe(double time, double speed, double up)
  {
    if (!this->model || !std::isfinite(time) || !std::isfinite(speed) || speed < 0 ||
        !std::isfinite(up))
    {
      this->Reset();
      return false;
    }
    const auto &p = this->model->policy;
    if (!this->history.empty() && time <= this->history.back().first)
      return false;
    if (!this->history.empty() && time - this->history.back().first > p.gap + 1e-9)
      this->Reset();
    const bool low = speed < p.lowSpeed && std::abs(up) <= p.maxUp;
    if (!low)
      this->slowSince.reset();
    else if (this->wasLow && !*this->wasLow)
      this->slowSince = time;
    this->wasLow = low;
    this->history.emplace_back(time, speed);
    const double cutoff = time - p.lag;
    while (this->history.size() >= 3 && this->history[1].first <= cutoff)
      this->history.pop_front();
    if (this->history.size() < 2 || this->history.front().first > cutoff + 1e-9)
      return true;
    const auto &[t0, v0] = this->history[0];
    const auto &[t1, v1] = this->history[1];
    const double past = v0 + (v1 - v0) * (cutoff - t0) / (t1 - t0);
    const double dwell = !low ? 0.0 : this->slowSince ? time - *this->slowSince :
        std::numeric_limits<double>::quiet_NaN();
    this->sample = detail::Features{speed, (speed - past)/p.lag, up, dwell, 0.0, 0.0};
    return true;
  }

  public: Prediction Predict(double payloadKg, double altitudeAgl, double tiltDeg) const
  {
    Prediction out;
    if (!this->sample)
      return out;
    detail::Features x = *this->sample;
    x[4] = altitudeAgl;
    x[5] = tiltDeg;
    out.acceleration = x[1];
    if (std::isfinite(x[3]))
      out.dwell = x[3];
    constexpr const char *regimes[]{"speed_up", "braking", "low_speed_settling",
                                   "settling", "nearly_steady_motion"};
    const auto regime = detail::Regime(x[0], x[1], this->model->policy);
    out.regime = regimes[regime];
    const auto &payloads = this->model->payloads;
    const auto it = std::find_if(payloads.begin(), payloads.end(),
        [payloadKg](const detail::Payload &p) { return p.kg == payloadKg; });
    if (it == payloads.end())
    {
      out.reason = out.localReason = "UNSUPPORTED_PAYLOAD";
      return out;
    }
    out.localReason = detail::LocalPower(x, regime, *it, this->model->policy, out);
    if (out.power)
      out.reason = out.localReason;
    else
      detail::PooledPower(x, *it, *this->model, out);
    return out;
  }

  private: const detail::Release *model{nullptr};
  private: std::deque<std::pair<double, double>> history;
  private: std::optional<bool> wasLow;
  private: std::optional<double> slowSince;
  private: std::optional<detail::Features> sample;
};
}  // namespace m100::transient
