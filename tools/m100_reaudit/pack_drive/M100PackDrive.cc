// Model-attached PX4 speed-command limiter. The CMU loaded pack-voltage fit
// describes whole-aircraft electrical demand, not windings, ESCs or absolute SOC.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/Actuators.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace m100
{
namespace
{
using Speeds = std::array<double, 4>;
using Number = std::optional<double>;
using Ecm = gz::sim::EntityComponentManager;
constexpr double kClockEpsilon = 1e-9;
constexpr const char *kDemandTopic = "/m100/pack_drive/demand";
constexpr const char *kStateTopic = "/m100/pack_drive/state";

double Required(const sdf::Element &sdf, const std::string &key)
{
  const auto value = sdf.Get<double>(key, 0.0);
  if (!sdf.HasElement(key) || !value.second || !std::isfinite(value.first))
    throw std::runtime_error("Missing or invalid M100PackDrive SDF value: " + key);
  return value.first;
}

struct Fit
{
  double intercept, r, k, minI, maxI, minQ, maxQ, refV, nominalOmega, maxAge;

  double Voltage(double amps, double q) const
  {
    return this->intercept - this->r * amps - this->k * q;
  }

  double Omega(double volts) const
  {
    return this->nominalOmega * std::clamp(volts / this->refV, 0.0, 1.0);
  }

  static Fit Read(const sdf::Element &sdf)
  {
    Fit f{Required(sdf, "voltageIntercept"), Required(sdf, "currentCoefficient"),
          Required(sdf, "dischargeCoefficient"), Required(sdf, "minCurrent"),
          Required(sdf, "maxCurrent"), Required(sdf, "minDischargedAh"),
          Required(sdf, "maxDischargedAh"), Required(sdf, "referenceVoltage"),
          Required(sdf, "nominalMaxRotVelocity"), Required(sdf, "maxDemandAge")};
    // P(I) must be positive and increasing throughout the identified branch.
    if (f.r <= 0 || f.k <= 0 || f.minI < 0 || f.maxI <= f.minI ||
        f.minQ < 0 || f.maxQ <= f.minQ || f.refV <= 0 || f.nominalOmega <= 0 ||
        f.maxAge <= 0 || f.Voltage(f.maxI, f.maxQ) <= f.r * f.maxI)
      throw std::runtime_error("Invalid M100PackDrive loaded-regression domain");
    return f;
  }
};

struct Demand
{
  double time;
  Number power;
  std::optional<bool> armed;
};

struct Raw
{
  double time;
  Speeds speed;
};

struct Inputs
{
  std::optional<Demand> demand;
  std::optional<Raw> raw;
  std::uint64_t discontinuity{0};
  bool awaitingZero{false};
};

bool Fresh(double now, double stamp, double maxAge)
{
  return stamp <= now + kClockEpsilon && now - stamp <= maxAge + kClockEpsilon;
}

// Each subscription epoch owns a different inbox. Delayed callbacks from an
// old connection can only modify the discarded inbox, even after a rewind.
class Inbox
{
  public: Inbox(const std::string &name, double age, bool requireZero)
      : model(QString::fromStdString(name)), maxAge(age)
  {
    this->latest.awaitingZero = requireZero;
  }

  public: Inputs At(double now)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    if (!this->firstClock)
      this->firstClock = now;
    this->clock = now;
    return this->latest;
  }

  public: void OnDemand(const gz::msgs::StringMsg &message)
  {
    const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(message.data()));
    if (!doc.isObject())
      return;
    const auto obj = doc.object();
    const auto time = obj.value("time_s");
    const auto power = obj.value("power_w");
    const auto armed = obj.value("armed");
    if (obj.value("model").toString() != this->model || !time.isDouble() ||
        !std::isfinite(time.toDouble()) || time.toDouble() < 0 ||
        !(armed.isBool() || armed.isNull()) ||
        !(power.isNull() || (power.isDouble() && std::isfinite(power.toDouble()) &&
                             power.toDouble() >= 0)))
      return;
    Demand next{time.toDouble(), std::nullopt, std::nullopt};
    if (power.isDouble())
      next.power = power.toDouble();
    if (armed.isBool())
      next.armed = armed.toBool();
    std::lock_guard<std::mutex> lock(this->mutex);
    const auto &prev = this->latest.demand;
    if (!this->clock || next.time < *this->firstClock - kClockEpsilon ||
        !Fresh(*this->clock, next.time, this->maxAge) ||
        (prev && next.time <= prev->time))
      return;
    if ((prev && next.time - prev->time > this->maxAge) ||
        !next.armed.value_or(false) || !next.power || *next.power <= 0)
      ++this->latest.discontinuity;
    this->latest.demand = next;
  }

  public: void OnRaw(const gz::msgs::Actuators &message)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    if (!this->clock)
      return;
    // Native PX4 GZMixingInterfaceESC sends no stamp. For that source the
    // best available freshness witness is receipt on this simulation clock.
    double stamp = *this->clock;
    if (message.has_header() && message.header().has_stamp())
    {
      const auto &source = message.header().stamp();
      if (source.sec() < 0 || source.nsec() < 0 || source.nsec() >= 1000000000)
        return;
      stamp = static_cast<double>(source.sec()) + 1e-9 * source.nsec();
      if (stamp < *this->firstClock - kClockEpsilon || !Fresh(*this->clock, stamp, this->maxAge) ||
          (this->lastSourceStamp && stamp <= *this->lastSourceStamp))
        return;
      this->lastSourceStamp = stamp;
    }
    Speeds speed{};
    if (message.velocity_size() != static_cast<int>(speed.size()) ||
        !std::all_of(message.velocity().begin(), message.velocity().end(),
                     [](double x) { return std::isfinite(x); }))
    {
      this->latest.raw.reset();
      ++this->latest.discontinuity;
      return;
    }
    std::copy(message.velocity().begin(), message.velocity().end(), speed.begin());
    const auto peak = *std::max_element(speed.begin(), speed.end());
    if (this->latest.awaitingZero && peak > 0)
      return;
    if (peak <= 0 || (this->latest.raw && stamp - this->latest.raw->time > this->maxAge))
      ++this->latest.discontinuity;
    this->latest.awaitingZero = false;
    this->latest.raw = Raw{stamp, speed};
  }

  private: const QString model;
  private: const double maxAge;
  private: std::mutex mutex;
  private: Number clock;
  private: Number firstClock;
  private: Number lastSourceStamp;
  private: Inputs latest;
};

struct State
{
  const char *status{"WAITING_FOR_DEMAND_CAP_HELD"};
  bool demandFresh{false}, rawFresh{false}, limited{false}, loadPossible{false};
  bool knownIdle{false};
  double capV{0};
  Number requestedV, requestedI, terminalV, current;
  Speeds output{};
};

QJsonValue JsonNumber(Number n)
{
  return n ? QJsonValue(*n) : QJsonValue(QJsonValue::Null);
}

QJsonArray JsonSpeeds(const Speeds &speeds)
{
  QJsonArray array;
  for (const double speed : speeds)
    array.append(speed);
  return array;
}
}  // namespace

class PackDrive final : public gz::sim::System,
                        public gz::sim::ISystemConfigure,
                        public gz::sim::ISystemConfigurePriority,
                        public gz::sim::ISystemReset,
                        public gz::sim::ISystemPreUpdate
{
  public: PriorityType ConfigurePriority() override { return -1; }

  public: void Configure(const gz::sim::Entity &entity,
                         const std::shared_ptr<const sdf::Element> &sdf,
                         Ecm &ecm, gz::sim::EventManager &) override
  {
    const gz::sim::Model model(entity);
    if (!model.Valid(ecm))
      throw std::runtime_error("M100PackDrive must be attached to a model");
    this->modelName = model.Name(ecm);
    this->entity = entity;
    this->fit = Fit::Read(*sdf);
    this->heldV = this->fit.Voltage(this->fit.maxI, this->fit.minQ);
    this->WriteMotors(ecm, {});
    this->publisher = this->outputNode.Advertise<gz::msgs::StringMsg>(kStateTopic);
    if (!this->publisher.Valid())
      throw std::runtime_error("Could not advertise M100PackDrive state");
    this->Connect(false);
  }

  public: void Reset(const gz::sim::UpdateInfo &, Ecm &ecm) override
  {
    // The same physical entity retains its modeled discharged Ah and cap.
    // Only a new instance, attached to a new spawn, starts a new pack ledger.
    this->lastStep.reset();
    this->previousCurrent.reset();
    this->previousLoad = false;
    this->previousIdle = false;
    this->lastPublish.reset();
    this->WriteMotors(ecm, {});
    this->Connect(true);
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &info, Ecm &ecm) override
  {
    if (!ecm.HasEntity(this->entity))
      return;
    const double now = std::chrono::duration<double>(info.simTime).count();
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt < 0 || (this->lastStep && now < *this->lastStep - kClockEpsilon))
      this->Reset(info, ecm);
    const auto inputs = this->inbox->At(now);
    auto state = this->Evaluate(inputs, now);
    const double elapsed = this->lastStep ? now - *this->lastStep : 0;
    if (!info.paused && dt > 0 && elapsed > 0)
    {
      double covered = 0;
      // Only bridge consecutive supported endpoints with the same input
      // history. Never fill missing power, stale commands, or rotor saturation.
      if (elapsed <= this->fit.maxAge && std::abs(elapsed - dt) <= kClockEpsilon &&
          this->previousBreak == inputs.discontinuity && this->previousCurrent &&
          state.current)
      {
        const double amps = 0.5 * (*this->previousCurrent + *state.current);
        covered = std::min(elapsed, (this->fit.maxQ - this->q) * 3600.0 / amps);
        this->q = (covered < elapsed) ? this->fit.maxQ :
                  std::min(this->fit.maxQ, this->q + covered * amps / 3600.0);
        this->coveredSeconds += covered;
        state = this->Evaluate(inputs, now);
      }
      if (this->previousLoad)
        this->uncoveredSeconds += elapsed - covered;
      else if (this->previousIdle)
        this->idleSeconds += elapsed;
    }
    this->heldV = state.capV;
    this->WriteMotors(ecm, state.output);
    const bool changed = this->lastStatus != state.status || this->lastLimited != state.limited;
    if (!this->lastPublish || changed ||
        now - *this->lastPublish >= std::min(0.05, this->fit.maxAge / 2.0))
    {
      this->Publish(inputs, state, now);
      this->lastPublish = now;
      this->lastStatus = state.status;
      this->lastLimited = state.limited;
    }
    this->lastStep = now;
    this->previousCurrent = info.paused ? std::nullopt : state.current;
    this->previousBreak = inputs.discontinuity;
    this->previousLoad = state.loadPossible;
    this->previousIdle = state.knownIdle;
  }

  private: void Connect(bool requireZero)
  {
    this->inputNode.reset();
    this->inbox = std::make_shared<Inbox>(this->modelName, this->fit.maxAge, requireZero);
    this->inputNode = std::make_unique<gz::transport::Node>();
    const auto in = this->inbox;
    if (!this->inputNode->Subscribe<gz::msgs::Actuators>(
            "/" + this->modelName + "/command/motor_speed",
            [in](const gz::msgs::Actuators &m) { in->OnRaw(m); }) ||
        !this->inputNode->Subscribe<gz::msgs::StringMsg>(
            kDemandTopic, [in](const gz::msgs::StringMsg &m) { in->OnDemand(m); }))
      throw std::runtime_error("Could not subscribe M100PackDrive inputs");
  }

  private: void WriteMotors(Ecm &ecm, const Speeds &bounded) const
  {
    // Gazebo 8 MulticopterMotorModel prefers this component over transport.
    // This system is its sole writer on PX4 spawns. Never republish raw input.
    gz::msgs::Actuators command;
    for (const double speed : bounded)
      command.add_velocity(speed);
    ecm.SetComponentData<gz::sim::components::Actuators>(this->entity, command);
  }

  private: State Evaluate(const Inputs &in, double now) const
  {
    State s;
    s.capV = this->heldV;
    s.demandFresh = in.demand && Fresh(now, in.demand->time, this->fit.maxAge);
    s.rawFresh = in.raw && Fresh(now, in.raw->time, this->fit.maxAge);
    const Speeds raw = s.rawFresh ? in.raw->speed : Speeds{};
    const double peak = std::max(0.0, *std::max_element(raw.begin(), raw.end()));
    const bool armKnown = s.demandFresh && in.demand->armed.has_value();
    const bool armed = armKnown && in.demand->armed.value();
    const bool powerKnown = s.demandFresh && in.demand->power.has_value();
    const double power = powerKnown ? in.demand->power.value() : -1.0;
    s.knownIdle = armKnown && !armed;
    s.loadPossible = !s.knownIdle && (armed || peak > 0);
    const double fitQ = std::clamp(this->q, this->fit.minQ, this->fit.maxQ);
    const double atMaxI = this->fit.Voltage(this->fit.maxI, fitQ);
    const bool exhausted = this->q >= this->fit.maxQ;
    const bool highLoad = powerKnown && power > this->fit.maxI * atMaxI;
    const bool lowLoad = powerKnown && power < this->fit.minI * this->fit.Voltage(this->fit.minI, fitQ);
    if (!exhausted && power > 0 && !highLoad && !lowLoad)
    {
      const double a = this->fit.Voltage(0, fitQ);
      const double voltage = 0.5 * (a + std::sqrt(a * a - 4.0 * this->fit.r * power));
      s.requestedV = voltage;
      s.requestedI = power / voltage;
    }
    // Above the identified load, at most allow the fitted maximum-current
    // voltage ceiling. It is a bound on commands, never a delivered V/I claim.
    if (highLoad || exhausted)
      s.capV = std::min(s.capV, atMaxI);
    if (exhausted)
      s.status = "DISCHARGE_DOMAIN_EXHAUSTED_CAP_HELD";
    else if (!s.demandFresh)
      s.status = in.demand ? "STALE_DEMAND_CAP_HELD" : "WAITING_FOR_DEMAND_CAP_HELD";
    else if (!armKnown)
      s.status = "ARM_STATE_UNKNOWN_CAP_HELD";
    else if (!armed)
      s.status = power == 0 ? "DISARMED_ZERO_DEMAND_CAP_HELD" : "DISARMED_CAP_HELD";
    else if (!powerKnown)
      s.status = "POWER_UNAVAILABLE_CAP_HELD";
    else if (power == 0)
      s.status = "ZERO_DEMAND_OUTSIDE_FIT_CAP_HELD";
    else if (highLoad)
      s.status = "CURRENT_ABOVE_FIT_CAP_HELD";
    else if (lowLoad)
      s.status = "CURRENT_BELOW_FIT_CAP_HELD";
    else if (!s.rawFresh)
      s.status = "NO_FRESH_MOTOR_COMMAND_CAP_HELD";
    else if (peak == 0)
      s.status = "RAW_ZERO_OUTSIDE_LOADED_FIT_CAP_HELD";
    else
    {
      s.capV = *s.requestedV;
      if (peak > this->fit.Omega(s.capV))
        s.status = "COMMAND_LIMITED_AT_REQUESTED_VOLTAGE";
      else
      {
        s.terminalV = s.requestedV;
        s.current = s.requestedI;
        s.status = this->q < this->fit.minQ ? "LOADED_FIT_STARTUP_Q_FLOOR" : "LOADED_FIT";
      }
    }
    const double maxSpeed = this->fit.Omega(s.capV);
    s.limited = !s.knownIdle && peak > maxSpeed;
    if (!s.knownIdle && s.rawFresh)
      std::transform(raw.begin(), raw.end(), s.output.begin(),
                     [maxSpeed](double x) { return std::clamp(x, 0.0, maxSpeed); });
    return s;
  }

  private: void Publish(const Inputs &in, const State &s, double now)
  {
    QJsonObject json;
    json["model"] = QString::fromStdString(this->modelName);
    json["time_s"] = now;
    json["status"] = s.status;
    json["demand_fresh"] = s.demandFresh;
    json["demand_time_s"] = JsonNumber(in.demand ? Number(in.demand->time) : std::nullopt);
    json["demand_power_w"] = JsonNumber(s.demandFresh ? in.demand->power : std::nullopt);
    json["armed"] = s.demandFresh && in.demand->armed ? QJsonValue(*in.demand->armed)
                                                       : QJsonValue(QJsonValue::Null);
    json["requested_load_voltage_v"] = JsonNumber(s.requestedV);
    json["requested_pack_current_a"] = JsonNumber(s.requestedI);
    json["voltage_v"] = JsonNumber(s.terminalV);
    json["current_a"] = JsonNumber(s.current);
    json["discharged_ah"] = this->q;
    json["fit_q_ah"] = std::clamp(this->q, this->fit.minQ, this->fit.maxQ);
    json["q_at_lower_boundary"] = this->q < this->fit.minQ;
    json["q_domain_exhausted"] = this->q >= this->fit.maxQ;
    json["covered_load_time_s"] = this->coveredSeconds;
    json["unaccounted_load_time_s"] = this->uncoveredSeconds;
    json["unmodelled_idle_time_s"] = this->idleSeconds;
    json["loaded_charge_coverage_complete"] = this->uncoveredSeconds == 0;
    json["cap_voltage_v"] = s.capV;
    const double omega = this->fit.Omega(s.capV);
    json["omega_limit_rad_s"] = omega;
    const double fraction = omega / this->fit.nominalOmega;
    json["thrust_limit_fraction"] = fraction * fraction;
    json["command_limited"] = s.limited;
    json["raw_command_fresh"] = s.rawFresh;
    json["raw_command_time_s"] = JsonNumber(in.raw ? Number(in.raw->time) : std::nullopt);
    json["awaiting_post_reset_zero_command"] = in.awaitingZero;
    json["raw_motor_rad_s"] = s.rawFresh ? QJsonValue(JsonSpeeds(in.raw->speed))
                                         : QJsonValue(QJsonValue::Null);
    json["applied_motor_rad_s"] = JsonSpeeds(s.output);
    // Clamp status cannot identify load served, motor currents or delivered
    // electrical power. The supported estimates above are at aircraft level.
    json["delivered_power_w"] = QJsonValue(QJsonValue::Null);
    gz::msgs::StringMsg message;
    message.set_data(QJsonDocument(json).toJson(QJsonDocument::Compact).toStdString());
    this->publisher.Publish(message);
  }

  private: gz::sim::Entity entity{gz::sim::kNullEntity};
  private: std::string modelName;
  private: Fit fit{};
  private: double q{0}, heldV{0};
  private: double coveredSeconds{0}, uncoveredSeconds{0}, idleSeconds{0};
  private: Number lastStep, previousCurrent, lastPublish;
  private: std::uint64_t previousBreak{0};
  private: bool previousLoad{false}, previousIdle{false}, lastLimited{false};
  private: std::string lastStatus;
  private: gz::transport::Node outputNode;
  private: gz::transport::Node::Publisher publisher;
  private: std::shared_ptr<Inbox> inbox;
  private: std::unique_ptr<gz::transport::Node> inputNode;
};
}  // namespace m100

GZ_ADD_PLUGIN(m100::PackDrive, gz::sim::System,
              m100::PackDrive::ISystemConfigure, m100::PackDrive::ISystemConfigurePriority,
              m100::PackDrive::ISystemReset, m100::PackDrive::ISystemPreUpdate)
