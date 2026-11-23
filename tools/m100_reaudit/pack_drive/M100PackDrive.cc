// Model-attached PX4 speed-command limiter. The CMU loaded pack-voltage fit
// describes whole-aircraft electrical demand, not windings, ESCs or absolute SOC.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QStringList>
#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Actuators.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include "M100TransientPower.hh"
#include "M100TheveninFit.hh"

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
constexpr const char *kLandTopic = "/m100/pack_drive/land";
// PX4 publishes land detection on changes or once a second at unchanged state.
constexpr double kLandReportAge = 1.5;
constexpr double kIntegralHistorySeconds = 5.0;

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
  double time{};
  Number power, phasePower, managerPower, payload;
  std::optional<bool> armed, landed;
  QString phase, phaseSource, runId;

  bool Transient() const
  {
    return this->phaseSource == "UNSUPPORTED_PHASE" &&
           (this->phase == "STABILIZING" || this->phase == "RETURN_BRAKING");
  }
};

struct LandReport
{
  double time, transition;
  std::uint64_t sequence;
  bool landed;
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
  std::optional<LandReport> land;
  std::uint64_t discontinuity{0};
  bool awaitingZero{false};
};

bool Fresh(double now, double stamp, double maxAge)
{
  return stamp <= now + kClockEpsilon && now - stamp <= maxAge + kClockEpsilon;
}

Number Nonnegative(const QJsonValue &value)
{
  if (value.isDouble() && std::isfinite(value.toDouble()) && value.toDouble() >= 0)
    return value.toDouble();
  return std::nullopt;
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
    Demand next;
    next.time = time.toDouble();
    next.power = Nonnegative(power);
    next.phasePower = Nonnegative(obj.value("phase_power_w"));
    next.managerPower = Nonnegative(obj.value("manager_power_w"));
    next.payload = Nonnegative(obj.value("payload_kg"));
    next.phase = obj.value("phase").toString();
    next.phaseSource = obj.value("phase_source").toString(obj.value("source").toString());
    next.runId = obj.value("run_id").toString();
    if (armed.isBool())
      next.armed = armed.toBool();
    if (obj.value("landed").isBool())
      next.landed = obj.value("landed").toBool();
    std::lock_guard<std::mutex> lock(this->mutex);
    const auto &prev = this->latest.demand;
    if (!this->clock || next.time < *this->firstClock - kClockEpsilon ||
        !Fresh(*this->clock, next.time, this->maxAge) ||
        (prev && next.time <= prev->time))
      return;
    const auto &land = this->latest.land;
    const bool nativeArmedInAir = land && Fresh(*this->clock, land->time, kLandReportAge) &&
                                  !land->landed && land->armed.value_or(false);
    if ((prev && next.time - prev->time > this->maxAge) ||
        (next.armed == false && !nativeArmedInAir) || !(next.power.value_or(0) > 0 ||
                                next.phasePower.value_or(0) > 0 || next.Transient()))
      ++this->latest.discontinuity;
    this->latest.demand = next;
  }

  public: void OnLand(const gz::msgs::StringMsg &message)
  {
    const auto obj = QJsonDocument::fromJson(QByteArray::fromStdString(message.data())).object();
    const auto time = Nonnegative(obj.value("time_s"));
    const auto edge = Nonnegative(obj.value("transition_s"));
    const auto seq = Nonnegative(obj.value("sequence"));
    if (obj.value("model").toString() != this->model || !time || !edge || !seq ||
        *edge > *time + kClockEpsilon || *seq < 1 || *seq > 9007199254740991.0 ||
        std::floor(*seq) != *seq || !obj.value("landed").isBool())
      return;
    LandReport next{*time, *edge, static_cast<std::uint64_t>(*seq),
                    obj.value("landed").toBool(), std::nullopt};
    if (obj.value("armed").isBool())
      next.armed = obj.value("armed").toBool();
    std::lock_guard<std::mutex> lock(this->mutex);
    const auto &old = this->latest.land;
    // A PX4 callback can arrive one physics tick ahead of PreUpdate. Consume it
    // only after that stamped simulation time has actually been reached.
    if (!this->clock || *time < *this->firstClock - kClockEpsilon ||
        *time > *this->clock + this->maxAge ||
        (old && (next.time < old->time || (next.time == old->time && next.sequence < old->sequence))))
      return;
    this->latest.land = next;
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

enum class Source { Pack, Local, Pooled, Manager, Missing };
constexpr std::array<const char *, 5> kSourceNames{
    "native_pack_model", "cmu_local", "cmu_pooled", "recorded_manager", "unavailable"};

struct Sum
{
  double seconds{}, wh{};
};
using Totals = std::array<Sum, kSourceNames.size()>;

struct Slice
{
  double start, end, p0, p1;
  Source source;
  Totals before;

  Totals At(double time) const
  {
    Totals value = this->before;
    const double dt = std::clamp(time - this->start, 0.0, this->end - this->start);
    auto &sum = value[static_cast<std::size_t>(this->source)];
    sum.seconds += dt;
    const double endPower = this->p0 + (this->p1 - this->p0) * dt / (this->end - this->start);
    sum.wh += (this->p0 + endPower) * 0.5 * dt / 3600.0;
    return value;
  }
};

// Save cumulative primitives at physics ticks. PX4 reports transitions after
// their source stamps. Clipping a stored trapezoid recovers both boundaries
// without assigning a ROS receive time or an entire publish interval to air.
class Integrals
{
  public: void Append(double start, double end, Source source, double p0 = 0, double p1 = 0)
  {
    if (end <= start)
      return;
    this->slices.push_back({start, end, p0, p1, source, this->total});
    this->total = this->slices.back().At(end);
    while (this->slices.front().end < end - kIntegralHistorySeconds)
      this->slices.pop_front();
  }

  public: std::optional<Totals> At(double time) const
  {
    for (auto it = this->slices.rbegin(); it != this->slices.rend(); ++it)
      if (it->start <= time + kClockEpsilon && time <= it->end + kClockEpsilon)
        return it->At(time);
    return std::nullopt;
  }

  public: const Totals &Current() const { return this->total; }
  private: Totals total{};
  private: std::deque<Slice> slices;
};

QJsonValue JsonNumber(Number n)
{
  return n ? QJsonValue(*n) : QJsonValue(QJsonValue::Null);
}

struct AirborneBudget
{
  QString run, status{"WAITING_FOR_RECORDING"}, reason;
  double recordingStart{};
  Number start, end;
  std::optional<LandReport> last;
  std::optional<Totals> origin, final;

  void Observe(const LandReport &land, const Integrals &energy, bool recording)
  {
    if (this->status == "INCOMPLETE" || this->run.isEmpty())
      return;
    if (this->end)
    {
      // A manual recording may include another sortie. A single planned-flight
      // row cannot use just the first completed gate of this multi-flight run.
      if (recording && land.sequence != this->last->sequence)
        this->Fail("MULTIPLE_OR_DISCONTINUOUS_AIRBORNE_WINDOWS");
      return;
    }
    if (!this->last)
    {
      if (land.landed)
      {
        this->last = land;
        this->status = "WAITING_FOR_TAKEOFF";
      }
      else if (land.transition >= this->recordingStart - kClockEpsilon)
        this->Fail("NO_PRECEDING_GROUND_OBSERVATION");
      return;
    }
    if (land.sequence == this->last->sequence && land.landed == this->last->landed &&
        land.transition == this->last->transition)
      return;
    if (land.sequence != this->last->sequence + 1 || land.landed == this->last->landed ||
        land.transition < this->last->transition || land.transition < this->recordingStart - kClockEpsilon)
    {
      this->Fail("DISCONTINUOUS_PX4_LAND_SEQUENCE");
      return;
    }
    this->last = land;
    if (!land.landed)
    {
      this->start = land.transition;
      this->origin = energy.At(land.transition);
      if (!this->origin)
        this->Fail("TAKEOFF_OUTSIDE_INTEGRATION_HISTORY");
      else
        this->status = "AIRBORNE";
      return;
    }
    this->end = land.transition;
    this->final = energy.At(land.transition);
    if (!this->final || !this->origin || !this->start || *this->end <= *this->start)
    {
      this->Fail("LANDING_OUTSIDE_INTEGRATION_HISTORY");
      return;
    }
    const auto missing = static_cast<std::size_t>(Source::Missing);
    if ((*this->final)[missing].seconds != (*this->origin)[missing].seconds)
      this->Fail("MISSING_AIRBORNE_POWER");
    else
      this->status = "COMPLETE";
  }

  void Fail(const char *detail)
  {
    this->status = "INCOMPLETE";
    this->reason = detail;
  }

  QJsonObject Json(const Integrals &energy, double now) const
  {
    QJsonObject object, sources;
    object["run_id"] = this->run.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(this->run);
    object["boundary_source"] = "PX4_VEHICLE_LAND_DETECTED";
    object["calibration_id"] = transient::kCalibrationId;
    object["status"] = this->status;
    object["reason"] = this->reason;
    object["start_s"] = JsonNumber(this->start);
    object["end_s"] = JsonNumber(this->end);
    const bool running = this->status == "AIRBORNE";
    object["duration_s"] = JsonNumber(this->start && (this->end || running) ?
        Number(this->end.value_or(now) - *this->start) : std::nullopt);
    const Totals &value = this->final ? *this->final : energy.Current();
    const bool available = this->origin && (running || this->final);
    double covered = 0, wh = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Source::Missing); ++i)
    {
      const double seconds = available ? value[i].seconds - (*this->origin)[i].seconds : 0;
      const double wattsHours = available ? value[i].wh - (*this->origin)[i].wh : 0;
      sources[kSourceNames[i]] = QJsonObject{{"seconds", seconds}, {"energy_Wh", wattsHours}};
      covered += seconds;
      wh += wattsHours;
    }
    object["sources"] = sources;
    object["covered_s"] = covered;
    object["known_energy_Wh"] = wh;
    const auto missing = static_cast<std::size_t>(Source::Missing);
    object["uncovered_s"] = JsonNumber(available ? Number(value[missing].seconds -
                                           (*this->origin)[missing].seconds) : std::nullopt);
    object["estimated_airborne_energy_Wh"] = JsonNumber(this->status == "COMPLETE" ? Number(wh) : std::nullopt);
    return object;
  }
};

struct State
{
  const char *status{"WAITING_FOR_DEMAND_CAP_HELD"};
  bool demandFresh{false}, rawFresh{false}, limited{false}, loadPossible{false};
  bool knownIdle{false};
  double capV{0};
  Number requestedV, requestedI, terminalV, current;
  Number effectivePower, managerPower, managerExpires;
  Source source{Source::Missing};
  Speeds output{};
};

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
    const char *setting = std::getenv("M100_PACK_DYNAMICS");
    const std::string mode = setting ? setting : "cmu_2rc";
    if (mode != "cmu_2rc" && mode != "cmu_static")
      throw std::runtime_error("M100_PACK_DYNAMICS must be cmu_2rc or cmu_static");
    this->dynamicPack = mode == "cmu_2rc";
    if (this->dynamicPack)
    {
      // Use the instantaneous resistance only once. Polarization is advanced
      // from supported past current and subtracted separately from the source.
      this->fit.intercept = rcfit::referenceV;
      this->fit.r = rcfit::r0;
      this->fit.k = rcfit::slope;
      const double minimumSource = this->fit.Voltage(0, this->fit.maxQ) -
                                   (rcfit::r1 + rcfit::r2) * this->fit.maxI;
      if (minimumSource <= 2.0 * this->fit.r * this->fit.maxI)
        throw std::runtime_error("Two-RC operating branch is not monotone within the current range");
    }
    this->predictor.Load();
    if (const char *initial = std::getenv("M100_INITIAL_SOC_PERCENT"))
    {
      bool ok = false;
      this->initialSoc = QString::fromUtf8(initial).toDouble(&ok);
      if (!ok || !std::isfinite(this->initialSoc) || this->initialSoc < 0 || this->initialSoc > 100)
        throw std::runtime_error("M100_INITIAL_SOC_PERCENT must be in [0, 100]");
    }
    // Translating a scenario SOC to an equivalent charge offset uses the
    // TB47D nameplate only. It is a transferable initial condition, not a BMS reading.
    this->initialQ = (100.0 - this->initialSoc) * 4.5 / 100.0;
    this->q = this->initialQ;
    this->heldV = this->fit.Voltage(this->fit.maxI, std::clamp(this->q, this->fit.minQ, this->fit.maxQ));
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
    this->previousState = State{};
    this->previousLoad = false;
    this->previousIdle = false;
    this->lastPublish.reset();
    this->body = gz::sim::kNullEntity;
    this->groundZ.reset();
    this->ResetMotion();
    this->integrals = {};
    this->budget = {};
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
    if (!info.paused && dt > 0 && (!this->lastStep || now > *this->lastStep))
      this->UpdateMotion(inputs, now, ecm);
    auto state = this->Evaluate(inputs, now);
    const double elapsed = this->lastStep ? now - *this->lastStep : 0;
    if (elapsed > 0)
    {
      double covered = 0;
      const bool contiguous = !info.paused && dt > 0 && elapsed <= this->fit.maxAge &&
                              std::abs(elapsed - dt) <= kClockEpsilon;
      // Only bridge consecutive supported endpoints with the same input
      // history. Never fill missing power, stale commands, or rotor saturation.
      if (contiguous && this->previousBreak == inputs.discontinuity && this->previousCurrent &&
          state.current)
      {
        const double amps = 0.5 * (*this->previousCurrent + *state.current);
        covered = std::min(elapsed, (this->fit.maxQ - this->q) * 3600.0 / amps);
        this->q = (covered < elapsed) ? this->fit.maxQ :
                  std::min(this->fit.maxQ, this->q + covered * amps / 3600.0);
        this->coveredSeconds += covered;
        this->AdvancePolarization(*this->previousCurrent, covered);
      }
      // A declared zero-power disarmed interval allows relaxation. A missing
      // sample, saturated rotor or unobserved interval never implies zero load.
      if (contiguous && this->previousBreak == inputs.discontinuity && this->previousIdle &&
          this->previousState.demandFresh && this->previousState.effectivePower == 0.0 &&
          state.knownIdle && state.demandFresh && state.effectivePower == 0.0)
        this->AdvancePolarization(0, elapsed);
      this->IntegratePower(*this->lastStep, now, covered, contiguous, state);
      if (covered > 0)
        state = this->Evaluate(inputs, now);
      if (this->previousLoad)
        this->uncoveredSeconds += elapsed - covered;
      else if (this->previousIdle)
        this->idleSeconds += elapsed;
    }
    this->heldV = state.capV;
    this->WriteMotors(ecm, state.output);
    const QString previousBudget = this->budget.status;
    if (state.demandFresh && !inputs.demand->runId.isEmpty() && inputs.demand->runId != this->budget.run)
    {
      this->budget = {};
      this->budget.run = inputs.demand->runId;
      this->budget.recordingStart = inputs.demand->time;
      this->budget.status = "WAITING_FOR_GROUND";
    }
    if (inputs.land && inputs.land->time <= now + kClockEpsilon &&
        (this->budget.last || Fresh(now, inputs.land->time, kLandReportAge)))
      this->budget.Observe(*inputs.land, this->integrals,
          state.demandFresh && inputs.demand->runId == this->budget.run);
    const bool changed = this->lastStatus != state.status || this->lastLimited != state.limited ||
                         previousBudget != this->budget.status || this->previousState.source != state.source;
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
    this->previousState = state;
    this->previousBreak = inputs.discontinuity;
    this->previousLoad = state.loadPossible;
    this->previousIdle = state.knownIdle;
  }

  private: void IntegratePower(double start, double end, double covered, bool contiguous, const State &next)
  {
    const State &previous = this->previousState;
    if (covered > 0)
    {
      // Both pack endpoint pairs passed the voltage/current fit and the uncapped
      // command checks. Across a source switch the prior command owned this tick.
      const double p0 = *previous.terminalV * *previous.current;
      const double p1 = previous.source == next.source ? *next.terminalV * *next.current : p0;
      const double fraction = std::clamp(covered / (end - start), 0.0, 1.0);
      const double cutoff = std::min(end, start + covered);
      this->integrals.Append(start, cutoff, previous.source, p0, p0 + (p1 - p0) * fraction);
      start = cutoff;
    }
    // Only a fresh, explicitly identified whole-aircraft manager estimate may
    // cover the remainder (including clipping). It never supplies charge or V/I.
    if (contiguous && previous.managerPower && previous.managerExpires)
    {
      const double until = std::clamp(*previous.managerExpires, start, end);
      this->integrals.Append(start, until, Source::Manager, *previous.managerPower, *previous.managerPower);
      start = until;
    }
    this->integrals.Append(start, end, Source::Missing);
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
            kDemandTopic, [in](const gz::msgs::StringMsg &m) { in->OnDemand(m); }) ||
        !this->inputNode->Subscribe<gz::msgs::StringMsg>(
            kLandTopic, [in](const gz::msgs::StringMsg &m) { in->OnLand(m); }))
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

  private: void ResetMotion()
  {
    this->predictor.Reset();
    this->prediction = {};
    this->predictionAt.reset();
    this->predictionPayload.reset();
  }

  private: void UpdateMotion(const Inputs &in, double now, Ecm &ecm)
  {
    if (this->body == gz::sim::kNullEntity)
    {
      this->body = gz::sim::Model(this->entity).LinkByName(ecm, "base_link");
      if (this->body == gz::sim::kNullEntity)
        return;
      gz::sim::Link(this->body).EnableVelocityChecks(ecm);
    }
    const auto pose = gz::sim::worldPose(this->body, ecm);
    const auto velocity = gz::sim::Link(this->body).WorldLinearVelocity(ecm);
    const bool demandFresh = in.demand && Fresh(now, in.demand->time, this->fit.maxAge);
    const bool nativeFresh = in.land && Fresh(now, in.land->time, kLandReportAge);
    const auto landed = nativeFresh ? std::optional<bool>(in.land->landed) :
                        demandFresh ? in.demand->landed : std::nullopt;
    if (!landed || *landed || !velocity || !velocity->IsFinite() || !pose.IsFinite())
    {
      if (landed == true && pose.IsFinite() && velocity && velocity->IsFinite() &&
          std::abs(velocity->Z()) < 0.1)
        this->groundZ = pose.Pos().Z();
      this->ResetMotion();
      return;
    }
    this->predictor.Observe(now, std::hypot(velocity->X(), velocity->Y()), velocity->Z());
    if (!demandFresh || !in.demand->Transient() || !in.demand->payload || !this->groundZ)
    {
      this->prediction = {};
      this->predictionAt.reset();
      return;
    }
    // Keep causal speed history at physics cadence. Local-neighbour lookup
    // follows the native telemetry cadence; integrate the applied value each tick.
    if (this->predictionAt && this->predictionPayload == in.demand->payload &&
        now - *this->predictionAt < 0.05 - kClockEpsilon)
      return;
    const double tilt = std::acos(std::clamp(pose.Rot().ZAxis().Z(), -1.0, 1.0)) *
                        (180.0 / std::acos(-1.0));
    this->prediction = this->predictor.Predict(*in.demand->payload, pose.Pos().Z() - *this->groundZ, tilt);
    this->predictionAt = now;
    this->predictionPayload = in.demand->payload;
  }

  private: State Evaluate(const Inputs &in, double now) const
  {
    State s;
    s.capV = this->heldV;
    s.demandFresh = in.demand && Fresh(now, in.demand->time, this->fit.maxAge);
    s.rawFresh = in.raw && Fresh(now, in.raw->time, this->fit.maxAge);
    const Speeds raw = s.rawFresh ? in.raw->speed : Speeds{};
    const double peak = std::max(0.0, *std::max_element(raw.begin(), raw.end()));
    const bool nativeFresh = in.land && Fresh(now, in.land->time, kLandReportAge);
    // A native in-air arming observation supersedes the slower MAVLink
    // heartbeat. On the ground retain the original arming and motor start path.
    const bool nativeArmKnown = nativeFresh && !in.land->landed && in.land->armed.has_value();
    const bool nativeAir = nativeArmKnown && *in.land->armed;
    const bool armKnown = s.demandFresh && (nativeArmKnown || in.demand->armed.has_value());
    const bool armed = armKnown && (nativeArmKnown ? nativeAir : in.demand->armed.value_or(false));
    if (s.demandFresh)
    {
      s.effectivePower = in.demand->power;
      if (nativeAir && in.demand->armed == false)
        s.effectivePower.reset();  // The echo is a zero from an older disarmed heartbeat.
      if (nativeAir && !s.effectivePower && in.demand->phasePower)
        s.effectivePower = in.demand->phasePower;
      if (s.effectivePower && *s.effectivePower > 0)
        s.source = Source::Pack;
      const bool airborne = nativeFresh ? !in.land->landed : in.demand->landed == false;
      if (armed && airborne && s.rawFresh && peak > 0 && !s.effectivePower &&
          in.demand->Transient() && this->predictionAt && this->prediction.power &&
          this->predictionPayload == in.demand->payload && Fresh(now, *this->predictionAt, 0.05))
      {
        s.effectivePower = this->prediction.power;
        s.source = std::string_view(this->prediction.source) == "cmu_local" ? Source::Local : Source::Pooled;
      }
      if (in.demand->managerPower.value_or(0) > 0)
      {
        s.managerPower = in.demand->managerPower;
        s.managerExpires = in.demand->time + this->fit.maxAge;
      }
    }
    const bool powerKnown = s.effectivePower.has_value();
    const double power = s.effectivePower.value_or(-1.0);
    s.knownIdle = armKnown && !armed;
    s.loadPossible = !s.knownIdle && (armed || peak > 0);
    const double fitQ = std::clamp(this->q, this->fit.minQ, this->fit.maxQ);
    const double sourceV = this->fit.Voltage(0, fitQ) - this->polarFast - this->polarSlow;
    const double atMaxI = sourceV - this->fit.r * this->fit.maxI;
    const bool exhausted = this->q >= this->fit.maxQ;
    const bool highLoad = powerKnown && power > this->fit.maxI * atMaxI;
    const bool lowLoad = powerKnown && power < this->fit.minI * (sourceV - this->fit.r * this->fit.minI);
    if (!exhausted && power > 0 && !highLoad && !lowLoad)
    {
      const double voltage = 0.5 * (sourceV + std::sqrt(sourceV * sourceV - 4.0 * this->fit.r * power));
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

  private: void AdvancePolarization(double current, double duration)
  {
    if (!this->dynamicPack || duration <= 0)
      return;
    this->polarFast += -std::expm1(-duration / rcfit::tau1) * (rcfit::r1 * current - this->polarFast);
    this->polarSlow += -std::expm1(-duration / rcfit::tau2) * (rcfit::r2 * current - this->polarSlow);
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
    json["effective_demand_power_w"] = JsonNumber(s.effectivePower);
    json["effective_demand_source"] = kSourceNames[static_cast<std::size_t>(s.source)];
    json["calibration_id"] = transient::kCalibrationId;
    json["transient_time_s"] = JsonNumber(this->predictionAt);
    json["transient_regime"] = this->prediction.regime;
    json["transient_reason"] = this->predictionAt ? this->prediction.reason : "NO_CURRENT_TRANSIENT_ESTIMATE";
    json["transient_local_reason"] = this->prediction.localReason;
    json["transient_support_flights"] = this->prediction.supportFlights;
    json["transient_acceleration_mps2"] = JsonNumber(this->prediction.acceleration);
    json["transient_low_speed_dwell_s"] = JsonNumber(this->prediction.dwell);
    QJsonArray flags;
    if (!this->prediction.flags.empty())
      for (const auto &flag : QString::fromStdString(this->prediction.flags).split('|'))
        flags.append(flag);
    json["transient_transfer_flags"] = flags;
    const QString model = this->dynamicPack ? "CMU_LOADED_THEVENIN_2RC_V1" : "CMU_STATIC_LOADED_V1";
    auto budgetJson = this->budget.Json(this->integrals, now);
    budgetJson["battery_dynamics_model"] = model;
    budgetJson["initial_nominal_soc_percent"] = this->initialSoc;
    budgetJson["battery_parameter_sha256"] = this->dynamicPack ? QJsonValue(rcfit::hash) : QJsonValue(QJsonValue::Null);
    json["airborne_budget"] = budgetJson;
    json["battery_dynamics_model"] = model;
    json["battery_parameter_sha256"] = budgetJson["battery_parameter_sha256"];
    json["battery_dynamics_scope"] = "CMU_DISCHARGE_RELATIVE_TRANSFER_NOT_TB47D_CHARACTERIZATION";
    if (this->dynamicPack)
    {
      json["polarization_fast_v"] = this->polarFast;
      json["polarization_slow_v"] = this->polarSlow;
      json["reference_source_v"] = this->fit.Voltage(0, std::clamp(this->q, this->fit.minQ, this->fit.maxQ));
      json["polarization_load_history_complete"] = this->uncoveredSeconds == 0;
    }
    json["px4_land_time_s"] = JsonNumber(in.land ? Number(in.land->time) : std::nullopt);
    json["px4_land_transition_s"] = JsonNumber(in.land ? Number(in.land->transition) : std::nullopt);
    json["px4_land_sequence"] = JsonNumber(in.land ? Number(in.land->sequence) : std::nullopt);
    json["px4_landed"] = in.land ? QJsonValue(in.land->landed) : QJsonValue(QJsonValue::Null);
    json["armed"] = s.demandFresh && in.demand->armed ? QJsonValue(*in.demand->armed)
                                                       : QJsonValue(QJsonValue::Null);
    json["requested_load_voltage_v"] = JsonNumber(s.requestedV);
    json["requested_pack_current_a"] = JsonNumber(s.requestedI);
    json["voltage_v"] = JsonNumber(s.terminalV);
    json["current_a"] = JsonNumber(s.current);
    json["initial_nominal_soc_percent"] = this->initialSoc;
    json["initial_nameplate_discharge_offset_ah"] = this->initialQ;
    json["discharged_ah"] = this->q - this->initialQ;
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
  private: gz::sim::Entity body{gz::sim::kNullEntity};
  private: std::string modelName;
  private: Fit fit{};
  private: bool dynamicPack{true};
  private: double initialQ{0}, initialSoc{100};
  private: double polarFast{0}, polarSlow{0};
  private: transient::Predictor predictor;
  private: transient::Prediction prediction;
  private: Number predictionAt, predictionPayload, groundZ;
  private: Integrals integrals;
  private: AirborneBudget budget;
  private: State previousState;
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
