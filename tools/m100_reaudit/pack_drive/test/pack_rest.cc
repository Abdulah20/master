// Exercise the actual message callbacks and branch update without starting Gazebo.
#include "../M100PackDrive.cc"

#include <cassert>

namespace
{
constexpr double step = 0.004;
constexpr double age = 0.2;

gz::msgs::StringMsg Message(QJsonObject object)
{
  gz::msgs::StringMsg msg;
  msg.set_data(QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString());
  return msg;
}

void Demand(m100::Inbox &inbox, double time, QJsonValue armed, QJsonValue power)
{
  inbox.OnDemand(Message({{"model", "M100"}, {"time_s", time},
                          {"armed", armed}, {"power_w", power}}));
}

void Motors(m100::Inbox &inbox, double first, int count = 4)
{
  gz::msgs::Actuators command;
  for (int i = 0; i < count; ++i)
    command.add_velocity(i == 0 ? first : 0.);
  inbox.OnRaw(command);
}

void Ground(m100::Inbox &inbox, double time, bool landed, bool armed, int sequence)
{
  inbox.OnLand(Message({{"model", "M100"}, {"time_s", time}, {"transition_s", time},
                        {"sequence", sequence}, {"landed", landed}, {"armed", armed}}));
}

struct FlightEnd
{
  m100::Inbox inbox{"M100", age, false};
  m100::PackRest rest;
  double time{0}, fast{0}, slow{0};

  void ZeroInputs()
  {
    Demand(inbox, time, false, 0.);
    Motors(inbox, 0.);
  }

  void Observe(bool contiguous = true)
  {
    rest.Observe(inbox.At(time), time, age, contiguous, step, fast, slow);
  }

  void Next(double elapsed = step)
  {
    time += elapsed;
    inbox.At(time);
  }

  void KnownRestStep()
  {
    Next();
    ZeroInputs();
    Observe();
  }
};

void Close(double left, double right)
{
  assert(std::abs(left - right) < 1e-12);
}
}

int main()
{
  FlightEnd f;
  // Known load, then rest. Both branches must follow their own analytic response.
  m100::AdvanceBranches(f.fast, f.slow, 20., 3.);
  const double fast0 = m100::rcfit::r1 * 20 * (1 - std::exp(-3. / m100::rcfit::tau1));
  const double slow0 = m100::rcfit::r2 * 20 * (1 - std::exp(-3. / m100::rcfit::tau2));
  Close(f.fast, fast0);
  Close(f.slow, slow0);
  f.inbox.At(0.);
  f.ZeroInputs();
  Ground(f.inbox, 0., true, false, 1);
  f.Observe(false);
  for (int i = 0; i < 500; ++i)
    f.KnownRestStep();
  Close(f.rest.seconds, 2.);
  Close(f.fast, fast0 * std::exp(-2. / m100::rcfit::tau1));
  Close(f.slow, slow0 * std::exp(-2. / m100::rcfit::tau2));
  assert(f.inbox.At(f.time).discontinuity >= 1000);  // Loaded history still breaks on zeros.

  // An intervening contradictory observation cannot disappear when both
  // callbacks arrive before the next physics step and the latest is zero.
  auto hidden = [&](auto interrupt) {
    const double before = f.rest.seconds;
    const double fast = f.fast, slow = f.slow;
    f.Next();
    interrupt();
    f.ZeroInputs();
    f.Observe();
    Close(f.rest.seconds, before);
    Close(f.fast, fast);
    Close(f.slow, slow);
    f.KnownRestStep();
    Close(f.rest.seconds, before + step);
  };
  hidden([&] { Demand(f.inbox, f.time-step/2, true, 0.); });
  hidden([&] { Demand(f.inbox, f.time-step/2, QJsonValue::Null, 0.); });
  hidden([&] { Demand(f.inbox, f.time-step/2, false, QJsonValue::Null); });
  hidden([&] { Demand(f.inbox, f.time-step/2, false, 20.); });
  hidden([&] { Motors(f.inbox, 15.); });
  hidden([&] { Motors(f.inbox, 0., 3); });
  hidden([&] {
    Ground(f.inbox, f.time-step/2, false, true, 2);
    Ground(f.inbox, f.time, true, false, 3);
  });

  // Keep fresh battery declarations while motors stop reporting. Recovery
  // may start a new rest only after two eligible observations.
  const double before = f.rest.seconds;
  f.Next(age + step);
  Demand(f.inbox, f.time, false, 0.);
  f.Observe(false);
  f.KnownRestStep();
  Close(f.rest.seconds, before);
  f.KnownRestStep();
  Close(f.rest.seconds, before + step);
  f.Next();
  f.ZeroInputs();
  f.Observe(false);  // paused, rewound or skipped physics step
  Close(f.rest.seconds, before + step);
}
