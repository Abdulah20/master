#include "uav_energy/MotorBatteryLoadSystem.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <gz/msgs/double.pb.h>
#include <gz/msgs/float_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/components/BatteryPowerLoad.hh>
#include <gz/sim/components/BatterySoC.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/WindMode.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/TopicUtils.hh>
#include <sdf/Element.hh>

#include "uav_energy/PowerModel.hpp"

namespace
{
template<typename T>
T ReadOrDefault(
    const std::shared_ptr<const sdf::Element> &_sdf,
    const std::string &_name,
    const T &_defaultValue)
{
  return _sdf->Get<T>(_name, _defaultValue).first;
}

void SetSimulationStamp(
    gz::msgs::Header *_header,
    const std::chrono::steady_clock::duration &_simTime)
{
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(_simTime);
  const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      _simTime - seconds);
  _header->mutable_stamp()->set_sec(seconds.count());
  _header->mutable_stamp()->set_nsec(static_cast<std::int32_t>(nanoseconds.count()));
}
}  // namespace

namespace uav_energy
{
void MotorBatteryLoadSystem::Configure(
    const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager &)
{
  this->modelEntity = _entity;
  this->model = gz::sim::Model(_entity);
  if (!this->model.Valid(_ecm))
  {
    throw std::runtime_error("MotorBatteryLoadSystem must be attached to a model");
  }
  this->modelName = this->model.Name(_ecm);

  this->batteryName = ReadOrDefault<std::string>(
      _sdf, "battery_name", this->batteryName);
  this->motorConstant = ReadOrDefault<double>(
      _sdf, "motor_constant", this->motorConstant);
  this->momentConstant = ReadOrDefault<double>(
      _sdf, "moment_constant", this->momentConstant);
  this->rotorVelocitySlowdown = ReadOrDefault<double>(
      _sdf, "rotor_velocity_slowdown", this->rotorVelocitySlowdown);
  this->propulsionEfficiency = ReadOrDefault<double>(
      _sdf, "propulsion_efficiency", this->propulsionEfficiency);
  this->auxiliaryPowerW = ReadOrDefault<double>(
      _sdf, "auxiliary_power_w", this->auxiliaryPowerW);
  this->maximumElectricalPowerW = ReadOrDefault<double>(
      _sdf, "maximum_electrical_power_w", this->maximumElectricalPowerW);
  const double publishRateHz = ReadOrDefault<double>(_sdf, "publish_rate_hz", 20.0);
  const bool enableWind = ReadOrDefault<bool>(_sdf, "enable_wind", false);

  if (this->batteryName.empty() || this->rotorVelocitySlowdown <= 0.0 ||
      !std::isfinite(this->rotorVelocitySlowdown) || publishRateHz <= 0.0 ||
      !std::isfinite(publishRateHz))
  {
    throw std::invalid_argument("Battery name, slowdown, or publish rate is invalid");
  }
  this->publishPeriodS = 1.0 / publishRateHz;

  for (std::size_t index = 0; index < this->jointNames.size(); ++index)
  {
    const std::string key = "joint_name_" + std::to_string(index);
    this->jointNames[index] = ReadOrDefault<std::string>(
        _sdf, key, this->jointNames[index]);
  }

  if (enableWind)
  {
    const std::vector<std::string> windLinkNames{
        "base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3"};
    for (const auto &linkName : windLinkNames)
    {
      const auto linkEntity = this->model.LinkByName(_ecm, linkName);
      if (linkEntity == gz::sim::kNullEntity)
      {
        throw std::runtime_error(
            "MotorBatteryLoadSystem cannot enable wind on link [" + linkName + "]");
      }
      if (_ecm.Component<gz::sim::components::WindMode>(linkEntity) == nullptr)
      {
        _ecm.CreateComponent(linkEntity, gz::sim::components::WindMode(true));
      }
      else
      {
        _ecm.SetComponentData<gz::sim::components::WindMode>(linkEntity, true);
      }
    }
  }

  this->jointEntities.reserve(this->jointNames.size());
  for (const auto &jointName : this->jointNames)
  {
    const auto jointEntity = this->model.JointByName(_ecm, jointName);
    if (jointEntity == gz::sim::kNullEntity)
    {
      throw std::runtime_error(
          "MotorBatteryLoadSystem cannot find joint [" + jointName + "]");
    }
    this->jointEntities.push_back(jointEntity);
    if (_ecm.Component<gz::sim::components::JointVelocity>(jointEntity) == nullptr)
    {
      _ecm.CreateComponent(
          jointEntity, gz::sim::components::JointVelocity(std::vector<double>{0.0}));
    }
  }

  // Validate all power-model parameters before simulation starts.
  static_cast<void>(ComputePower(
      std::vector<double>(this->jointEntities.size(), 0.0),
      this->motorConstant,
      this->momentConstant,
      this->propulsionEfficiency,
      this->auxiliaryPowerW,
      this->maximumElectricalPowerW));

  const std::string topicRoot = "/model/" + this->modelName + "/energy";
  this->rotorSpeedPublisher = this->node.Advertise<gz::msgs::Float_V>(
      gz::transport::TopicUtils::AsValidTopic(topicRoot + "/rotor_speed_rad_s"));
  this->shaftPowerPublisher = this->node.Advertise<gz::msgs::Double>(
      gz::transport::TopicUtils::AsValidTopic(topicRoot + "/shaft_power_w"));
  this->electricalLoadPublisher = this->node.Advertise<gz::msgs::Double>(
      gz::transport::TopicUtils::AsValidTopic(topicRoot + "/electrical_load_w"));
  this->cumulativeEnergyPublisher = this->node.Advertise<gz::msgs::Double>(
      gz::transport::TopicUtils::AsValidTopic(topicRoot + "/cumulative_energy_j"));

  gzmsg << "MotorBatteryLoadSystem configured for model [" << this->modelName
        << "] with fidelity [C_NOT_CALIBRATED]" << std::endl;
}

bool MotorBatteryLoadSystem::InitializeBatteryConsumer(
    gz::sim::EntityComponentManager &_ecm)
{
  _ecm.Each<
      gz::sim::components::BatterySoC,
      gz::sim::components::Name>(
      [&](const gz::sim::Entity &_entity,
          const gz::sim::components::BatterySoC *,
          const gz::sim::components::Name *_name) -> bool
      {
        // Gazebo requires battery names to be unique.  Searching by that
        // contract is more robust than assuming that an included model's
        // battery entity is parented directly to this plugin's model entity.
        if (_name->Data() == this->batteryName)
        {
          this->batteryEntity = _entity;
          return false;
        }
        return true;
      });

  if (this->batteryEntity == gz::sim::kNullEntity)
  {
    return false;
  }

  this->consumerEntity = _ecm.CreateEntity();
  _ecm.CreateComponent(
      this->consumerEntity,
      gz::sim::components::Name(this->modelName + "_motor_battery_load"));
  _ecm.CreateComponent(
      this->consumerEntity,
      gz::sim::components::BatteryPowerLoad(
          gz::sim::components::BatteryPowerLoadInfo{
              this->batteryEntity, this->auxiliaryPowerW}));
  _ecm.SetParentEntity(this->consumerEntity, this->batteryEntity);
  gzmsg << "MotorBatteryLoadSystem connected to battery [" << this->batteryName
        << "]" << std::endl;
  return true;
}

void MotorBatteryLoadSystem::PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm)
{
  if (_info.paused)
  {
    return;
  }
  if (this->consumerEntity == gz::sim::kNullEntity &&
      !this->InitializeBatteryConsumer(_ecm))
  {
    return;
  }

  std::vector<double> physicalRotorSpeedRadS;
  physicalRotorSpeedRadS.reserve(this->jointEntities.size());
  for (const auto jointEntity : this->jointEntities)
  {
    const auto *velocity =
        _ecm.Component<gz::sim::components::JointVelocity>(jointEntity);
    const double simulatedSpeed =
        (velocity != nullptr && !velocity->Data().empty()) ? velocity->Data()[0] : 0.0;
    physicalRotorSpeedRadS.push_back(
        simulatedSpeed * this->rotorVelocitySlowdown);
  }

  const PowerResult power = ComputePower(
      physicalRotorSpeedRadS,
      this->motorConstant,
      this->momentConstant,
      this->propulsionEfficiency,
      this->auxiliaryPowerW,
      this->maximumElectricalPowerW);

  _ecm.SetComponentData<gz::sim::components::BatteryPowerLoad>(
      this->consumerEntity,
      gz::sim::components::BatteryPowerLoadInfo{
          this->batteryEntity, power.electricalLoadW});

  const double deltaS = std::chrono::duration<double>(_info.dt).count();
  if (deltaS > 0.0 && std::isfinite(deltaS))
  {
    this->cumulativeEnergyJ += power.electricalLoadW * deltaS;
    this->timeSincePublishS += deltaS;
  }
  if (this->timeSincePublishS >= this->publishPeriodS)
  {
    this->PublishState(
        physicalRotorSpeedRadS,
        power.totalShaftPowerW,
        power.electricalLoadW,
        _info.simTime);
    this->timeSincePublishS = 0.0;
  }
}

void MotorBatteryLoadSystem::PublishState(
    const std::vector<double> &_physicalRotorSpeedRadS,
    const double _shaftPowerW,
    const double _electricalLoadW,
    const std::chrono::steady_clock::duration _simTime)
{
  gz::msgs::Float_V rotorMessage;
  SetSimulationStamp(rotorMessage.mutable_header(), _simTime);
  for (const double speed : _physicalRotorSpeedRadS)
  {
    rotorMessage.add_data(static_cast<float>(speed));
  }
  this->rotorSpeedPublisher.Publish(rotorMessage);

  gz::msgs::Double scalarMessage;
  SetSimulationStamp(scalarMessage.mutable_header(), _simTime);
  scalarMessage.set_data(_shaftPowerW);
  this->shaftPowerPublisher.Publish(scalarMessage);
  scalarMessage.set_data(_electricalLoadW);
  this->electricalLoadPublisher.Publish(scalarMessage);
  scalarMessage.set_data(this->cumulativeEnergyJ);
  this->cumulativeEnergyPublisher.Publish(scalarMessage);
}
}  // namespace uav_energy

GZ_ADD_PLUGIN(
    uav_energy::MotorBatteryLoadSystem,
    gz::sim::System,
    uav_energy::MotorBatteryLoadSystem::ISystemConfigure,
    uav_energy::MotorBatteryLoadSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
    uav_energy::MotorBatteryLoadSystem,
    "uav_energy::MotorBatteryLoadSystem")
