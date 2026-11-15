#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace uav_energy
{
class MotorBatteryLoadSystem final:
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager &_eventManager) final;

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) final;

private:
  bool InitializeBatteryConsumer(gz::sim::EntityComponentManager &_ecm);
  void PublishState(
      const std::vector<double> &_physicalRotorSpeedRadS,
      double _shaftPowerW,
      double _electricalLoadW,
      std::chrono::steady_clock::duration _simTime);

  gz::sim::Entity modelEntity{gz::sim::kNullEntity};
  gz::sim::Model model{gz::sim::kNullEntity};
  std::string modelName;
  std::string batteryName{"x500_energy_battery"};
  std::vector<std::string> jointNames{
      "rotor_0_joint", "rotor_1_joint", "rotor_2_joint", "rotor_3_joint"};
  std::vector<gz::sim::Entity> jointEntities;
  gz::sim::Entity batteryEntity{gz::sim::kNullEntity};
  gz::sim::Entity consumerEntity{gz::sim::kNullEntity};

  double motorConstant{8.54858e-06};
  double momentConstant{0.016};
  double rotorVelocitySlowdown{10.0};
  double propulsionEfficiency{0.70};
  double auxiliaryPowerW{10.0};
  double maximumElectricalPowerW{2000.0};
  double cumulativeEnergyJ{0.0};
  double publishPeriodS{0.05};
  double timeSincePublishS{0.0};

  gz::transport::Node node;
  gz::transport::Node::Publisher rotorSpeedPublisher;
  gz::transport::Node::Publisher shaftPowerPublisher;
  gz::transport::Node::Publisher electricalLoadPublisher;
  gz::transport::Node::Publisher cumulativeEnergyPublisher;
};
}  // namespace uav_energy
