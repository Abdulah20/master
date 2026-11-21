// Headless Gazebo Harmonic runner. Gazebo's Lee controller drives the four
// ordinary MulticopterMotorModel systems on the M100 research airframe.
#include <gz/msgs/twist.pb.h>
#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using gz::math::Vector3d;
namespace sim = gz::sim;
using SimSeconds = std::chrono::duration<double>;

struct Conditions {
  double height, speed, route, hover, windEast, windNorth, datumC, dragArea;
};

class Mission final : public sim::System, public sim::ISystemPreUpdate,
                      public sim::ISystemPostUpdate {
 public:
  Mission(Conditions inputs, const std::string &output)
      : c(inputs), trace(output),
        publisher(node.Advertise<gz::msgs::Twist>("/research_m100/velocity")) {
    trace.exceptions(std::ios::badbit | std::ios::failbit);
    trace << "time_s,phase,x_m,y_m,z_m,ve_mps,vn_mps,vu_mps,roll_rad,pitch_rad,yaw_rad\n";
    trace << std::setprecision(13);
  }

  void PreUpdate(const sim::UpdateInfo &info, sim::EntityComponentManager &ecm) override {
    if (done || info.paused) return;
    if (body == sim::kNullEntity) {
      auto model = ecm.EntityByComponents(sim::components::Name("research_m100"), sim::components::Model());
      if (model == sim::kNullEntity) return;
      body = sim::Model(model).LinkByName(ecm, "base_link");
      if (body == sim::kNullEntity) throw std::runtime_error("missing base link");
      sim::Link(body).EnableVelocityChecks(ecm, true);
    }
    auto link = sim::Link(body);
    auto velocity = link.WorldLinearVelocity(ecm).value_or(Vector3d::Zero);
    auto position = sim::worldPose(body, ecm);
    const double t = SimSeconds(info.simTime).count() - SimSeconds(info.dt).count();
    const auto p = position.Pos();
    const double z = p.Z()-groundZ;
    // Same dry-atmosphere equation and horizontal drag area as the source
    // runtime. Only this term applies an external horizontal wind force.
    const double kelvin = c.datumC + 273.15;
    const double local = kelvin-0.0065*std::max(0.0,z);
    const double density = 101325*std::pow(local/kelvin,9.80665/(287.05287*0.0065))/(287.05287*local);
    Vector3d relative{velocity.X()-c.windEast, velocity.Y()-c.windNorth, 0};
    link.AddWorldForce(ecm, -0.5*density*c.dragArea*relative.Length()*relative);
    // Use an outer velocity integrator for stationary drag rejection. The
    // installed controller still computes attitude and individual rotor speeds.
    if (t + 1e-8 < nextCommand) return;
    nextCommand += 0.04;
    double targetEast = 0, targetNorth = 0, targetUp = 0;
    double hError = c.height-z;
    auto holdHeight = [&]() {return std::clamp(1.2*hError,-2.0,2.5);};
    auto alongTarget = [&](double distance, double limit) {
      return std::clamp(0.8*distance,-std::min(limit,std::sqrt(4.0*std::abs(distance))),
                                       std::min(limit,std::sqrt(4.0*std::abs(distance))));
    };
    std::string oldPhase=phase;
    if (phase=="GROUND") {
      groundZ=p.Z();
      if (t>=1.0) phase="CLIMB";
    } else if (phase=="CLIMB") {
      targetUp=holdHeight(); targetEast=alongTarget(-p.X(),3);
      if (std::abs(hError)<0.25 && std::abs(velocity.Z())<0.35) phase="STABILIZING";
    } else if (phase=="STABILIZING") {
      targetUp=holdHeight(); targetEast=alongTarget(-p.X(),3);
      if (t-phaseTime>=3.0) phase=c.speed>0 ? "ACCELERATING" : "HOVER";
    } else if (phase=="HOVER") {
      targetUp=holdHeight(); targetEast=alongTarget(-p.X(),3);
      if (t-phaseTime >= c.hover) phase="LANDING";
    } else if (phase=="ACCELERATING" || phase=="CRUISE") {
      targetEast=c.speed; targetUp=holdHeight();
      if (phase=="ACCELERATING") {
        const bool atTarget=std::abs(velocity.X()-c.speed)<=std::max(0.3,0.05*c.speed)
            && std::abs(hError)<=0.5 && std::abs(velocity.Y())<=0.3;
        if (atTarget) {
          if (readySince<0) readySince=t;
          if (t-readySince>=2.0) { startX=p.X(); phase="CRUISE"; }
        } else {
          readySince=-1;
        }
      } else if (p.X()-startX >= c.route) {
        phase="RETURN_BRAKING";
      }
    } else if (phase=="RETURN_BRAKING") {
      targetUp=holdHeight();
      if (std::hypot(velocity.X(),velocity.Y())<0.4) phase="RETURNING";
    } else if (phase=="RETURNING") {
      targetUp=holdHeight();
      targetEast=alongTarget(-p.X(),std::min(10.0,c.speed));
      if (std::hypot(p.X(),p.Y())<0.6 && velocity.Length()<0.65) phase="LANDING";
    } else if (phase=="LANDING") {
      targetEast=alongTarget(-p.X(),3);
      targetUp= -std::clamp(z*0.9,0.25,1.5);
      if (z<0.045 && std::abs(velocity.Z())<0.12) {
        phase="LANDED"; result="completed"; done=true;
      }
    }
    targetNorth=alongTarget(-p.Y(),3);
    // The elapsed counter still belongs to the previous phase at a boundary.
    // Assess the current phase's timeout only while it is still active.
    if (phase==oldPhase && phase!="HOVER" &&
        t-phaseTime>std::max(100.,2.0*std::max(c.height/1.5,c.route/std::max(1.0,std::min(10.0,c.speed)))+80.0)) {
      result="phase_timeout_"+phase; done=true;
    }
    if (std::abs(position.Rot().Roll())>1.35 || std::abs(position.Rot().Pitch())>1.35 ||
        !std::isfinite(p.X()+p.Y()+p.Z()) || z < -1) {
      result="attitude_or_position_failure"; done=true;
    }
    if (phase!=oldPhase) {
      // Log both one-sided phase values at the identical physical boundary.
      // PostUpdate will log the next physics state with the new label.
      write(t,oldPhase,position,velocity);
      write(t,phase,position,velocity);
      phaseTime=t;
    }
    if (phase=="GROUND") return;  // no commands until the landing rails settle
    if (!publisher.HasConnections() && t>3.0) {
      result="gazebo_velocity_controller_not_connected"; done=true; return;
    }
    Vector3d target{targetEast,targetNorth,targetUp};
    Vector3d error=target-velocity;
    for (int axis=0; axis<3; ++axis) {
      // Do not integrate large velocity errors while the acceleration limit
      // is active. This prevents a known setpoint-acquisition overshoot.
      if (std::abs(error[axis])<1.0)
        velocityIntegral[axis]=std::clamp(velocityIntegral[axis]+0.04*error[axis],-2.5,2.5);
    }
    if (phase=="LANDING" || phase=="LANDED") velocityIntegral.Z(0);
    target += 0.35*velocityIntegral;
    // LeeVelocityController accepts body-frame linear commands.
    Vector3d localCmd=position.Rot().RotateVectorReverse(target);
    gz::msgs::Twist msg;
    msg.mutable_linear()->set_x(localCmd.X());
    msg.mutable_linear()->set_y(localCmd.Y());
    msg.mutable_linear()->set_z(localCmd.Z());
    publisher.Publish(msg);
  }

  void PostUpdate(const sim::UpdateInfo &info, const sim::EntityComponentManager &ecm) override {
    if (body==sim::kNullEntity || info.paused || done) return;
    const double t=SimSeconds(info.simTime).count();
    if (t+1e-8 < nextSample) return;
    nextSample += 0.2;
    auto v=sim::Link(body).WorldLinearVelocity(ecm);
    if (v) write(t,phase,sim::worldPose(body,ecm),*v);
  }

  void write(double t,const std::string &label,const gz::math::Pose3d &pose,const Vector3d &vel) {
    // At a scheduled sample/edge collision keep one record per (time,phase).
    if (t+1e-9 < lastTime) throw std::runtime_error("clock ran backward");
    if (std::abs(t-lastTime)<1e-8 && label==lastPhase) return;
    auto p=pose.Pos(); auto r=pose.Rot();
    trace<<t<<','<<label<<','<<p.X()<<','<<p.Y()<<','<<p.Z()<<','<<vel.X()<<','<<vel.Y()<<','<<vel.Z()
         <<','<<r.Roll()<<','<<r.Pitch()<<','<<r.Yaw()<<'\n';
    lastTime=t; lastPhase=label;
  }

  bool done=false;
  std::string result="mission_time_limit";

 private:
  Conditions c;
  std::ofstream trace;
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  sim::Entity body=sim::kNullEntity;
  std::string phase="GROUND",lastPhase;
  double groundZ=0.224,phaseTime=0,startX=0,readySince=-1,nextCommand=0,nextSample=0,lastTime=-1;
  Vector3d velocityIntegral;
};

int main(int argc, char** argv) {
  if (argc!=12) { std::cerr<<"usage: engine world.sdf output.csv dt H speed route hover windE windN T CdA\n"; return 2; }
  sim::ServerConfig config;
  config.SetSdfFile(argv[1]);
  config.SetPhysicsEngine("gz-physics-dartsim-plugin");
  config.SetUpdateRate(0.0);
  config.SetSeed(0);
  sim::Server server(config);
  const double dt=std::stod(argv[3]);
  Conditions c{std::stod(argv[4]),std::stod(argv[5]),std::stod(argv[6]),std::stod(argv[7]),
               std::stod(argv[8]),std::stod(argv[9]),std::stod(argv[10]),std::stod(argv[11])};
  auto mission=std::make_shared<Mission>(c,argv[2]);
  if (!server.AddSystem(mission).value_or(false)) return 3;
  const size_t block=static_cast<size_t>(std::round(1.0/dt));
  const size_t limit=static_cast<size_t>(300+c.height*2.0+c.route*3/std::max(1.0,std::min(10.0,c.speed))+c.hover);
  for (size_t k=0;k<limit && !mission->done;++k) {
    if (!server.Run(true,block,false)) return 4;
  }
  std::cout<<"RESULT "<<mission->result<<"\n";
  return 0;
}
