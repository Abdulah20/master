"""Run the research M100 with the native Gazebo Lee velocity controller and DART."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

from .historical import SUMMARY_NAME, original_study, read_study
from .physics import Model, PORTABLE_PX4, PORTABLE_STUDY, wind_enu

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/m100_reaudit/gazebo_batch"
ENGINE = Path(__file__).with_name("gazebo_batch_engine.cc")
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "m100-research-gazebo"
DEFAULT_PX4 = PORTABLE_PX4  # The historical Lee controller writes its own Actuators component.
INPUTS = ["payload_kg", "cruise_speed_mps", "wind_speed_mps", "wind_direction_toward_deg_ENU",
          "altitude_m", "temperature_c", "measurement_distance_m", "hover_duration_s"]
# Fixed before accepting simulations. Same speed/height tolerance as the native audit.
BASE_STEP = 0.008
FINE_STEP = BASE_STEP / 2
BUDGET_ATOL_WH = 1e-7
DURATION_ATOL_S = 1e-7
CONVERGENCE_POWER_W = 2.0
CONVERGENCE_ENERGY_REL = 0.04
FIRST_RELEASE = "8101e1263fe1f5969909605cb3b7a962c1c4311b"
FIRST_MANIFEST_SHA256 = "78569dba04fba113b8047ae5f5120adfb72c3e4a71a5735a82e3835a36290c5b"


def research_study(explicit=None):
    try:
        return original_study(explicit)
    except FileNotFoundError:
        if explicit or not (PORTABLE_STUDY/'protocol.json').is_file():
            raise
        return PORTABLE_STUDY


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def add(parent, tag, value=None, **attrs):
    child = ET.SubElement(parent, tag, attrs)
    if value is not None:
        child.text = str(value)
    return child


def make_world(assets, payload_kg, step, dest):
    base = assets / "m100_cmu_v2/model.sdf"
    aircraft = ET.parse(base).getroot().find("model")
    if aircraft.find("plugin[@name='m100::PackDrive']") is not None:
        raise ValueError("The Lee campaign writes rotor commands directly. For a voltage-coupled flight "
                         "use the native PX4 manager. For the published Lee baseline select runtime_snapshot/px4.")
    files = [base]
    if payload_kg:
        wrapper = assets / f"m100_cmu_v2_payload_{round(100*payload_kg):03d}/model.sdf"
        extra = ET.parse(wrapper).getroot().find('model')
        for elem in extra:
            if elem.tag in ('link', 'joint'):
                aircraft.append(elem)
        files.append(wrapper)
    # Preserve all motor, mass, inertia, joint, and contact parameters; omit only
    # visuals and sensors which this ground-truth controller does not use.
    for link in aircraft.findall("link"):
        for elem in list(link):
            if elem.tag in ("visual", "sensor"):
                link.remove(elem)
    motors = []
    for elem in aircraft.findall("plugin"):
        if elem.get("name") == "gz::sim::systems::MulticopterMotorModel":
            motors.append(elem)
        else:
            aircraft.remove(elem)
    if len(motors) != 4:
        raise ValueError("expected four source rotor models")
    physical = sum(float(link.findtext('inertial/mass')) for link in aircraft.findall('link'))
    if not math.isclose(physical, 3.680+payload_kg, abs_tol=1e-9):
        raise ValueError("SDF research airframe and payload mass do not match inventory")
    aircraft.set("name", "research_m100")
    root = ET.Element("sdf", version="1.9")
    world = add(root, "world", name="research")
    add(world, "gravity", "0 0 -9.80665")
    phy = add(world, "physics", name="dart_step", type="ignored")
    add(phy, "max_step_size", step)
    add(phy, "real_time_factor", 0)
    add(world, "plugin", filename="gz-sim-physics-system", name="gz::sim::systems::Physics")
    ground = add(world, "model", name="ground")
    add(ground, "static", "true")
    collider = add(add(ground, "link", name="floor"), "collision", name="floor")
    plane = add(add(collider, "geometry"), "plane")
    add(plane, "normal", "0 0 1")
    add(plane, "size", "5000 5000")
    world.append(aircraft)
    ctrl = add(aircraft, "plugin", filename="gz-sim-multicopter-control-system",
               name="gz::sim::systems::MulticopterVelocityControl")
    for key, value in {"robotNamespace": "research_m100", "commandSubTopic": "velocity",
                       "comLinkName": "base_link", "velocityGain": f"{2.5*physical} {2.5*physical} {3.0*physical}",
                       "attitudeGain": "3 3 0.8", "angularRateGain": "0.8 0.8 0.35",
                       "maximumLinearAcceleration": "3 3 3"}.items():
        add(ctrl, key, value)
    rotors = add(ctrl, "rotorConfiguration")
    for m in sorted(motors, key=lambda x: int(x.findtext("motorNumber"))):
        rotor = add(rotors, "rotor")
        for dest_tag, src_tag in [("jointName","jointName"), ("forceConstant", "motorConstant"), ("momentConstant", "momentConstant")]:
            add(rotor,dest_tag,m.findtext(src_tag))
        add(rotor, "direction", 1 if m.findtext("turningDirection")=="ccw" else -1)
    dest.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {str(p.relative_to(assets)): sha(p) for p in files}


def compile_engine(dest):
    flags = subprocess.check_output(['pkg-config','--cflags','--libs','gz-sim8','gz-transport13','gz-msgs10'],text=True).split()
    subprocess.run(['g++','-std=c++17','-O2',str(ENGINE),'-o',str(dest),*flags],check=True)


def scenarios(study):
    protocol, _, _ = read_study(study)
    original = protocol['cases']
    rows = [{'case_id': row['case_id'], 'cohort': 'original', 'design': 'historical',
             **{k: float(row[k]) for k in INPUTS}} for row in original]
    if len(rows) != 69:
        raise ValueError("original protocol count changed")
    center = dict(zip(INPUTS, (0.25, 6., 0., 0., 50., 15., 125., 20.), strict=True))
    added = set()

    def design(family, **overrides):
        case = {**center, **overrides}
        key = tuple(case[k] for k in INPUTS)
        if key in added:
            return
        added.add(key)
        rows.append({'case_id': f'exp_{len(added):03d}', 'cohort': 'expanded', 'design': family, **case})

    for name, levels in [('temperature_c', (0,15,25,35)), ('altitude_m',(25,50,75,100,200)),
                         ('cruise_speed_mps',(4,6,8,10,12,17,20)), ('payload_kg',(0,.25,.5,.75,1,1.25)),
                         ('measurement_distance_m',(50,125,250,500))]:
        for value in levels:
            design('matched_'+name, **{name: value})
    for speed in (4,6,8,10,12):
        design('matched_wind',cruise_speed_mps=speed)
        for wind_speed in (3.,6.):
            for direction in (0.,90.,180.):
                design('matched_wind',cruise_speed_mps=speed,wind_speed_mps=wind_speed,wind_direction_toward_deg_ENU=direction)
    for h in (3.,5.,25.):
        for temp in (0.,15.,25.,35.):
            design('hover', cruise_speed_mps=0.,payload_kg=0.,altitude_m=h,temperature_c=temp,
                   measurement_distance_m=0.,hover_duration_s=15.)
    # A reproducible set of distinct operating points. Randomness belongs to
    # design selection only. Deterministic simulated flights are not replicated as data.
    rng = np.random.default_rng(20261121)
    while len(added) < 104:
        wind = float(rng.choice([0,3,6,9]))
        design('sampled_operating_points',
               payload_kg=float(rng.choice([0,.25,.5,.75,1,1.25])),
               cruise_speed_mps=float(rng.choice([4,6,8,10,12,15])),
               wind_speed_mps=wind,wind_direction_toward_deg_ENU=float(rng.choice([0,90,180]) if wind else 0),
               altitude_m=float(rng.choice([25,50,75,100,200])),temperature_c=float(rng.choice([0,15,25,35])),
               measurement_distance_m=float(rng.choice([50,125,250,500])))
    design('research_outside_stock_wind',wind_speed_mps=12,wind_direction_toward_deg_ENU=180)
    design('short_route',cruise_speed_mps=10,measurement_distance_m=25)
    design('nominal_endurance_boundary',cruise_speed_mps=0,altitude_m=100,payload_kg=1.25,measurement_distance_m=0,hover_duration_s=600)
    design('airframe_limit',payload_kg=2.0)
    design('command_limit',cruise_speed_mps=23)
    design('command_limit',altitude_m=250)
    design('command_limit',measurement_distance_m=0)
    design('command_limit',temperature_c=65)
    assert len(rows)==181
    return rows


def assess(model, case):
    return model.assess(case['payload_kg'],case['cruise_speed_mps'],case['wind_speed_mps'],case['altitude_m'],
                        case['temperature_c'],case['measurement_distance_m'],case['hover_duration_s'],
                        case['wind_direction_toward_deg_ENU'])


def run_dart(executable, world, case, step, raw, drag_area):
    east, north, _ = wind_enu(case['wind_speed_mps'],case['wind_direction_toward_deg_ENU'])
    args = [step,case['altitude_m'],case['cruise_speed_mps'],case['measurement_distance_m'],
            case['hover_duration_s'],east,north,case['temperature_c'],drag_area]
    attempt = subprocess.run([str(executable),str(world),str(raw),*map(str,args)],
                             env={**os.environ, 'GZ_PARTITION': 'm100research_'+os.urandom(10).hex()},
                             capture_output=True,text=True,timeout=240)
    marker = [line[7:] for line in attempt.stdout.splitlines() if line.startswith('RESULT ')]
    if attempt.returncode or len(marker)!=1 or not raw.is_file():
        raise RuntimeError(f"Gazebo exit {attempt.returncode}, stdout={attempt.stdout[-300:]}, stderr={attempt.stderr[-700:]}")
    return marker[0]


def scheduled_runs(cases):
    fine_cases = {'case_001','case_007','exp_001','exp_012'}
    # Include an actual windy and short-hover event independently of list ordering.
    fine_cases.add(next(c['case_id'] for c in cases if c['design']=='matched_wind' and c['wind_speed_mps']>0))
    fine_cases.add(next(c['case_id'] for c in cases if c['design']=='hover'))
    return [(c,BASE_STEP) for c in cases]+[(c,FINE_STEP) for c in cases if c['case_id'] in fine_cases]


def truth_stage(study, assets, cache, jobs, run_ids=None):
    model = Model.load()
    wind = yaml.safe_load((model.runtime_root/'ros2_ws/src/drone_nav/config/m100_wind_simulation_v1.yaml').read_text())
    drag = wind['horizontal_body_drag']; drag_area=drag['drag_coefficient']*drag['reference_area_m2']
    cases = scenarios(study)
    jobspec = scheduled_runs(cases)
    envs = {}
    cache.mkdir(parents=True,exist_ok=True)
    binary=cache/'engine'
    signature = sha(ENGINE)+subprocess.check_output(['pkg-config','--modversion','gz-sim8'],text=True).strip()
    sig_file=cache/'engine.signature'
    if not binary.is_file() or not sig_file.is_file() or sig_file.read_text()!=signature:
        compile_engine(binary)
        sig_file.write_text(signature)
    allowed=[(c,dt) for c,dt in jobspec if assess(model,c).runtime_eligible]
    for c,dt in allowed:
        key=(c['payload_kg'],dt)
        if key in envs: continue
        file=cache/f'm100_{round(100*key[0]):03d}_{dt}.sdf'
        sources=make_world(assets,key[0],dt,file)
        envs[key]=(file,sha(file),sources)
    environment_versions = {name: subprocess.check_output(['pkg-config','--modversion',name],text=True).strip()
                            for name in ('gz-sim8','gz-physics7-dartsim-plugin','gz-sim8-multicopter-control-system')}
    def fly(item):
        c,dt=item
        tag=c['case_id']+('_fine' if dt==FINE_STEP else '')
        raw=cache/(tag+'.csv'); receipt=cache/(tag+'.json')
        file,sdf_sha,asset_sha=envs[(c['payload_kg'],dt)]
        # Physics output can be reused after reporting-code changes. Include the
        # engine source, exact SDF, inputs and transfer coefficient in the key.
        key=hashlib.sha256(json.dumps([sha(ENGINE),sdf_sha,c,dt,drag_area],sort_keys=True).encode()).hexdigest()
        if receipt.is_file() and raw.is_file():
            prior=json.loads(receipt.read_text())
            if (prior['key']==key and prior['truth_sha256']==sha(raw)
                and prior.get('versions')==environment_versions):
                return
        temp=raw.with_suffix('.partial')
        why=run_dart(binary,file,c,dt,temp,drag_area)
        temp.replace(raw)
        record=dict(key=key,reason=why,case_id=c['case_id'],step_s=dt,world_sha256=sdf_sha,
                    versions=environment_versions,model_source_sha256=asset_sha,truth_sha256=sha(raw))
        interim=receipt.with_suffix('.tmp')
        interim.write_text(json.dumps(record,sort_keys=True)); interim.replace(receipt)
        print(tag,why,flush=True)
    live=[(case,dt) for case,dt in allowed
          if run_ids is None or case['case_id']+('_fine' if dt==FINE_STEP else '') in run_ids]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(fly,live))
    return jobspec


class RuntimePower:
    """Evaluate the package used by the regular M100 manager on Gazebo truth."""
    def __init__(self, model, case):
        from drone_nav.m100_stable_cruise_speed import StableCruiseSpeedFit
        from drone_nav.m100_transient_power import TransientPowerEstimator, TransientPowerFit
        from drone_nav.m100_wind_simulation import WindSimulation
        self.model, self.case = model, case
        params = model.settings['energy']
        root = model.runtime_root/'ros2_ws/src/drone_nav/config'
        fit = TransientPowerFit.load(root/params['transient_config'])
        self.motion = TransientPowerEstimator(fit,.5)
        self.wind = WindSimulation.load(root/params['wind_config'])
        self.vector = wind_enu(case['wind_speed_mps'],case['wind_direction_toward_deg_ENU'])
        self.config = model.contract.build_config(case['temperature_c'],case['payload_kg'],case['wind_speed_mps'],
                        case['wind_direction_toward_deg_ENU'],case['cruise_speed_mps'],case['altitude_m'],
                        case['measurement_distance_m'],case['hover_duration_s'])
        self.speed_delta = StableCruiseSpeedFit.load(root/params['stable_cruise_config']).adjustment_w(self.config)
        self.hover_mean = model.contract.m100_v2_energy_assessment(self.config, 'HOVER', measurement_active=True).use_limited_hover_mean

    def evaluate(self, phase, time, v, density):
        # The acceleration window precedes the measurement; the runtime's
        # kinematic stabilization proxy supplies its electrical estimate.
        energy_phase = 'STABILIZING' if phase=='ACCELERATING' else phase
        p = self.case['payload_kg']
        transient = self.motion.estimate(phase=energy_phase,timestamp_s=time,velocity_enu_mps=v,payload_kg=p,density_kg_m3=density)
        fixed = self.model.energy.estimate_power(energy_phase,p,density,
                        limited_hover_eligible=self.hover_mean and energy_phase=='HOVER',
                        stable_cruise_speed_adjustment_w=self.speed_delta if phase=='CRUISE' else None)
        estimate = transient if transient is not None and not (self.hover_mean and energy_phase=='HOVER') else fixed
        if estimate.power_w is None:
            return np.nan, np.nan, 'unavailable'
        delta=0.
        if self.case['wind_speed_mps']:
            delta=self.wind.power_delta_w(self.motion.fit,payload_kg=p,density_kg_m3=density,
                                         velocity_enu_mps=v,wind_enu_mps=self.vector)
        power=estimate.power_w+delta
        if not (math.isfinite(power) and power>0):
            return np.nan,np.nan,'unavailable'
        return power,delta,estimate.model_id


def csv_gzip(df, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix('.partial')
    with partial.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=6,mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='ascii',newline='') as stream:
            df.to_csv(stream,index=False,float_format='%.11g',lineterminator='\n')
    partial.replace(path)


def integrate(df):
    t=df.time_s.to_numpy()
    phase=df.phase.to_numpy()
    p=df.power_W.to_numpy()
    dt=np.diff(t)
    if not (np.isfinite(dt).all() and (dt>=-1e-9).all() and (dt<=.5+1e-8).all()):
        raise ValueError('unexpected truth recording clock interval')
    if np.any((phase[1:]!=phase[:-1]) & (dt>1e-8)):
        raise ValueError('unpaired phase boundary')
    adjacent=(phase[1:]==phase[:-1]) & (dt>1e-9)
    valid=adjacent & np.isfinite(p[1:]) & np.isfinite(p[:-1])
    de=np.zeros(len(df))
    de[1:][valid] = .5*(p[:-1][valid]+p[1:][valid])*dt[valid]/3600
    # Missing estimates accumulate nothing but leave an explicit coverage mask.
    return np.cumsum(de), valid, dt


def summarize_truth(raw, case, run_id, result, model, out, step, source_meta):
    from .physics import relative_wind
    flags=assess(model,case)
    f=pd.read_csv(raw)
    if not len(f) or not np.isfinite(f.select_dtypes('number').to_numpy()).all():
        raise ValueError(f'{run_id}: empty/nonfinite physics truth')
    home_rows=f[f.phase=='GROUND']
    if home_rows.empty:
        raise ValueError(f'{run_id}: missing ground reference')
    settled_z=float(home_rows.iloc[-1].z_m)
    f['height_agl_m']=f.z_m-settled_z
    f['altitude_msl_m']=f.height_agl_m+model.settings['wind']['world_ground_msl_m']
    pmodel=RuntimePower(model,case)
    f['wind_e_mps'],f['wind_n_mps'],f['wind_u_mps']=pmodel.vector
    f['air_e_mps']=f.ve_mps-pmodel.vector[0]; f['air_n_mps']=f.vn_mps-pmodel.vector[1]
    f['air_u_mps']=f.vu_mps-pmodel.vector[2]
    f['mass_kg']=flags.physical_mass_kg
    # Tiny contact penetration belongs to the ground datum. The geometric
    # heights stay signed; the air-property query uses the boundary value.
    atmos=[model.atmosphere(case['temperature_c'],max(0.,h)) for h in f.height_agl_m]
    f['rho_kg_m3']=[a.density_kg_m3 for a in atmos]
    f['air_pressure_Pa']=[a.pressure_pa for a in atmos]
    f['local_air_temp_C']=[a.temperature_k_at_altitude-273.15 for a in atmos]
    estimates=[pmodel.evaluate(r.phase,r.time_s,(r.ve_mps,r.vn_mps,r.vu_mps),r.rho_kg_m3) for r in f.itertuples()]
    f['power_W']=[e[0] for e in estimates];f['wind_delta_W']=[e[1] for e in estimates]
    f['power_model']=[e[2] for e in estimates]
    f['voltage_V']=np.nan;f['current_A']=np.nan
    f['cumulative_supported_Wh'],valid,dt=integrate(f)
    nominal=model.settings['battery']['nominal_energy_wh']
    reserve=nominal*model.settings['battery']['planning_reserve_fraction']
    f['nominal_remaining_Wh']=nominal-f.cumulative_supported_Wh
    f['battery_state']=np.select([f.nominal_remaining_Wh<=0,f.nominal_remaining_Wh<reserve],
                                  ['NAMEPLATE_EXHAUSTED','BELOW_NOMINAL_RESERVE'],default='NOMINAL_ACCOUNTING')
    f['incoming_interval_has_power']=np.r_[False,valid]
    reasons=[]
    exhaustion=f.index[f.nominal_remaining_Wh<=0]
    exhaustion_time=np.nan
    if len(exhaustion):
        exhaustion_time=float(f.loc[exhaustion[0],'time_s'])
        # Beyond this event DART continues to provide a counterfactual kinematic
        # trajectory. The modeled battery budget cannot support the remaining task.
        reasons.append('nameplate_exhausted_before_mission_end')
    elif min(f.nominal_remaining_Wh)<reserve:
        reasons.append('nominal_planning_reserve_crossed')
    if result!='completed':
        reasons.append(result)
    seg=[]
    geo=np.sqrt(np.diff(f.x_m)**2+np.diff(f.y_m)**2)
    speed=np.hypot(f.ve_mps,f.vn_mps).to_numpy()
    duration_covered=sum(dt[valid]); seen_total=f.cumulative_supported_Wh.iat[-1]
    for label,g in f.groupby('phase',sort=False):
        if not np.all(np.diff(g.index)==1):
            raise ValueError(f'{run_id}: unexpected phase recurrence')
        first,last=int(g.index[0]),int(g.index[-1])
        covered=float(sum(dt[first:last][valid[first:last]]))
        energy=g.cumulative_supported_Wh.iat[-1]-g.cumulative_supported_Wh.iat[0]
        obs_distance=float(sum(geo[first:last]))
        span=g.time_s.iat[-1]-g.time_s.iat[0]
        seg.append(dict(run_id=run_id,case_id=case['case_id'],phase=label,start_s=g.time_s.iat[0],end_s=g.time_s.iat[-1],
                        duration_s=span,covered_s=covered,energy_Wh=energy if covered>0 else np.nan,
                        horizontal_path_m=obs_distance,along_course_m=g.x_m.iat[-1]-g.x_m.iat[0],
                        weighted_mean_power_W=energy*3600/covered if covered>0 else np.nan,
                        covered_fraction=covered/span if span>0 else np.nan))
    measurement_label='HOVER' if case['cruise_speed_mps']==0 else 'CRUISE'
    measure=f[f.phase==measurement_label]
    by_phase={s['phase']:s for s in seg}
    d=by_phase.get(measurement_label,{})
    available_mean=d.get('weighted_mean_power_W',np.nan)
    baseline_mean=np.nan
    mean_rho=np.nan
    if len(measure)>1 and d['duration_s']>0:
        times=np.diff(measure.time_s)
        dens=measure.rho_kg_m3.to_numpy()
        mean_rho=float(np.dot(times,0.5*(dens[:-1]+dens[1:]))/d['duration_s'])
        if measurement_label=='CRUISE':
            baseline=np.array([model.phase_power('CRUISE',case['payload_kg'],rho) for rho in dens])
            baseline_mean=float(np.dot(times,0.5*(baseline[:-1]+baseline[1:]))/d['duration_s'])

    goal=case['cruise_speed_mps']; hover=goal==0
    if len(measure)<2:
        reasons.append('measurement_not_reached')
        speed_error=height_error=track_error=np.nan
    else:
        speed_error=float(np.max(np.abs(speed[measure.index]-goal)))
        height_error=float(np.max(np.abs(measure.height_agl_m-case['altitude_m'])))
        track_error=float(np.max(abs(measure.y_m)))
        if speed_error > max(0.3,0.05*goal)+1e-7: reasons.append('measurement_speed_outside_tolerance')
        if height_error > 0.5+1e-7: reasons.append('measurement_altitude_outside_tolerance')
        if track_error > 0.75+1e-7: reasons.append('measurement_cross_track_outside_tolerance')
        if hover:
            difference=d['duration_s']-case['hover_duration_s']
            if not (-1e-8 <= difference <= .041+1e-8): reasons.append('hover_duration_incomplete')
        else:
            difference=d['along_course_m']-case['measurement_distance_m']
            if not (-1e-7 <= difference <= max(0.2, (goal+max(.3,.05*goal))*.041)+1e-7):
                reasons.append('outbound_distance_incomplete')
        if abs(d['covered_s']-d['duration_s'])>DURATION_ATOL_S or not math.isfinite(available_mean):
            reasons.append('incomplete_measurement_power')
    duty=f.time_s.iat[-1]-f.time_s.iat[0]
    airborne=[x for x in seg if x['phase'] not in ('GROUND','LANDED')]
    eclose=float(abs(sum(s['energy_Wh'] for s in seg if math.isfinite(s['energy_Wh']))-seen_total))
    tclose=float(abs(sum(s['duration_s'] for s in seg)-duty))
    if eclose>BUDGET_ATOL_WH or tclose>DURATION_ATOL_S:
        raise ValueError(f'{run_id}: phase budget does not close')
    air_span=sum(s['duration_s'] for s in airborne)
    if air_span>duration_covered+DURATION_ATOL_S:
        reasons.append('airborne_power_gap')
    path=out/'traces'/f'{run_id}.csv.gz'
    f.insert(0,'case_id',case['case_id'])
    csv_gzip(f,path)
    # Reload and integrate the stored raw samples with an independent reducer.
    serial=pd.read_csv(path)
    qerr, qtime = independent_check(serial,seg)
    along, cross=relative_wind(0,case['wind_speed_mps'],case['wind_direction_toward_deg_ENU'])
    along=0. if abs(along)<1e-10 else along;cross=0. if abs(cross)<1e-10 else cross
    battery=model.budget(float(seen_total),complete_coverage=air_span>0 and abs(air_span-duration_covered)<DURATION_ATOL_S)
    return {**case,'run_id':run_id,'step_s':step,'engine_result':result,'status':'failed' if reasons else 'passed',
            'failure_reasons':'|'.join(reasons),'stock_eligible':not flags.stock_reasons,
            'stock_exclusions':'|'.join(flags.stock_reasons),'model_evidence':'|'.join(flags.evidence_flags),
            'seed':0,'wind_along_eastbound_mps':along,'wind_cross_eastbound_mps':cross,
            'wind_transfer':'C_NOT_CALIBRATED' if case['wind_speed_mps'] else 'no_commanded_wind',
            'physical_mass_kg':flags.physical_mass_kg,'regression_mass_kg':flags.regression_mass_kg,
            'measurement_mean_power_W':available_mean,'measurement_Wh':d.get('energy_Wh',np.nan),
            'measurement_mean_density_kg_m3':mean_rho,'phase_baseline_only_mean_W':baseline_mean,
            'measurement_duration_s':d.get('duration_s',np.nan),
            'measurement_along_m':d.get('along_course_m',np.nan) if not hover else np.nan,
            'measurement_Wh_per_km':1000*d['energy_Wh']/d['along_course_m'] if not hover and d.get('along_course_m',0)>0 else np.nan,
            'worst_speed_error_mps':speed_error,'worst_altitude_error_m':height_error,'worst_cross_track_m':track_error,
            'mission_supported_Wh':seen_total,'airborne_duration_s':air_span,'covered_s':duration_covered,
            'nominal_budget_status':battery.status,'nominal_remaining_Wh':battery.nominal_remaining_wh,
            'first_sample_nameplate_exhaustion_s':exhaustion_time,
            'code_phase_error_Wh':eclose,'independent_integration_error_Wh':qerr,'independent_time_error_s':qtime,
            'samples':len(f),'sha256':sha(path),'truth_sha256':source_meta['truth_sha256'],
            'ideal_steady_endurance_min':60*nominal/available_mean if available_mean>0 else np.nan,
            'ideal_steady_range_km':(3.6*goal*nominal/available_mean if not hover and available_mean>0 else np.nan),
            'mean_observed_horizontal_speed_mps':(d['horizontal_path_m']/d['duration_s']
                 if d.get('duration_s',0)>0 else np.nan),
            'mean_observed_height_agl_m':(float(np.dot(np.diff(measure.time_s),
                 (measure.height_agl_m.to_numpy()[:-1]+measure.height_agl_m.to_numpy()[1:])*.5))/d['duration_s']
                 if d.get('duration_s',0)>0 else np.nan),
            'world_sha256':source_meta['world_sha256']},seg


def independent_check(frame,segments):
    """Second integration using the delivered power and times, without the counter as an input."""
    errors=[]; times=[]
    for s in segments:
        g=frame.loc[frame.phase==s['phase']]
        t=g.time_s.to_numpy();p=g.power_W.to_numpy()
        watts=[];spans=[]
        for t0,t1,p0,p1 in zip(t[:-1],t[1:],p[:-1],p[1:],strict=True):
            if math.isfinite(p0) and math.isfinite(p1) and t1>t0:
                watts.append((t1-t0)*(p0+p1)/2)
                spans.append(t1-t0)
        actual=math.fsum(watts)/3600
        if math.isfinite(s['energy_Wh']): errors.append(abs(actual-s['energy_Wh']))
        times.append(abs(math.fsum(spans)-s['covered_s']))
        stored=g.cumulative_supported_Wh.iat[-1]-g.cumulative_supported_Wh.iat[0]
        errors.append(abs(stored-actual))
    peak=max(errors); elapsed=max(times)
    if peak>BUDGET_ATOL_WH or elapsed>DURATION_ATOL_S:
        raise ValueError(f'independent audit failed: {peak} Wh / {elapsed} s')
    return peak,elapsed


def verify_repeats(study, assets, cache, out, base):
    """Quantify numerical reproducibility across fresh Gazebo process launches."""
    model = Model.load()
    runtime = model.runtime_root / 'ros2_ws/src/drone_nav/config/m100_wind_simulation_v1.yaml'
    drag = yaml.safe_load(runtime.read_text())['horizontal_body_drag']
    area = drag['drag_coefficient'] * drag['reference_area_m2']
    cases = scenarios(study)
    reference = next(c for c in cases if c['case_id']=='case_001')
    windy = next(c for c in cases if c['design']=='matched_wind' and c['wind_speed_mps']>0)
    checks = []
    for case in (reference,windy):
        name=case['case_id']
        tag='repeat_'+name
        world=cache/f"m100_{round(100*case['payload_kg']):03d}_{BASE_STEP}.sdf"
        path=cache/f'{tag}.csv'
        reason=run_dart(cache/'engine',world,case,BASE_STEP,path,area)
        rec=dict(truth_sha256=sha(path),world_sha256=sha(world))
        rep,_=summarize_truth(path,case,tag,reason,model,out,BASE_STEP,rec)
        prior=base.loc[base.case_id==name].iloc[0]
        diff_power=rep['measurement_mean_power_W']-prior.measurement_mean_power_W
        diff_energy=rep['measurement_Wh']-prior.measurement_Wh
        diff_total=rep['mission_supported_Wh']-prior.mission_supported_Wh
        checks.append(dict(case_id=name,seed=0,step_s=BASE_STEP,base_status=prior.status,
                           repeat_status=rep['status'],repeat_trace_sha256=rep['sha256'],
                           repeated_sample_rows=rep['samples'],raw_byte_identical=rep['truth_sha256']==prior.truth_sha256,
                           measurement_delta_W=diff_power,measurement_delta_Wh=diff_energy,
                           mission_delta_Wh=diff_total,airborne_delta_s=rep['airborne_duration_s']-prior.airborne_duration_s,
                           within_declared_limits=(prior.status==rep['status']=='passed' and abs(diff_power)<=CONVERGENCE_POWER_W
                               and abs(diff_energy/prior.measurement_Wh)<=CONVERGENCE_ENERGY_REL
                               and abs(diff_total/prior.mission_supported_Wh)<=CONVERGENCE_ENERGY_REL)))
    return pd.DataFrame(checks)


def write_table(frame,path):
    partial = path.with_suffix('.partial')
    frame.to_csv(partial,index=False,float_format='%.11g',lineterminator='\n')
    partial.replace(path)


def tracking_and_convergence(frame):
    base=frame[frame.step_s==BASE_STEP].set_index('case_id')
    fine=frame[frame.step_s==FINE_STEP]
    rows=[]
    for _,b in fine.iterrows():
        a=base.loc[b.case_id]
        def delta(field):
            return b[field]-a[field]
        def relative(field):
            return delta(field)/a[field] if a[field] and math.isfinite(a[field]) else np.nan
        change=relative('measurement_Wh')
        total=relative('mission_supported_Wh')
        dp=delta('measurement_mean_power_W')
        same_outcome = tuple(a[k] for k in ['engine_result','status']) == tuple(b[k] for k in ['engine_result','status'])
        acceptable=bool(same_outcome and a.status=='passed' and abs(dp)<=CONVERGENCE_POWER_W
                        and abs(change)<=CONVERGENCE_ENERGY_REL and abs(total)<=CONVERGENCE_ENERGY_REL)
        rows.append(dict(case_id=b.case_id,coarse_dt_s=BASE_STEP,fine_dt_s=FINE_STEP,
                         coarse_status=a.status,fine_status=b.status,measurement_mean_difference_W=dp,
                         measurement_relative_energy_difference=change,mission_relative_energy_difference=total,
                         measurement_duration_difference_s=delta('measurement_duration_s'),
                         mission_duration_difference_s=delta('airborne_duration_s'),
                         measurement_distance_difference_m=delta('measurement_along_m'),
                         passes_declared_sensitivity_limits=acceptable))
    return pd.DataFrame(rows)


def split_assignment(payload, speed):
    """Keep every wind, temperature, route and altitude at the same operating pair together."""
    family = f'{payload:.2f}kg_{speed:.2f}mps'
    rank = int.from_bytes(hashlib.sha256(('m100-lee-split-1:' + family).encode()).digest()[:8], 'big') % 10
    split = 'train' if rank < 6 else 'validation' if rank < 8 else 'test'
    return family, split


def training_tables(base, out):
    # This dataset is strictly for the alternate research controller and its
    # current energy estimator. All rows are outside the stock takeoff limit.
    x=base[base.status=='passed'].copy()
    if x.empty:
        raise ValueError('no physically accepted research runs for training')
    x['target_phase']=np.where(x.cruise_speed_mps==0,'HOVER','CRUISE')
    # Every variant of the same planned speed/payload belongs to one family.
    # This deliberately keeps temperatures, winds, lengths and altitudes at
    # a given pair together and does not use outcomes to decide the split.
    groups=[split_assignment(p,v) for p,v in zip(x.payload_kg,x.cruise_speed_mps,strict=True)]
    x['split_group']=[g for g,_ in groups]; x['split']=[s for _,s in groups]
    x['route_for_cruise_m']=x.measurement_distance_m.where(x.cruise_speed_mps>0,np.nan)
    x['hover_for_measurement_s']=x.hover_duration_s.where(x.cruise_speed_mps==0,np.nan)
    columns=['case_id','split_group','split','payload_kg','cruise_speed_mps','wind_speed_mps',
             'wind_along_eastbound_mps','wind_cross_eastbound_mps','altitude_m','temperature_c',
             'route_for_cruise_m','hover_for_measurement_s','target_phase','measurement_mean_power_W']
    # Identical original and additional scenarios are a single example, while
    # case-level provenance stays in cases.csv. No synthetic measurement noise.
    features=columns[3:-1]
    x=x.drop_duplicates(features)
    write_table(x[columns],out/'training_research.csv')
    strict=x[x.stock_eligible]
    write_table(strict[columns],out/'training_stock.csv')
    return x, strict


def record_design(study, out):
    from .physics import relative_wind
    rows=[]
    for case in scenarios(study):
        along,cross=relative_wind(0.,case['wind_speed_mps'],case['wind_direction_toward_deg_ENU'])
        family, split = split_assignment(case['payload_kg'], case['cruise_speed_mps'])
        rows.append({**case, 'split_group':family, 'split':split, 'outbound_course_deg_enu':0,'along_wind_mps':0. if abs(along)<1e-10 else along,
                     'cross_wind_mps':0. if abs(cross)<1e-10 else cross,'engine_seed':0,'primary_dt_s':BASE_STEP})
    out.mkdir(parents=True,exist_ok=True)
    write_table(pd.DataFrame(rows),out/'scenario_plan.csv')


def original_acquisition(out):
    """Bind this generation to the original study only when the base/fine case records match."""
    relative = "results/m100_reaudit/gazebo_batch/study.json"
    # The saved copy also supports portable archives distributed without Git history.
    archive = Path(__file__).with_name("gazebo_batch_first_manifest.json")
    if sha(archive) != FIRST_MANIFEST_SHA256:
        raise ValueError("registered first Gazebo campaign manifest was changed")
    first = json.loads(archive.read_text())
    return {"first_release_git_commit": FIRST_RELEASE, "first_release_path": relative,
            "first_manifest_sha256": FIRST_MANIFEST_SHA256, "input_sha256": first["input_sha256"],
            "base_and_fine_cases_identical_to_first_release":
                first["output_table_sha256"]["cases.csv"] == sha(out/"cases.csv")}


def summarize(study, assets, cache,out, jobspec):
    out.mkdir(parents=True,exist_ok=True)
    model=Model.load()
    rows=[];phases=[];source_hashes={}
    for c,dt in jobspec:
        tag=c['case_id']+('_fine' if dt==FINE_STEP else '')
        assessment=assess(model,c)
        if not assessment.runtime_eligible:
            rows.append({**c,'run_id':tag,'step_s':dt,'seed':0,'engine_result':'not_launched',
                         'status':'excluded','failure_reasons':assessment.runtime_rejection,
                         'stock_eligible':not assessment.stock_reasons,
                         'stock_exclusions':'|'.join(assessment.stock_reasons),
                         'physical_mass_kg':assessment.physical_mass_kg,'regression_mass_kg':assessment.regression_mass_kg,
                         'model_evidence':'|'.join(assessment.evidence_flags)})
            continue
        src=json.loads((cache/(tag+'.json')).read_text())
        if sha(cache/(tag+'.csv'))!=src['truth_sha256']:
            raise ValueError('cached truth changed: '+tag)
        for path,digest in src['model_source_sha256'].items():
            if sha(assets/path)!=digest: raise ValueError('research airframe changed: '+path)
            source_hashes[path]=digest
        row,seg=summarize_truth(cache/(tag+'.csv'),c,tag,src['reason'],model,out,dt,src)
        rows.append(row);phases.extend(seg)
    frame=pd.DataFrame(rows)
    frame['model_version']='M100_RESEARCH_GAZEBO_DART_LEE_V1'
    allphases=pd.DataFrame(phases)
    base=frame[frame.step_s==BASE_STEP].copy()
    trained,strict=training_tables(base,out)
    write_table(frame,out/'cases.csv')
    write_table(allphases,out/'phases.csv')
    repeats=verify_repeats(study,assets,cache,out,base)
    write_table(repeats,out/'repeatability.csv')
    conv=tracking_and_convergence(frame)
    write_table(conv,out/'convergence.csv')
    # This comparison includes original inputs retained at the exact historical
    # case_id, while the controller, world and measurement trajectory differ.
    _,historic,_=read_study(study)
    before=[]
    for _,r in base[base.cohort=='original'].iterrows():
        record=historic[r.case_id]['metrics']
        before.append(dict(case_id=r.case_id,original_run=record['run_id'],research_status=r.status,
                           original_supported_mission_Wh=record['supported_phase_mission_energy_Wh'],
                           research_supported_mission_Wh=r.mission_supported_Wh,
                           original_measurement_Wh=record['measurement_energy_Wh'],research_measurement_Wh=r.measurement_Wh,
                           difference_measurement_Wh=r.measurement_Wh-record['measurement_energy_Wh'],
                           original_mean_power_W=record['sampled_mean_power_W'],research_mean_power_W=r.measurement_mean_power_W))
    write_table(pd.DataFrame(before),out/'before_after.csv')
    configdir=model.runtime_root/'ros2_ws/src/drone_nav/config'
    modules=model.runtime_root/'ros2_ws/src/drone_nav/drone_nav'
    source_hashes.update({f'drone_nav/config/{n}':sha(configdir/n) for n in
                      [model.settings['energy'][k] for k in ('published_phase_config','stable_cruise_config','transient_config','wind_config')]})
    source_hashes.update({f'drone_nav/{n}':sha(modules/n) for n in
                      ['m100_published_energy_v2.py','m100_atmosphere.py','m100_stable_cruise_speed.py',
                       'm100_wind_simulation.py','m100_transient_power.py','m100_contract.py']})
    source_hashes.update({p.name:sha(p) for p in [ENGINE,Path(__file__),Path(__file__).with_name('physics.py'),study/'protocol.json',study/SUMMARY_NAME,
                          ROOT/'config/m100_reaudit/physics.json']})
    tables = ['scenario_plan.csv', 'cases.csv', 'phases.csv', 'convergence.csv', 'repeatability.csv',
              'training_research.csv', 'training_stock.csv', 'before_after.csv']
    manifest={
        'schema':'M100_RESEARCH_GAZEBO_DART_LEE_V1', 'engine':'Gazebo Harmonic, DART, native Gazebo Lee velocity control and four MulticopterMotorModel systems',
        'true_px4_sitl':False,'onboard_sensors_used_by_controller':False,'primary_step_s':BASE_STEP,'fine_step_s':FINE_STEP,
        'readout_interval_s':.2,'guidance_interval_s':.04,'gust_model':None,'random_sensor_noise':False,'seed':0,
        'numerical_reproducibility':'Externally scheduled Gazebo Transport velocity commands may shift phase boundaries by a control tick, check repeatability.csv; no repeated run is an independent training example.',
        'motor_model_and_collision_shapes':'Preserved from the source research airframe incl. fixed-joint payload variants',
        'pack_voltage_limits_motor_commands':False,
        'versions':{pkg:subprocess.check_output(['pkg-config','--modversion',pkg],text=True).strip()
                    for pkg in ('gz-sim8','gz-physics7-dartsim-plugin','gz-sim8-multicopter-control-system','sdformat14')},
        'thresholds':{'maximum_window_height_error_m':.5,'maximum_cross_track_m':.75,'speed_tolerance':'max(0.3,0.05*command) m/s',
                      'energy_check_abs_Wh':BUDGET_ATOL_WH,'duration_abs_s':DURATION_ATOL_S,
                      'convergence_measurement_mean_abs_W':CONVERGENCE_POWER_W,'convergence_energy_relative':CONVERGENCE_ENERGY_REL},
        'input_sha256':source_hashes,
        'original_acquisition':original_acquisition(out),
        'output_table_sha256':{name:sha(out/name) for name in tables},
        'original_conditions':int((base.cohort=='original').sum()),'additional_conditions':int((base.cohort=='expanded').sum()),
        'base_statuses':base.status.value_counts().to_dict(),
        'counts_by_cohort':{name:g.status.value_counts().to_dict() for name,g in base.groupby('cohort')},
        'engine_completions':base.engine_result.value_counts().to_dict(),'base_stock_eligible':int(base.stock_eligible.sum()),
        'measured_quality_failures':base.loc[base.status=='failed','failure_reasons'].value_counts().to_dict(),
        'rejected_commands':base.loc[base.status=='excluded','failure_reasons'].value_counts().to_dict(),
        'actual_base_simulated':int((base.engine_result!='not_launched').sum()),'fine_runs':len(conv),
        'fresh_repeat_runs':len(repeats),'repeats_within_tolerance':int(repeats.within_declared_limits.sum()),
        'extra_repeat_trace_rows':int(repeats.repeated_sample_rows.sum()),
        'convergence_within_tolerance':int(conv.passes_declared_sensitivity_limits.sum()),
        'accepted_research_unique_rows':len(trained),'research_split_counts':trained.split.value_counts().to_dict(),
        'stock_training_rows':len(strict),'trace_rows':int(frame.samples.sum()),
        'maximum_independent_energy_error_Wh':float(frame.independent_integration_error_Wh.max()),
        'maximum_independent_duration_error_s':float(frame.independent_time_error_s.max()),
        'source_claims':'Simulation target with uncalibrated wind and research controller; all stock-ineligible. Field validation separately in external/.',
        'bundle_bytes':sum(p.stat().st_size for p in (out/'traces').glob('*.gz')),
    }
    partial = out/'study.json.partial'
    partial.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    partial.replace(out/'study.json')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,help='Historical protocol directory or 69-case CSV')
    parser.add_argument('--px4',type=Path,default=Path(os.environ.get('PX4_ROOT',DEFAULT_PX4)),
                        help='Checkout containing the traceable m100_cmu_v2 SDF assets')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--cache',type=Path,help='Resumable raw Gazebo work directory; defaults outside versioned results')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.jobs<1: parser.error('at least one worker is required')
    study=research_study(args.study)
    out=args.output.resolve(); cache=(args.cache or DEFAULT_CACHE).resolve()
    if out.is_relative_to(study) or cache.is_relative_to(study) or cache==out:
        raise ValueError("choose independent output and cache directories")
    assets=args.px4/'Tools/simulation/gz/models'
    record_design(study,out)
    jobspec=truth_stage(study,assets,cache,args.jobs)
    summary=summarize(study,assets,cache,out,jobspec)
    print(json.dumps({k:summary[k] for k in ('base_statuses','actual_base_simulated','fine_runs',
                           'convergence_within_tolerance','accepted_research_unique_rows','stock_training_rows')},indent=2))


if __name__=='__main__':
    main()
