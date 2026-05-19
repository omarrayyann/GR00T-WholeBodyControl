import time
import mujoco
import mujoco.viewer

from client import G1Client

HOST = "192.168.1.61"
PORT = 5055
MJCF = "g1/g1.xml"
RATE_HZ = 30
SEND_DURATION = 0.15

env = G1Client(HOST, PORT)

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)

# Full ordered upper-body name list — we always send a target of this length.
ub_names = env.get_upper_body_names()

# Joints we actually drive via the viewer's Control sliders (skip hand joints —
# different gripper). Map name -> (actuator id, qpos address).
driven = {}
for n in ub_names:
    if "hand" in n:
        continue
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    if aid >= 0 and jid >= 0:
        driven[n] = (aid, model.jnt_qposadr[jid])

# Seed UPPER BODY ONLY from the robot's current pose. We never read any other
# part of the robot state — lower body / floating base stay at MJCF defaults.
all_q = env.get_joints_dict()
seed = {n: all_q[n] for n in ub_names if n in all_q}
hold = {n: seed.get(n, 0.0) for n in ub_names if n not in driven}
for n, (aid, addr) in driven.items():
    if n in seed:
        data.ctrl[aid] = seed[n]
        data.qpos[addr] = seed[n]
mujoco.mj_forward(model, data)

print(f"streaming all {len(ub_names)} upper-body joints "
      f"({len(driven)} driven, {len(hold)} held).")

period = 1.0 / RATE_HZ

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()

        # Mirror ctrl -> qpos for driven joints so the model tracks sliders
        # without stepping physics.
        for n, (aid, addr) in driven.items():
            data.qpos[addr] = data.ctrl[aid]

        mujoco.mj_forward(model, data)
        viewer.sync()

        # Build the full target vector in upper-body order.
        target = [
            float(data.ctrl[driven[n][0]]) if n in driven else hold[n]
            for n in ub_names
        ]

        try:
            env.set_upper_body(target=target, duration=SEND_DURATION)
        except Exception as e:
            print("set_upper_body failed:", e)

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
