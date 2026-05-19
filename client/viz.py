import time
import mujoco
import mujoco.viewer

from client import G1Client

HOST = "192.168.1.61"
PORT = 5055
MJCF = "g1/g1.xml"
RATE_HZ = 50

env = G1Client(HOST, PORT)
# env.stop_policy()

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)

# Map robot joint name -> mjModel qpos address, skipping hand joints
# (real robot has a different gripper than the MJCF).
robot_names = env.get_joint_names()
name_to_addr = {}
for n in robot_names:
    if "hand" in n:
        continue
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    if jid >= 0:
        name_to_addr[n] = model.jnt_qposadr[jid]

print(f"mirroring {len(name_to_addr)} joints (skipped {len(robot_names) - len(name_to_addr)})")

period = 1.0 / RATE_HZ

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        try:
            obs = env.get_state()
        except Exception as e:
            print("get_state failed:", e)
            time.sleep(period)
            continue

        fb = obs.get("floating_base_pose")
        if fb is not None and len(fb) >= 7:
            data.qpos[0:3] = fb[0:3]
            data.qpos[3:7] = fb[3:7]

        q = obs.get("q")
        if q is not None and len(q) == len(robot_names):
            for n, addr in name_to_addr.items():
                data.qpos[addr] = q[robot_names.index(n)]

        mujoco.mj_forward(model, data)
        viewer.sync()

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
