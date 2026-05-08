"""Run the G1 control loop with a Flask HTTP server for remote commands.

Same as run_g1_control_loop.py, plus a Flask thread on port 5055 that exposes:
  POST /cmd   {"vx":..,"vy":..,"vyaw":..,"height":..,
               "active":bool,"freq":..,"roll":..,
               "pitch":..,"yaw":..}    -> absolute setpoints (deg for r/p/y)
  POST /stop                           -> zero cmd + deactivate policy
  GET  /state                          -> current commanded values
  GET  /obs                            -> latest robot observation (joint positions,
                                          velocities, base pose, IMU, ...)
  GET  /joint_names                    -> ordered list of joint names (indices match obs.q)
  GET  /upper_body_names               -> upper-body joint names + their indices in obs.q
  POST /upper_body  {"target":[..n..]} OR {"joints":{"name":val,...}}
                    [, "duration": seconds]      -> command upper-body joint targets
  GET  /cameras                        -> list of available camera/topic names
  GET  /rgb[?camera=<name>]            -> latest RGB frame as JPEG bytes
  GET  /gripper                        -> latest dex1 gripper state(s)
  POST /gripper {"side":"right|left|both", "q": <rad>}      OR
                {"right": <rad>, "left": <rad>}            -> command gripper(s)

Bind address defaults to 0.0.0.0 so the Mac on the LAN can reach it.

RGB source: JPEG-over-TCP from a gst-launch tcpserversink running on the G1.
  Default 192.168.123.164:5000. Override:
      G1_CAMERA_HOST=...   (set empty to disable the latch)
      G1_CAMERA_PORT=...

Dex1 gripper:
  Talks to dex1_1_gripper_server over Unitree DDS topics
  rt/dex1/{right,left}/{cmd,state}. Configure with:
      G1_GRIPPER=0        disable
      G1_GRIPPER_IFACE=enp4s0   network interface for the unitree channel factory
      G1_GRIPPER_SIDES=right,left  which sides to subscribe / publish
"""
from copy import deepcopy
import os
import socket
import threading
import time

from flask import Flask, Response, jsonify, request
import numpy as np
import tyro

from decoupled_wbc.control.envs.g1.g1_env import G1Env
from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    JOINT_SAFETY_STATUS_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
    ROSKeyboardDispatcher,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
    ROSServiceServer,
)
from decoupled_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "ControlPolicyServer"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5055
DEFAULT_CAMERA_HOST = "192.168.123.164"
DEFAULT_CAMERA_PORT = 5000
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
DEX1_KP = 5.0
DEX1_KD = 0.05
DEFAULT_GRIPPER_IFACE = "enp4s0"


class JpegTcpLatch:
    """Background thread that maintains a TCP connection to a gst-launch
    tcpserversink (or any source emitting concatenated JPEGs) and latches
    the most recent complete JPEG. Auto-reconnects if the link drops."""

    def __init__(self, host: str, port: int, name: str = "g1_camera"):
        self.host = host
        self.port = port
        self.name = name
        self._lock = threading.Lock()
        self._jpeg = None
        self._timestamp = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"jpeg-latch-{name}")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5)
                sock.setblocking(False)
                self._connected = True
                print(f"[server] camera latch connected to {self.host}:{self.port}")
                buf = b""
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(1 << 20)
                    except BlockingIOError:
                        time.sleep(0.005)
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    last_eoi = buf.rfind(JPEG_EOI)
                    if last_eoi >= 0:
                        last_soi = buf.rfind(JPEG_SOI, 0, last_eoi)
                        if last_soi >= 0:
                            jpeg = buf[last_soi : last_eoi + 2]
                            buf = buf[last_eoi + 2 :]
                            with self._lock:
                                self._jpeg = jpeg
                                self._timestamp = time.time()
                    if len(buf) > 5_000_000:
                        buf = buf[-1_000_000:]
                sock.close()
                self._connected = False
            except (OSError, socket.timeout) as e:
                self._connected = False
                if not self._stop.is_set():
                    print(f"[server] camera latch reconnect in 2s ({e})")
                    self._stop.wait(2.0)

    def get_jpeg(self):
        with self._lock:
            return self._jpeg, self._timestamp

    def is_connected(self) -> bool:
        return self._connected

    def stop(self):
        self._stop.set()


class Dex1Latch:
    """Subscribes to rt/dex1/{side}/state for state readback and publishes
    to rt/dex1/{side}/cmd for commands. Talks to dex1_1_gripper_server on
    the G1 via Unitree's DDS channel."""

    def __init__(self, sides=("right", "left"), iface=DEFAULT_GRIPPER_IFACE,
                 kp=DEX1_KP, kd=DEX1_KD):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
            MotorCmd_, MotorCmds_, MotorStates_,
        )
        try:
            ChannelFactoryInitialize(0, iface)
        except Exception as e:
            # already initialized by env in --interface real, that's fine
            print(f"[server][dex1] ChannelFactoryInitialize note: {e}")

        self._MotorCmd_ = MotorCmd_
        self._MotorCmds_ = MotorCmds_
        self._kp = float(kp)
        self._kd = float(kd)
        self._lock = threading.Lock()
        self._states = {}        # side -> {q, dq, tau, ts}
        self._pubs = {}          # side -> ChannelPublisher
        self._subs = []
        self.sides = list(sides)

        for side in self.sides:
            sub = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
            sub.Init(handler=self._make_state_cb(side), queueLen=1)
            self._subs.append(sub)
            pub = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
            pub.Init()
            self._pubs[side] = pub
            print(f"[server][dex1] sub rt/dex1/{side}/state, pub rt/dex1/{side}/cmd")

    def _make_state_cb(self, side):
        def cb(msg):
            if not msg.states:
                return
            s = msg.states[0]
            with self._lock:
                self._states[side] = {
                    "q": float(s.q),
                    "dq": float(s.dq),
                    "tau": float(s.tau_est),
                    "ts": time.time(),
                }
        return cb

    def get_states(self):
        with self._lock:
            return {k: dict(v) for k, v in self._states.items()}

    def set_position(self, side: str, q: float,
                     kp: float | None = None, kd: float | None = None):
        if side not in self._pubs:
            raise ValueError(f"unknown side: {side!r} (have {list(self._pubs)})")
        cmd = self._MotorCmds_()
        cmd.cmds = [self._MotorCmd_(
            mode=1, q=float(q), dq=0.0, tau=0.0,
            kp=float(self._kp if kp is None else kp),
            kd=float(self._kd if kd is None else kd),
            reserve=[0, 0, 0],
        )]
        self._pubs[side].Write(cmd)


class LatestSnapshot:
    """Thread-safe holder for the most recent observation as JSON-able dict.
    Image fields (anything ending in '_image') are stripped — RGB comes
    from the camera latch, not the obs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._json = None

    def set(self, obs):
        with self._lock:
            self._json = _nested_jsonable(
                {k: v for k, v in obs.items()
                 if not (isinstance(k, str) and k.endswith("_image"))}
            )

    def get_json(self):
        with self._lock:
            return self._json


def _nested_jsonable(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _nested_jsonable(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
        else:
            out[k] = v
    return out


def build_flask_app(wbc_policy, latest, camera, dex1):
    app = Flask(__name__)

    def lower():
        return getattr(wbc_policy, "lower_body_policy", wbc_policy)

    @app.post("/cmd")
    def post_cmd():
        body = request.get_json(force=True) or {}
        p = lower()
        if "vx" in body:    p.cmd[0] = float(body["vx"])
        if "vy" in body:    p.cmd[1] = float(body["vy"])
        if "vyaw" in body:  p.cmd[2] = float(body["vyaw"])
        if "height" in body: p.height_cmd = float(body["height"])
        if "freq" in body:  p.freq_cmd = max(1.0, min(2.0, float(body["freq"])))
        if "roll" in body:  p.roll_cmd = np.deg2rad(float(body["roll"]))
        if "pitch" in body: p.pitch_cmd = np.deg2rad(float(body["pitch"]))
        if "yaw" in body:   p.yaw_cmd = np.deg2rad(float(body["yaw"]))
        if "active" in body: p.use_policy_action = bool(body["active"])
        return jsonify(ok=True, state=_state(p))

    @app.post("/stop")
    def post_stop():
        p = lower()
        p.cmd[0] = p.cmd[1] = p.cmd[2] = 0.0
        p.use_policy_action = False
        return jsonify(ok=True, state=_state(p))

    @app.get("/state")
    def get_state():
        return jsonify(_state(lower()))

    @app.get("/obs")
    def get_obs():
        data = latest.get_json()
        if data is None:
            return jsonify(error="no observation yet"), 503
        return jsonify(data)

    @app.get("/joint_names")
    def get_joint_names():
        return jsonify(joint_names=list(wbc_policy.robot_model.joint_names))

    def _ub_names_indices():
        rm = wbc_policy.robot_model
        idxs = list(rm.get_joint_group_indices("upper_body"))
        names = [rm.joint_names[i] for i in idxs]
        return names, idxs

    @app.get("/upper_body_names")
    def get_upper_body_names():
        names, idxs = _ub_names_indices()
        return jsonify(joint_names=names, indices=idxs)

    @app.post("/upper_body")
    def post_upper_body():
        body = request.get_json(force=True) or {}
        duration = float(body.get("duration", 1.0))
        if duration <= 0:
            return jsonify(error="duration must be > 0 seconds"), 400

        ub_names, ub_idxs = _ub_names_indices()

        # Start from current upper-body pose so partial updates leave others alone
        obs = latest.get_json()
        if obs is None or "q" not in obs:
            return jsonify(error="no observation yet"), 503
        q_full = np.asarray(obs["q"], dtype=float)
        target = q_full[ub_idxs].copy()

        if "target" in body:
            full = np.asarray(body["target"], dtype=float)
            if full.shape != (len(ub_idxs),):
                return jsonify(
                    error=f"'target' length {full.size} != {len(ub_idxs)}",
                    expected_length=len(ub_idxs),
                ), 400
            target = full
        elif "joints" in body:
            partial = body["joints"] or {}
            unknown = [n for n in partial if n not in ub_names]
            if unknown:
                return jsonify(
                    error=f"unknown upper-body joints: {unknown}",
                    available=ub_names,
                ), 400
            for n, v in partial.items():
                target[ub_names.index(n)] = float(v)
        else:
            return jsonify(
                error="provide 'target' (full vector) or 'joints' (dict by name)",
                upper_body_joint_count=len(ub_idxs),
            ), 400

        now = time.monotonic()
        wbc_policy.set_goal({
            "target_upper_body_pose": target,
            "target_time": now + duration,
            "interpolation_garbage_collection_time": now - 0.04,
        })
        return jsonify(
            ok=True,
            target=target.tolist(),
            joint_names=ub_names,
            duration=duration,
        )

    @app.get("/cameras")
    def get_cameras():
        if camera is None:
            return jsonify(cameras=[])
        return jsonify(cameras=[camera.name],
                       connected=camera.is_connected(),
                       source=f"{camera.host}:{camera.port}")

    @app.get("/rgb")
    def get_rgb():
        if camera is None:
            return jsonify(error="camera latch disabled (set G1_CAMERA_HOST)"), 503
        jpeg, ts = camera.get_jpeg()
        if jpeg is None:
            return jsonify(
                error="no image yet",
                connected=camera.is_connected(),
                source=f"{camera.host}:{camera.port}",
                hint="is the gst-launch tcpserversink running on the G1?",
            ), 503
        return Response(
            jpeg,
            mimetype="image/jpeg",
            headers={
                "X-Camera-Name": camera.name,
                "X-Frame-Timestamp": f"{ts:.6f}",
            },
        )

    @app.get("/gripper")
    def get_gripper():
        if dex1 is None:
            return jsonify(error="gripper disabled (set G1_GRIPPER=1 to enable)"), 503
        states = dex1.get_states()
        if not states:
            return jsonify(
                error="no gripper state yet",
                hint="is dex1_1_gripper_server running on the G1?",
                sides_subscribed=dex1.sides,
            ), 503
        return jsonify(states)

    @app.post("/gripper")
    def post_gripper():
        if dex1 is None:
            return jsonify(error="gripper disabled (set G1_GRIPPER=1 to enable)"), 503
        body = request.get_json(force=True) or {}
        applied = {}
        try:
            if "side" in body and "q" in body:
                side = body["side"]
                q = float(body["q"])
                targets = dex1.sides if side == "both" else [side]
                for s in targets:
                    dex1.set_position(s, q)
                    applied[s] = q
            else:
                for s in ("right", "left"):
                    if s in body:
                        dex1.set_position(s, float(body[s]))
                        applied[s] = float(body[s])
        except ValueError as e:
            return jsonify(error=str(e)), 400
        if not applied:
            return jsonify(
                error="provide {'side':'right|left|both','q':<rad>} or {'right':<rad>,'left':<rad>}",
            ), 400
        return jsonify(ok=True, applied=applied)

    return app


def _state(p):
    return {
        "vx": float(p.cmd[0]),
        "vy": float(p.cmd[1]),
        "vyaw": float(p.cmd[2]),
        "height": float(p.height_cmd),
        "freq": float(p.freq_cmd),
        "roll_deg": float(np.rad2deg(p.roll_cmd)),
        "pitch_deg": float(np.rad2deg(p.pitch_cmd)),
        "yaw_deg": float(np.rad2deg(p.yaw_cmd)),
        "active": bool(p.use_policy_action),
    }


def start_flask_thread(wbc_policy, latest, camera, dex1):
    app = build_flask_app(wbc_policy, latest, camera, dex1)
    t = threading.Thread(
        target=lambda: app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, use_reloader=False),
        daemon=True,
        name="flask-cmd-server",
    )
    t.start()
    print(f"[server] HTTP command server listening on {SERVER_HOST}:{SERVER_PORT}")
    return t


def start_camera_latch():
    host = os.environ.get("G1_CAMERA_HOST", DEFAULT_CAMERA_HOST)
    if not host:
        print("[server] camera latch disabled (G1_CAMERA_HOST is empty)")
        return None
    port = int(os.environ.get("G1_CAMERA_PORT", DEFAULT_CAMERA_PORT))
    return JpegTcpLatch(host=host, port=port, name="g1_camera")


def start_dex1_latch():
    if os.environ.get("G1_GRIPPER", "1") in ("0", "false", "False", ""):
        print("[server] dex1 gripper latch disabled (G1_GRIPPER=0)")
        return None
    iface = os.environ.get("G1_GRIPPER_IFACE", DEFAULT_GRIPPER_IFACE)
    sides = [s.strip() for s in
             os.environ.get("G1_GRIPPER_SIDES", "right,left").split(",")
             if s.strip()]
    try:
        return Dex1Latch(sides=tuple(sides), iface=iface)
    except Exception as e:
        print(f"[server][dex1] failed to start latch: {e}")
        return None


def main(config: ControlLoopConfig):
    ros_manager = ROSManager(node_name=CONTROL_NODE_NAME)
    node = ros_manager.node

    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())

    wbc_config = config.load_wbc_yaml()

    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)

    telemetry = Telemetry(window_size=100)

    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()

    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config, config.upper_body_joint_speed)

    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    else:
        raise ValueError(
            f"Invalid keyboard dispatcher: {config.keyboard_dispatcher_type}, please use 'raw' or 'ros'"
        )
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(keyboard_estop)
    dispatcher.start()

    latest = LatestSnapshot()
    camera = start_camera_latch()
    dex1 = start_dex1_latch()
    start_flask_thread(wbc_policy, latest, camera, dex1)

    rate = node.create_rate(config.control_frequency)

    upper_body_policy_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)

    last_teleop_cmd = None
    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            with telemetry.timer("total_loop"):
                with telemetry.timer("step_simulator"):
                    if env.sim and config.sim_sync_mode:
                        env.step_simulator()

                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)
                    latest.set(obs)

                with telemetry.timer("policy_setup"):
                    upper_body_cmd = upper_body_policy_subscriber.get_msg()
                    t_now = time.monotonic()

                    wbc_goal = {}
                    if upper_body_cmd:
                        wbc_goal = upper_body_cmd.copy()
                        last_teleop_cmd = upper_body_cmd.copy()
                        if config.ik_indicator:
                            env.set_ik_indicator(upper_body_cmd)
                    if wbc_goal:
                        wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                            1 / config.control_frequency
                        )
                        wbc_policy.set_goal(wbc_goal)

                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                with telemetry.timer("publish_status"):
                    policy_use_action = False
                    try:
                        if hasattr(wbc_policy, "lower_body_policy"):
                            policy_use_action = getattr(
                                wbc_policy.lower_body_policy, "use_policy_action", False
                            )
                    except (AttributeError, TypeError):
                        policy_use_action = False

                    lower_body_policy_status_pub.publish(
                        {"use_policy_action": policy_use_action, "timestamp": t_now}
                    )
                    joint_safety_status_pub.publish(
                        {"joint_safety_ok": env.get_joint_safety_status(), "timestamp": t_now}
                    )

                if wbc_goal.get("toggle_data_collection", False):
                    dispatcher.handle_key("c")
                if wbc_goal.get("toggle_data_abort", False):
                    dispatcher.handle_key("x")

                if env.use_sim and wbc_goal.get("reset_env_and_policy", False):
                    print("Resetting sim environment and policy")
                    dispatcher.handle_key("k")
                    upper_body_policy_subscriber._msg = None
                    upper_body_cmd = {
                        "target_upper_body_pose": obs["q"][
                            robot_model.get_joint_group_indices("upper_body")
                        ],
                        "wrist_pose": DEFAULT_WRIST_POSE,
                        "base_height_command": DEFAULT_BASE_HEIGHT,
                        "navigate_cmd": DEFAULT_NAV_CMD,
                    }
                    last_teleop_cmd = upper_body_cmd.copy()
                    time.sleep(0.5)

                msg = deepcopy(obs)
                for key in obs.keys():
                    if key.endswith("_image"):
                        del msg[key]

                if last_teleop_cmd:
                    msg.update(
                        {
                            "action": wbc_action["q"],
                            "action.eef": last_teleop_cmd.get("wrist_pose", DEFAULT_WRIST_POSE),
                            "base_height_command": last_teleop_cmd.get(
                                "base_height_command", DEFAULT_BASE_HEIGHT
                            ),
                            "navigate_command": last_teleop_cmd.get(
                                "navigate_cmd", DEFAULT_NAV_CMD
                            ),
                            "timestamps": {
                                "main_loop": time.time(),
                                "proprio": time.time(),
                            },
                        }
                    )
                data_exp_pub.publish(msg)
                end_time = time.monotonic()

            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                raise RuntimeError("Simulator thread is not alive")

            rate.sleep()

            if config.verbose_timing:
                telemetry.log_timing_info(context="G1 Control Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.control_frequency) and not config.sim_sync_mode:
                telemetry.log_timing_info(context="G1 Control Loop Missed", threshold=0.001)

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")
    finally:
        print("Cleaning up...")
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
