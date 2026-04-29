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

Bind address defaults to 0.0.0.0 so the Mac on the LAN can reach it.

RGB sources (in priority order):
  1. ROS sensor_msgs/Image topic. Default '/camera/color/image_raw'.
     Override with env var G1_RGB_TOPIC. Comma-separate to subscribe to multiple
     (then /rgb?camera=<name> selects one; name is the topic without leading '/').
  2. The sim env's offscreen-rendered cameras (only present if the sim was
     started with image rendering — the basic --interface sim does not).
"""
from copy import deepcopy
import os
import threading
import time

import cv2
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
DEFAULT_RGB_TOPIC = "/camera/color/image_raw"


class RosImageLatch:
    """Subscribes to one or more sensor_msgs/Image topics and latches the
    most recent BGR frame from each. Frame name is the topic with the
    leading '/' stripped (e.g. 'camera/color/image_raw')."""

    def __init__(self, ros_node, topics):
        from sensor_msgs.msg import Image
        from decoupled_wbc.control.utils.cv_bridge import CvBridge
        self._lock = threading.Lock()
        self._frames = {}  # name -> (bgr_ndarray, timestamp_sec)
        self._bridge = CvBridge()
        self._subs = []
        for topic in topics:
            name = topic.lstrip("/")

            def make_cb(n):
                def cb(msg):
                    try:
                        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    except Exception as e:
                        print(f"[server] cv_bridge failed for {n}: {e}")
                        return
                    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    with self._lock:
                        self._frames[n] = (img, ts)
                return cb

            sub = ros_node.create_subscription(Image, topic, make_cb(name), 1)
            self._subs.append(sub)
            print(f"[server] subscribed to RGB topic: {topic}")

    def get(self, name=None):
        with self._lock:
            if not self._frames:
                return None, None, None
            if name is None:
                name = next(iter(self._frames))
            entry = self._frames.get(name)
            if entry is None:
                return None, None, None
            img, ts = entry
            return name, img, ts

    def list_names(self):
        with self._lock:
            return list(self._frames.keys())


class LatestSnapshot:
    """Thread-safe holder for the most recent observation (split into JSON + images)."""
    def __init__(self):
        self._lock = threading.Lock()
        self._json = None
        self._images = {}  # camera_name -> latest np.ndarray (HxWx3 uint8 RGB)

    def set(self, obs):
        json_data = {}
        images = {}
        for k, v in obs.items():
            if isinstance(k, str) and k.endswith("_image"):
                # strip "_image" suffix to get the camera name
                images[k[: -len("_image")]] = v
            elif hasattr(v, "tolist"):
                json_data[k] = v.tolist()
            elif isinstance(v, dict):
                json_data[k] = _nested_jsonable(v)
            elif isinstance(v, (list, tuple)):
                json_data[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
            else:
                json_data[k] = v
        with self._lock:
            self._json = json_data
            self._images = images

    def get_json(self):
        with self._lock:
            return self._json

    def get_image(self, camera=None):
        with self._lock:
            if not self._images:
                return None, None
            if camera is None:
                camera = next(iter(self._images))
            img = self._images.get(camera)
            return camera, img

    def list_cameras(self):
        with self._lock:
            return list(self._images.keys())


def _nested_jsonable(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _nested_jsonable(v)
        else:
            out[k] = v
    return out


def build_flask_app(wbc_policy, latest):
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
        return jsonify(cameras=latest.list_cameras())

    @app.get("/rgb")
    def get_rgb():
        camera = request.args.get("camera")
        name, img = latest.get_image(camera)
        if img is None:
            return jsonify(
                error="no image available",
                hint="run sim with --enable-offscreen / --image-publish, "
                     "or start the camera forwarder for real robot",
                cameras=latest.list_cameras(),
            ), 503
        # obs images are RGB; cv2 expects BGR for JPEG encoding
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return jsonify(error="jpeg encode failed"), 500
        return Response(
            buf.tobytes(),
            mimetype="image/jpeg",
            headers={"X-Camera-Name": name},
        )

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


def start_flask_thread(wbc_policy, latest):
    app = build_flask_app(wbc_policy, latest)
    t = threading.Thread(
        target=lambda: app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True, use_reloader=False),
        daemon=True,
        name="flask-cmd-server",
    )
    t.start()
    print(f"[server] HTTP command server listening on {SERVER_HOST}:{SERVER_PORT}")
    return t


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
    start_flask_thread(wbc_policy, latest)

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
