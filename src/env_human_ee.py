# ============================================================================
# Windows 环境: 直接使用 pygame 读取手柄输入
# 运行方式: python env_human_ee.py --repo_id=./data/your_data
# ============================================================================
import os
import argparse

# Offscreen rendering configuration
os.environ["MUJOCO_GL"] = "glfw"

import time
import threading
import logging
import mujoco
import mujoco.viewer
import numpy as np
import subprocess
import io
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from helper import *
import pygame


def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="MuJoCo SO100 data collection script")

    parser.add_argument(
        "--repo_id",
        type=str,
        default="./data/NewData3.9-ee-2d-pos",
        help="Path to the LeRobot dataset repository"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Recording frame rate"
    )
    parser.add_argument(
        "--move_speed",
        type=float,
        default=0.05,
        help="Mocap translation speed"
    )
    parser.add_argument(
        "--rot_speed",
        type=float,
        default=1.0,
        help="Mocap rotation speed"
    )
    parser.add_argument(
        "--stream_port",
        type=int,
        default=8080,
        help="Port for MJPEG stream server"
    )

    args = parser.parse_args()

    # --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # --- Global configuration ---
    MOVE_SPEED = args.move_speed
    ROT_SPEED = args.rot_speed
    DEADZONE = 0.1
    FPS = args.fps
    VIDEO_STEP = 1.0 / FPS
    OBS_IMAGES_SHAPE = (224, 224, 3)

    # Environment configuration
    pos_random_range = 0.05

    # LeRobot dataset path and parameters
    REPO_ID = Path(args.repo_id).absolute()
    ROBOT_TYPE = "so100_arm"
    TOLERANCE = 0.0001

    # Recording state
    is_recording = False
    record_buffer = []
    save_thread = None

    # Button debounce
    buttonCooldown = 0.0
    COOLDOWN_SEC = 0.3

    # --- MJPEG Stream Server ---
    _latest_frame_top = None
    _latest_frame_side = None
    _frame_lock = threading.Lock()
    _streaming_active = True
    _stream_info = {
        "is_recording": False,
        "frame_count": 0,
        "mocap_x": 0.0,
        "mocap_y": 0.0,
    }

    # --- Initialize pygame joystick ---
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
    if joystick:
        joystick.init()
        logger.info(f"Joystick initialized: {joystick.get_name()}")
    else:
        logger.error("No joystick detected")
        return

    class MJPEGHandler(BaseHTTPRequestHandler):
        """Handle MJPEG stream requests for camera feeds"""

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                html = b'''<!DOCTYPE html>
<html>
<head>
    <title>SO100 Camera Streams</title>
    <style>
        body { margin: 0; background: #1a1a2e; display: flex; flex-direction: column; align-items: center; min-height: 100vh; font-family: Arial, sans-serif; color: #fff; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px; }
        .camera-box { display: flex; flex-direction: column; align-items: center; }
        .camera-box h3 { margin: 0 0 10px 0; color: #4fc3f7; }
        .camera-box img { border: 2px solid #4fc3f7; border-radius: 8px; max-width: 100%; height: auto; }
        .info-panel { background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px 25px; margin: 20px; text-align: center; }
        .info-panel .recording { color: #ff5252; font-weight: bold; animation: blink 1s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.3; } }
        .info-panel .idle { color: #4caf50; }
    </style>
</head>
<body>
    <div class="info-panel">
        <div id="status">Status: <span id="rec-status">Idle</span></div>
        <div>Frames: <span id="frame-count">0</span></div>
        <div>Mocap X: <span id="mocap-x">0.000</span>, Y: <span id="mocap-y">0.000</span></div>
    </div>
    <div class="container">
        <div class="camera-box"><h3>Top View</h3><img src="/stream/top" /></div>
        <div class="camera-box"><h3>Side View</h3><img src="/stream/side" /></div>
    </div>
    <script>
        setInterval(() => {
            fetch('/info').then(r => r.json()).then(data => {
                const recStatus = document.getElementById('rec-status');
                recStatus.textContent = data.recording ? 'REC' : 'Idle';
                recStatus.className = data.recording ? 'recording' : 'idle';
                document.getElementById('frame-count').textContent = data.frame_count;
                document.getElementById('mocap-x').textContent = data.mocap_x.toFixed(3);
                document.getElementById('mocap-y').textContent = data.mocap_y.toFixed(3);
            });
        }, 100);
    </script>
</body>
</html>'''
                self.wfile.write(html)

            elif self.path == '/info':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                import json
                with _frame_lock:
                    info = {
                        "recording": _stream_info["is_recording"],
                        "frame_count": _stream_info["frame_count"],
                        "mocap_x": float(_stream_info["mocap_x"]),
                        "mocap_y": float(_stream_info["mocap_y"]),
                    }
                self.wfile.write(json.dumps(info).encode())

            elif self.path == '/stream/top':
                self._send_mjpeg_stream('top')
            elif self.path == '/stream/side':
                self._send_mjpeg_stream('side')
            else:
                self.send_error(404)

        def _send_mjpeg_stream(self, camera):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            nonlocal _streaming_active
            while _streaming_active:
                with _frame_lock:
                    frame = _latest_frame_top if camera == 'top' else _latest_frame_side
                    if frame is not None:
                        frame = frame.copy()
                if frame is not None:
                    try:
                        img = Image.fromarray(frame)
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=80)
                        frame_data = buffer.getvalue()
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame_data))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                    except Exception:
                        break
                time.sleep(0.03)

    def start_mjpeg_server(port=8080):
        nonlocal _streaming_active
        def run_server():
            server = ThreadingHTTPServer(('0.0.0.0', port), MJPEGHandler)
            logger.info(f"HTTP server listening on 0.0.0.0:{port}")
            server.serve_forever()
        _streaming_active = True
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread

    def update_camera_frames(frame_top, frame_side, is_recording=False, frame_count=0, mocap_x=0.0, mocap_y=0.0):
        nonlocal _latest_frame_top, _latest_frame_side
        with _frame_lock:
            if frame_top is not None:
                _latest_frame_top = frame_top.copy()
            if frame_side is not None:
                _latest_frame_side = frame_side.copy()
            _stream_info["is_recording"] = is_recording
            _stream_info["frame_count"] = frame_count
            _stream_info["mocap_x"] = mocap_x
            _stream_info["mocap_y"] = mocap_y

    # --- Load MuJoCo model ---
    XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
    mocap_name = "target_mocap"
    mocap_id = model.body(mocap_name).mocapid[0]

    t_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
    t_qpos_adr = model.jnt_qposadr[model.body_jntadr[t_body_id]]
    is_success = False

    dataset = None

    def init_lerobot_dataset(repo_path):
        import shutil
        features = {
            "observation.images.cam_top": {"dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]},
            "observation.images.cam_side": {"dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]},
            "observation.state": {"dtype": "float32", "shape": (5,), "names": act_names},
            "action": {"dtype": "float32", "shape": (2,), "names": ["mocap_x", "mocap_y"]}
        }
        meta_path = os.path.join(repo_path, "meta")
        tasks_path = os.path.join(meta_path, "tasks.parquet")
        if os.path.exists(repo_path):
            if os.path.exists(tasks_path):
                logger.info(f"Existing dataset detected, loading: {repo_path}")
                try:
                    return LeRobotDataset(repo_path)
                except Exception as e:
                    logger.error(f"Failed to load existing dataset: {e}")
                    raise
            else:
                logger.warning(f"Folder exists but is not a valid dataset: {repo_path}")
                logger.warning("Removing incomplete dataset folder...")
                shutil.rmtree(repo_path)
                logger.info(f"Removed incomplete dataset folder: {repo_path}")
        logger.info(f"Creating new dataset repository: {repo_path}")
        return LeRobotDataset.create(repo_id=repo_path, fps=FPS, robot_type=ROBOT_TYPE, features=features)

    logger.info("Using direct pygame joystick control")
    logger.info("Pre-initializing dataset...")
    try:
        dataset = init_lerobot_dataset(repo_path=str(REPO_ID))
        logger.info("Dataset initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        dataset = None

    def get_mocap_4d_pose(mocap_pos, mocap_quat):
        x, y, z = mocap_pos
        return np.array([x, y], dtype=np.float32)

    def np_allabs(x):
        return np.all(np.abs(x) < TOLERANCE)

    def async_save_to_lerobot(frames_list):
        nonlocal dataset
        if not dataset:
            dataset = init_lerobot_dataset(repo_path=str(REPO_ID))
        if len(frames_list) < 2:
            return
        num_frames = len(frames_list) - 1
        logger.info(f"[Background thread] Start processing new episode ({num_frames} frames)")
        added_count = 0
        for i in range(num_frames):
            curr_frame = frames_list[i]
            target_action = curr_frame["mocap_pose_2d"]
            frame_data = {
                "observation.images.cam_top": curr_frame["cam_top"],
                "observation.images.cam_side": curr_frame["cam_side"],
                "observation.state": curr_frame["state"].astype(np.float32),
                "action": target_action,
                "task": "pushT"
            }
            dataset.add_frame(frame_data)
            added_count += 1
        if added_count > 0:
            dataset.save_episode()
            logger.info(f"[Background thread] Save completed, valid frames: {added_count}, total episodes: {dataset.num_episodes}")
        else:
            logger.warning("[Background thread] Skipped: no valid action changes.")

    def record_toggle():
        nonlocal is_recording, record_buffer, save_thread
        if is_recording:
            is_recording = False
            logger.info("Recording stopped.")
            if len(record_buffer) > 0:
                if save_thread and save_thread.is_alive():
                    logger.info("Waiting for previous save to complete...")
                    save_thread.join()
                save_thread = threading.Thread(target=async_save_to_lerobot, args=(list(record_buffer),), daemon=True)
                save_thread.start()
            record_buffer = []
        else:
            logger.info("Recording started (Recording ON)")
            record_buffer = []
            is_recording = True

    def reset_env():
        nonlocal is_success, is_recording, record_buffer
        is_success = False
        is_recording = False
        record_buffer.clear()
        key_id = 0
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.mocap_pos[:] = model.key_mpos[key_id]
        data.mocap_quat[:] = model.key_mquat[key_id]
        data.qpos[t_qpos_adr:t_qpos_adr+2] = [
            np.random.uniform(0.25 - pos_random_range, 0.25 + pos_random_range),
            np.random.uniform(-pos_random_range, pos_random_range)
        ]
        data.qpos[t_qpos_adr+2] = 0.01
        rad = np.random.uniform(-3.14, 3.14)
        data.qpos[t_qpos_adr+3:t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
        mujoco.mj_forward(model, data)

    def joystick_control():
        nonlocal buttonCooldown
        if not joystick:
            return False
        pygame.event.pump()
        ax0, ax1 = joystick.get_axis(0), joystick.get_axis(1)
        dx = (abs(ax0) > DEADZONE) * ax0 * MOVE_SPEED * model.opt.timestep
        dy = -(abs(ax1) > DEADZONE) * ax1 * MOVE_SPEED * model.opt.timestep
        dz = (joystick.get_button(4) - joystick.get_button(0)) * MOVE_SPEED * model.opt.timestep
        data.mocap_pos[mocap_id] += np.array([dx, dy, dz])
        ax2 = joystick.get_axis(2)
        dr = -(abs(ax2) > DEADZONE) * ax2 * ROT_SPEED * model.opt.timestep
        if abs(dr) > 0:
            q = data.mocap_quat[mocap_id]
            r_curr = R.from_quat([q[1], q[2], q[3], q[0]])
            new_q = (R.from_euler('y', dr) * r_curr).as_quat()
            data.mocap_quat[mocap_id] = [new_q[3], new_q[0], new_q[1], new_q[2]]
        now = time.time()
        if joystick.get_button(3) and (now - buttonCooldown > COOLDOWN_SEC):
            buttonCooldown = now
            reset_env()
        if joystick.get_button(1) and (now - buttonCooldown > COOLDOWN_SEC):
            buttonCooldown = now
            record_toggle()
        return joystick.get_button(11)

    # --- Main loop ---
    renderer = mujoco.Renderer(model, height=OBS_IMAGES_SHAPE[0], width=OBS_IMAGES_SHAPE[1])
    reset_env()
    video_time = 0.0

    server_port = args.stream_port
    start_mjpeg_server(server_port)
    logger.info(f"MJPEG stream server started at http://localhost:{server_port}")

    try:
        subprocess.run(['start', f'http://localhost:{server_port}'], shell=True, check=True)
        logger.info("Camera stream opened in browser!")
    except Exception as e:
        logger.warning(f"Could not open browser: {e}")
        logger.info(f"Please manually open: http://localhost:{server_port}")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                data.ctrl[act_ids] = data.qpos[act_ids]
                joystick_control()
                mujoco.mj_step(model, data)
                ok, dxy, dyaw = check_xy_pose_match(model, data, "T_sign_anchor", "T_block_anchor")
                if ok and not is_success:
                    logger.info(f"Task succeeded! Position error: {dxy:.4f}")
                    is_success = True
                video_time += model.opt.timestep
                if video_time >= VIDEO_STEP:
                    video_time = 0.0
                    renderer.update_scene(data, camera="top_view")
                    img_top = renderer.render().copy()
                    renderer.update_scene(data, camera="side_view")
                    img_side = renderer.render().copy()
                    current_mocap_2d = get_mocap_4d_pose(data.mocap_pos[mocap_id].copy(), data.mocap_quat[mocap_id].copy())
                    if is_recording:
                        record_buffer.append({"cam_top": img_top, "cam_side": img_side, "mocap_pose_2d": current_mocap_2d, "state": data.qpos[act_ids].copy()})
                    update_camera_frames(frame_top=img_top, frame_side=img_side, is_recording=is_recording, frame_count=len(record_buffer), mocap_x=current_mocap_2d[0], mocap_y=current_mocap_2d[1])
                    viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    finally:
        logger.info("Shutting down safely...")
        if save_thread and save_thread.is_alive():
            logger.info("Waiting for the last batch of data to be saved...")
            save_thread.join()
        logger.info("Saving dataset index and releasing resources...")
        _streaming_active = False
        pygame.quit()
        logger.info("Exit complete.")


if __name__ == "__main__":
    main()
