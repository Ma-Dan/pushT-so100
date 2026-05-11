import os
import argparse

# Offscreen rendering configuration
os.environ["MUJOCO_GL"] = "glfw"

import time
import threading
import logging
import sys
import io
import subprocess
import multiprocessing as mp
from http.server import HTTPServer, BaseHTTPRequestHandler
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from helper import *

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Module-level constants ---
DEADZONE = 0.1
OBS_IMAGES_SHAPE = (224, 224, 3)
ROBOT_TYPE = "so100_arm"
TOLERANCE = 0.0001  # Tolerance for action deduplication
COOLDOWN_SEC = 0.3
XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml"

os.environ.setdefault("SDL_JOYSTICK_DEVICE", "/dev/input/js0")

# --- MJPEG streaming globals ---
_latest_frame = None
_frame_lock = threading.Lock()
_streaming_active = True


class MJPEGHandler(BaseHTTPRequestHandler):
    """处理 MJPEG 流请求的 HTTP 处理器"""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = b'''<!DOCTYPE html>
<html>
<head><title>SO-ARM100 Camera Stream</title></head>
<body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;min-height:100vh;">
<img src="/stream" style="max-width:100%;max-height:100vh;">
</body>
</html>'''
            self.wfile.write(html)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            global _streaming_active
            while _streaming_active:
                with _frame_lock:
                    if _latest_frame is not None:
                        try:
                            img = Image.fromarray(_latest_frame)
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
        else:
            self.send_error(404)


def start_mjpeg_server(port=8080):
    """在后台线程启动 MJPEG 服务器"""
    global _streaming_active

    def run_server():
        server = HTTPServer(('localhost', port), MJPEGHandler)
        server.serve_forever()

    _streaming_active = True
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread


def update_camera_frame(frame):
    """更新最新的相机帧（拼接后的图像）"""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()

# Recording state (module-level globals shared across functions)
is_recording = False
record_buffer = []  # Temporary buffer for the current episode
save_thread = None  # Used to track the background saving thread
buttonCooldown = 0.0
is_success = False
dataset = None
pos_random_range = 0.05


def _joystick_proc(state, ready):
    """子进程主循环：在自己的主线程上运行 pygame，避开 mjpython 的线程限制。"""
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("joystick not found")
        ready.set()
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    state['name'] = js.get_name()
    state['num_buttons'] = js.get_numbuttons()
    state['connected'] = True
    print(f"joystick found: {js.get_name()}")
    print(f"button count: {js.get_numbuttons()}")
    ready.set()
    try:
        while True:
            pygame.event.pump()
            try:
                state['axis_0'] = js.get_axis(0)
                state['axis_1'] = js.get_axis(1)
                state['axis_2'] = js.get_axis(2)
                state['button_0'] = js.get_button(0)
                state['button_1'] = js.get_button(1)
                state['button_3'] = js.get_button(3)
                state['button_4'] = js.get_button(4)
                state['button_11'] = js.get_button(11)
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                # 父进程退出时 Manager 已关闭，安静地结束子进程
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


class JoystickController:
    """Joystick控制器：pygame 在子进程里跑（macOS + mjpython 主线程冲突的解决办法）"""
    def __init__(self):
        ctx = mp.get_context('spawn')
        # 用同 conda 环境里的普通 python，而不是 mjpython，避免子进程里也被搬到次线程
        plain_python = os.path.join(os.path.dirname(sys.executable), 'python')
        if os.path.exists(plain_python):
            ctx.set_executable(plain_python)
        self.manager = ctx.Manager()
        self.state = self.manager.dict({
            'axis_0': 0.0, 'axis_1': 0.0, 'axis_2': 0.0,
            'button_0': 0, 'button_1': 0, 'button_3': 0,
            'button_4': 0, 'button_11': 0,
            'connected': False, 'name': '', 'num_buttons': 0,
        })
        self.ready = ctx.Event()
        self.process = ctx.Process(target=_joystick_proc, args=(self.state, self.ready), daemon=True)
        self.process.start()
        self.ready.wait(timeout=10)

    def is_connected(self):
        return bool(self.state.get('connected', False))

    def get_axis(self, idx):
        return float(self.state.get(f'axis_{idx}', 0.0))

    def get_button(self, idx):
        return int(self.state.get(f'button_{idx}', 0))

    def cleanup(self):
        try:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        except Exception:
            pass
        try:
            self.manager.shutdown()
        except Exception:
            pass


# --- Initialize LeRobot dataset ---
def init_lerobot_dataset(repo_path, fps, act_names):
    features = {
        "observation.images.cam_top": {
            "dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]
        },
        "observation.images.cam_side": {
            "dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]
        },
        "observation.state": {
            "dtype": "float32", "shape": (5,), "names": act_names
        },
        "action": {
            "dtype": "float32", "shape": (2,), "names": ["mocap_x", "mocap_y"]
        }
    }

    if os.path.exists(repo_path):
        logger.info(f"Existing dataset detected, loading: {repo_path}")
        return LeRobotDataset(repo_path)
    else:
        logger.info(f"Dataset not found, creating new repository: {repo_path}")
        return LeRobotDataset.create(
            repo_id=repo_path,
            fps=fps,
            robot_type=ROBOT_TYPE,
            features=features
        )


def get_mocap_4d_pose(mocap_pos, mocap_quat):
    """Extract x, y, z and Y-axis rotation from mocap data."""
    x, y, z = mocap_pos
    return np.array([x, y], dtype=np.float32)


# Utility function: check whether all absolute values are below tolerance
def np_allabs(x):
    return np.all(np.abs(x) < TOLERANCE)


def async_save_to_lerobot(frames_list, repo_id, fps, act_names):
    """Process data and write it to disk in a separate thread."""
    global dataset
    if not dataset:
        dataset = init_lerobot_dataset(repo_path=repo_id, fps=fps, act_names=act_names)
    if len(frames_list) < 2:
        return

    num_frames = len(frames_list) - 1
    logger.info(f"[Background thread] Start processing new episode ({num_frames} frames)")
    added_count = 0
    last_action = None

    for i in range(num_frames):
        curr_frame = frames_list[i]
        target_action = curr_frame["mocap_pose_2d"]

        # --- Action deduplication ---
        if last_action is not None and np_allabs(target_action - last_action):
            continue
        frame_data = {
            "observation.images.cam_top": curr_frame["cam_top"],
            "observation.images.cam_side": curr_frame["cam_side"],
            "observation.state": curr_frame["state"].astype(np.float32),
            "action": target_action,
            "task": "pushT"
        }
        dataset.add_frame(frame_data)
        added_count += 1
        last_action = target_action

    if added_count > 0:
        dataset.save_episode()
        logger.info(f"[Background thread] Save completed, valid frames: {added_count}, total episodes: {dataset.num_episodes}")
    else:
        logger.warning("[Background thread] Skipped: no valid action changes.")


# --- Joystick and recording control ---
def record_toggle(repo_id, fps, act_names):
    global is_recording, record_buffer, save_thread
    if is_recording:
        is_recording = False
        logger.info(f"[DEBUG] record_toggle: STOP — buffered frames={len(record_buffer)}, "
                    f"repo_id={repo_id}, fps={fps}, is_success={is_success}")
        if len(record_buffer) > 0:
            # Start a new thread for saving to avoid blocking the main loop
            save_thread = threading.Thread(
                target=async_save_to_lerobot,
                args=(list(record_buffer), repo_id, fps, act_names),
                daemon=True
            )
            save_thread.start()
            logger.info(f"[DEBUG] record_toggle: save_thread started (alive={save_thread.is_alive()})")
        else:
            logger.warning("[DEBUG] record_toggle: STOP but buffer is empty, nothing to save")
        record_buffer = []
    else:
        logger.info(f"[DEBUG] record_toggle: START — repo_id={repo_id}, fps={fps}, act_names={act_names}")
        record_buffer = []
        is_recording = True


def reset_env(model, data, t_qpos_adr):
    global is_success, is_recording, record_buffer, pos_random_range
    is_success = False
    is_recording = False
    record_buffer.clear()
    key_id = 0

    # Reset model state
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.mocap_pos[:] = model.key_mpos[key_id]
    data.mocap_quat[:] = model.key_mquat[key_id]

    # Randomize object position
    data.qpos[t_qpos_adr:t_qpos_adr+2] = [
        np.random.uniform(0.25 - pos_random_range, 0.25 + pos_random_range),
        np.random.uniform(-pos_random_range, pos_random_range)
    ]
    data.qpos[t_qpos_adr+2] = 0.01
    rad = np.random.uniform(-3.14, 3.14)
    data.qpos[t_qpos_adr+3:t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
    mujoco.mj_forward(model, data)


def joystick_control(controller, model, data, mocap_id, move_speed, rot_speed,
                     repo_id, fps, act_names, t_qpos_adr):
    global buttonCooldown
    if not controller.is_connected():
        return False

    # Position control (left stick)
    ax0, ax1 = controller.get_axis(0), controller.get_axis(1)
    dx = (abs(ax0) > DEADZONE) * ax0 * move_speed * model.opt.timestep
    dy = -(abs(ax1) > DEADZONE) * ax1 * move_speed * model.opt.timestep
    dz = (controller.get_button(4) - controller.get_button(0)) * move_speed * model.opt.timestep
    data.mocap_pos[mocap_id] += np.array([dx, dy, dz])

    # Rotation control (right stick)
    ax2 = controller.get_axis(2)
    dr = -(abs(ax2) > DEADZONE) * ax2 * rot_speed * model.opt.timestep
    if abs(dr) > 0:
        q = data.mocap_quat[mocap_id]
        r_curr = R.from_quat([q[1], q[2], q[3], q[0]])
        new_q = (R.from_euler('y', dr) * r_curr).as_quat()
        data.mocap_quat[mocap_id] = [new_q[3], new_q[0], new_q[1], new_q[2]]

    now = time.time()

    # X button (3): reset environment
    if controller.get_button(3) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        reset_env(model, data, t_qpos_adr)

    # B button (1): toggle recording
    if controller.get_button(1) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        record_toggle(repo_id, fps, act_names)

    return controller.get_button(11)  # Start button exits


def main():
    global save_thread, dataset, is_success, is_recording, record_buffer

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
    args = parser.parse_args()

    move_speed = args.move_speed
    rot_speed = args.rot_speed
    fps = args.fps
    video_step = 1.0 / fps
    repo_id = Path(args.repo_id).absolute()

    # --- Load MuJoCo model ---
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Get joint and mocap IDs
    act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
    act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
    mocap_name = "target_mocap"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Address of target object T_block
    t_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
    t_qpos_adr = model.jnt_qposadr[model.body_jntadr[t_body_id]]

    # --- Initialize joystick subprocess ---
    controller = JoystickController()
    if not controller.is_connected():
        logger.error("No joystick detected")
        controller.cleanup()
        return

    # --- Main loop ---
    renderer = mujoco.Renderer(model, height=OBS_IMAGES_SHAPE[0], width=OBS_IMAGES_SHAPE[1])
    reset_env(model, data, t_qpos_adr)
    video_time = 0.0

    # 启动 MJPEG 流服务器
    server_port = 8080
    start_mjpeg_server(server_port)
    logger.info(f"MJPEG stream server started at http://localhost:{server_port}")
    try:
        subprocess.run(['open', f'http://localhost:{server_port}'], check=True)
    except Exception as e:
        logger.warning(f"Could not open browser: {e}. Please open http://localhost:{server_port} manually.")

    logger.info("Control ready: X=reset, B=toggle recording, Start=exit")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # IK simulation and physics stepping
                data.ctrl[act_ids] = data.qpos[act_ids]
                if joystick_control(controller, model, data, mocap_id,
                                    move_speed, rot_speed,
                                    repo_id, fps, act_names, t_qpos_adr):
                    break
                mujoco.mj_step(model, data)

                # Check task success
                ok, dxy, dyaw = check_xy_pose_match(model, data, "T_sign_anchor", "T_block_anchor")
                if ok and not is_success:
                    is_success = True
                    logger.info(f"[DEBUG] FINISH — Task succeeded! dxy={dxy:.4f}, dyaw={dyaw:.4f}, "
                                f"is_recording={is_recording}, buffered frames={len(record_buffer)}, "
                                f"mocap_pos={data.mocap_pos[mocap_id].tolist()}, "
                                f"t_block_xy={data.qpos[t_qpos_adr:t_qpos_adr+2].tolist()}")

                # Visual rendering and data sampling
                video_time += model.opt.timestep
                if video_time >= video_step:
                    video_time = 0.0

                    renderer.update_scene(data, camera="top_view")
                    img_top = renderer.render().copy()
                    renderer.update_scene(data, camera="side_view")
                    img_side = renderer.render().copy()

                    # Get current mocap 2D pose
                    current_mocap_2d = get_mocap_4d_pose(
                        data.mocap_pos[mocap_id].copy(),
                        data.mocap_quat[mocap_id].copy()
                    )

                    if is_recording:
                        record_buffer.append({
                            "cam_top": img_top,
                            "cam_side": img_side,
                            "mocap_pose_2d": current_mocap_2d,
                            "state": data.qpos[act_ids].copy()
                        })

                    # 拼接顶视图与侧视图后推送到 MJPEG 流
                    combined = np.concatenate([img_top, img_side], axis=1)
                    update_camera_frame(combined)

                    viewer.sync()

                # Maintain physics rate
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    finally:
        # Cleanup logic when the program exits
        logger.info("Shutting down safely...")

        global _streaming_active
        _streaming_active = False

        if save_thread and save_thread.is_alive():
            logger.info("Waiting for the last batch of data to be saved...")
            save_thread.join()

        logger.info("Saving dataset index and releasing resources...")
        del dataset

        controller.cleanup()
        logger.info("Exit complete.")


if __name__ == "__main__":
    main()
