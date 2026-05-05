# ============================================================================
# IMPORTANT: 此脚本必须使用 mjpython 运行以解决 macOS NSWindow 线程问题
# 运行方式: mjpython interactive_gym_smolvla.py --policy <checkpoint_path> --dataset <dataset_path>
# ============================================================================
import os
# 使用 glfw 渲染
os.environ.setdefault('MUJOCO_GL', 'glfw')

import mujoco
import mujoco.viewer
import torch
import time
import numpy as np
import subprocess
import threading
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import sys
import os.path as osp
from pathlib import Path

# 当前文件在 src 目录下，直接导入同目录的模块
from env_gym_ee import PushT

# 用于 MJPEG 流的全局变量
_latest_frame = None
_frame_lock = threading.Lock()
_streaming_active = True


class MJPEGHandler(BaseHTTPRequestHandler):
    """处理 MJPEG 流请求的 HTTP 处理器"""

    def log_message(self, format, *args):
        """禁用默认的日志输出"""
        pass

    def do_GET(self):
        if self.path == '/':
            # 返回简单的 HTML 页面
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
            # MJPEG 流
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            global _streaming_active
            while _streaming_active:
                with _frame_lock:
                    if _latest_frame is not None:
                        try:
                            # 将 numpy 数组转换为 JPEG
                            img = Image.fromarray(_latest_frame)
                            buffer = io.BytesIO()
                            img.save(buffer, format='JPEG', quality=80)
                            frame_data = buffer.getvalue()

                            # 发送 MJPEG 帧
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(frame_data))
                            self.end_headers()
                            self.wfile.write(frame_data)
                            self.wfile.write(b'\r\n')
                        except Exception:
                            break
                time.sleep(0.03)  # 约 30 FPS
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
    """更新相机帧"""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()


class DummyViewer:
    """虚拟 viewer，用于在 mujoco.viewer 不可用时提供替代方案"""
    def __init__(self, max_steps=1000):
        self._running = True
        self._step_count = 0
        self._max_steps = max_steps

    def is_running(self):
        if self._step_count >= self._max_steps:
            return False
        return self._running

    def sync(self):
        self._step_count += 1

    def close(self):
        self._running = False


class RandomPolicy:
    """简单的随机策略，用于测试环境"""
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, observation):
        return self.action_space.sample()

    def reset(self):
        pass


class SmolVLAPolicyWrapper:
    """SmolVLA Policy 包装器，用于与 PushT 环境交互"""

    def __init__(self, policy, preprocessor, postprocessor, dataset_metadata, device, image_keys=None):
        """
        Args:
            policy: SmolVLAPolicy 模型
            preprocessor: 预处理器
            postprocessor: 后处理器
            dataset_metadata: 数据集元数据
            device: 推理设备
            image_keys: 图像特征键列表
        """
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataset_metadata = dataset_metadata
        self.device = device
        self.image_keys = image_keys or ["observation.images.cam_top", "observation.images.cam_side"]

        # Get default task string from dataset metadata (task_index 0)
        tasks_df = dataset_metadata.tasks
        self.default_task = list(tasks_df.index)[0] if len(tasks_df) > 0 else ""

        # 导入 build_inference_frame
        from lerobot.policies.utils import build_inference_frame
        self.build_inference_frame = build_inference_frame

    def __call__(self, observation):
        """
        根据观测选择动作

        Args:
            observation: 环境观测字典

        Returns:
            action: numpy 数组，形状为 (action_dim,)
        """
        with torch.no_grad():
            # 构建推理帧
            obs_frame = self.build_inference_frame(
                observation=observation,
                ds_features=self.dataset_metadata.features,
                device=self.device,
                task=self.default_task,
            )

            # 预处理
            obs_tensor = self.preprocessor(obs_frame)

            # 对于 SmolVLA，需要处理观测特征的时间维度
            # SmolVLA 期望的观测数据形状为 (B, feature_dim)，而不是 (B, 1, feature_dim)
            for key in obs_tensor:
                if isinstance(obs_tensor[key], torch.Tensor) and obs_tensor[key].dim() >= 3:
                    # 检查是否是 action padding 标记
                    if not key.endswith("_is_pad") and key != "action":
                        # 如果时间维度为 1，则压缩它
                        if obs_tensor[key].shape[1] == 1:
                            obs_tensor[key] = obs_tensor[key].squeeze(1)

            # 策略推理
            actions_sequence = self.policy.select_action(obs_tensor)

            # 后处理
            actions_sequence = self.postprocessor(actions_sequence)

            # 返回预测序列的第一个动作
            return actions_sequence[0].cpu().numpy()

    def reset(self):
        """重置策略内部状态"""
        self.policy.reset()


def load_smolvla_policy(policy_path, dataset_path, device, image_keys=None):
    """
    加载 SmolVLA Policy 模型

    Args:
        policy_path: 策略 checkpoint 路径
        dataset_path: 数据集路径
        device: 推理设备
        image_keys: 图像特征键列表

    Returns:
        policy_wrapper: SmolVLAPolicyWrapper 实例
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy_path = Path(policy_path)
    dataset_path = Path(dataset_path)

    print(f"Loading SmolVLA policy from: {policy_path}")
    print(f"Loading dataset metadata from: {dataset_path}")

    # 解析数据集路径获取 repo_id 和 root
    parts = dataset_path.parts
    if len(parts) >= 2:
        repo_id = f"{parts[-2]}/{parts[-1]}"
        root_path = Path(*parts[:-2])
    else:
        repo_id = dataset_path.name
        root_path = dataset_path.parent

    # 加载模型
    policy = SmolVLAPolicy.from_pretrained(policy_path.absolute())
    policy.eval()
    policy.to(device)

    # 加载数据集元数据
    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root_path)

    # 创建预处理器和后处理器
    # 注意：需要传递预处理器覆盖配置来设置正确的设备
    preprocessor_overrides = {
        "device_processor": {"device": str(device)},
    }

    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_metadata.stats,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )
    except TypeError:
        # 如果 make_pre_post_processors 不支持 preprocessor_overrides 参数
        # 尝试只传递必要参数
        print("Warning: make_pre_post_processors doesn't support preprocessor_overrides, trying without it...")
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_metadata.stats,
            pretrained_path=policy_path,
        )

    return SmolVLAPolicyWrapper(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_metadata=dataset_metadata,
        device=device,
        image_keys=image_keys
    )


def make_env_and_policy(xml_path=None, policy_path=None, dataset_path=None, image_keys=None):
    """
    初始化本地 SO-ARM100 PushT 环境和 SmolVLA 策略

    Args:
        xml_path: MuJoCo XML 文件路径，默认使用 human_env.xml
        policy_path: 策略模型路径（可选）
        dataset_path: 数据集路径（可选，加载策略时需要）
        image_keys: 图像特征键列表
    """
    # 默认使用本地的 human_env.xml
    # 注意：XML 文件在项目根目录下，需要从 src 目录向上一级
    if xml_path is None:
        project_root = osp.dirname(osp.dirname(__file__))  # 从 src 目录向上一级到项目根目录
        xml_path = osp.join(
            project_root,
            'chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100',
            'human_env.xml'
        )

    print(f"Loading environment from: {xml_path}")

    # 创建 PushT 环境
    env = PushT(xml_path=xml_path, render_mode='rgb_array')

    # 确定推理设备
    # 优先使用 MPS (Apple Silicon GPU) 或 CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 创建策略
    if policy_path is not None:
        if dataset_path is None:
            print("Warning: dataset_path is required when loading a policy. Using random policy instead.")
            policy = RandomPolicy(env.action_space)
        else:
            try:
                policy = load_smolvla_policy(policy_path, dataset_path, device, image_keys)
                print("SmolVLA Policy loaded successfully!")
            except Exception as e:
                print(f"Error loading policy: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to random policy.")
                policy = RandomPolicy(env.action_space)
    else:
        print("Using random policy for testing")
        policy = RandomPolicy(env.action_space)

    return env, policy


def main(env, policy):
    """
    运行主要的交互仿真循环
    """
    print("Starting interactive simulation with SmolVLA policy...")

    # 启动 MuJoCo passive viewer
    # 注意：必须使用 mjpython 运行此脚本，以确保 GLFW 在主线程初始化
    try:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        print("MuJoCo viewer launched successfully.")
    except Exception as e:
        print(f"Error launching MuJoCo viewer: {e}")
        print("Falling back to headless mode.")
        viewer = DummyViewer()

    # 启动 MJPEG 流服务器
    server_port = 8080
    start_mjpeg_server(server_port)
    print(f"MJPEG stream server started at http://localhost:{server_port}")

    # 在浏览器中打开
    try:
        subprocess.run(['open', f'http://localhost:{server_port}'], check=True)
        print("Camera stream opened in browser!")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"Please manually open: http://localhost:{server_port}")

    # 获取环境中的 camera 名称列表
    model = env.model
    camera_names = []
    if hasattr(model, 'camera') and hasattr(model.camera, 'names'):
        camera_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                        for name in model.camera.names if name]
    print(f"Available cameras: {camera_names if camera_names else 'default camera'}")

    # 重置环境获取初始观测
    observation, info = env.reset(seed=42)
    print(f"Initial observation keys: {list(observation.keys())}")
    print(f"Action space: {env.action_space}")

    step_count = 0
    episode_count = 0

    # 性能统计
    inference_times = []
    frame_times = []

    print("Running... (Close viewer window to exit)")

    # 主循环
    while viewer.is_running():
        frame_start = time.time()

        try:
            # 使用策略选择动作（计时）
            inference_start = time.time()
            action = policy(observation)
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)

            # 调试：打印终止条件信息
            if step_count % 10 == 0:
                print(f"Step {step_count}: dxy={info.get('dxy', 'N/A'):.4f}, dyaw={info.get('dyaw', 'N/A'):.2f}, terminated={terminated}")

            step_count += 1

            # 获取渲染图像并更新 MJPEG 流
            try:
                camera_image = env.render()
                if camera_image is not None:
                    update_camera_frame(camera_image)
            except Exception as render_error:
                pass

            # 同步 viewer
            viewer.sync()

            # 如果 episode 结束，重置环境
            if terminated or truncated:
                episode_count += 1
                # 计算平均推理时间
                avg_inference = np.mean(inference_times) if inference_times else 0
                print(f"Episode {episode_count} finished at step {step_count}. "
                      f"Reward: {reward:.2f}, Avg inference: {avg_inference*1000:.1f}ms")
                if info:
                    print(f"  Info: dxy={info.get('dxy', 'N/A'):.4f}, dyaw={info.get('dyaw', 'N/A'):.2f}")

                observation, info = env.reset(seed=42 + episode_count)
                policy.reset()
                step_count = 0
                inference_times = []  # 重置统计
                viewer.sync()

            # 记录帧时间
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # 每 100 步打印一次性能统计
            if step_count % 100 == 0:
                avg_inference = np.mean(inference_times[-100:]) if len(inference_times) >= 100 else np.mean(inference_times)
                avg_frame = np.mean(frame_times[-100:]) if len(frame_times) >= 100 else np.mean(frame_times)
                fps = 1.0 / avg_frame if avg_frame > 0 else 0
                print(f"Step {step_count}: FPS={fps:.1f}, Avg inference={avg_inference*1000:.1f}ms")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    viewer.close()
    env.close()

    # 停止 MJPEG 流
    global _streaming_active
    _streaming_active = False

    print("Viewer closed. Exiting.")


if __name__ == "__main__":
    # These settings can improve performance on NVIDIA GPUs.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ============================================================================
    # IMPORTANT for macOS users:
    # This script MUST be run with `mjpython` instead of regular `python`:
    #
    #   mjpython interactive_gym_smolvla.py
    #
    # The `mjpython` command ensures that all UI operations (including GLFW window
    # creation) happen on the main thread, which is required by macOS.
    #
    # If you don't have mjpython, you can install MuJoCo which includes it:
    #   pip install mujoco
    #
    # Then find mjpython at: <python_env>/bin/mjpython
    # ============================================================================

    import argparse
    parser = argparse.ArgumentParser(description='Interactive SO-ARM100 PushT Environment with SmolVLA Policy')
    parser.add_argument('--xml', type=str, default=None,
                        help='Path to MuJoCo XML file (default: human_env.xml)')
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to SmolVLA policy checkpoint directory (e.g., outputs/pusht_smolvla/final_model)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset directory (required when loading policy, e.g., /path/to/lerobot/qian1dqs/so100-pusht)')
    parser.add_argument('--image-keys', nargs='*', default=["observation.images.cam_top", "observation.images.cam_side"],
                        help='Image feature keys to use (default: observation.images.cam_top observation.images.cam_side)')
    args = parser.parse_args()

    # 创建环境和策略
    env, policy = make_env_and_policy(
        xml_path=args.xml,
        policy_path=args.policy,
        dataset_path=args.dataset,
        image_keys=args.image_keys
    )

    # 运行主仿真
    main(env, policy)
