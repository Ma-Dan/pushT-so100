from pathlib import Path
import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import math
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT Policy on LeRobot dataset")

    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/baqian/qba/pusht/NewData3.9-ee-2d-pos",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pusht_act",
        help="Directory to save logs and checkpoints",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--training-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=5000, help="Checkpoint save frequency")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")

    parser.add_argument("--n-obs-steps", type=int, default=1, help="Number of observation steps (must be 1 for ACT)")
    # chunk_size: 增大以学习更平滑的动作序列，原始 ACT 论文使用 100
    parser.add_argument("--chunk-size", type=int, default=30, help="Action chunk size / prediction horizon (larger = smoother actions)")
    # n_action_steps: 每次推理执行的动作步数，应小于 chunk_size
    parser.add_argument("--n-action-steps", type=int, default=10, help="Number of action steps to execute per inference")
    parser.add_argument("--vision-backbone", type=str, default="resnet18", help="Vision backbone")

    # ACT specific arguments
    parser.add_argument("--n-encoder-layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--n-decoder-layers", type=int, default=7, help="Number of decoder layers (7 as in original ACT)")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim-feedforward", type=int, default=3200, help="Feedforward dimension")
    parser.add_argument("--dim-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--latent-dim", type=int, default=32, help="VAE latent dimension")
    parser.add_argument("--n-vae-encoder-layers", type=int, default=4, help="Number of VAE encoder layers")
    # kl_weight: 降低以减少 VAE 对动作抖动的影响
    parser.add_argument("--kl-weight", type=float, default=5.0, help="KL divergence weight (lower = more stable actions)")
    # temporal_ensemble_coeff: 启用时序集成来平滑动作输出
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None, help="Temporal ensemble coefficient for smoothing actions (e.g., 0.01)")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cuda, mps, cpu, or auto (auto-detect best available)",
    )

    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=["observation.images.cam_top", "observation.images.cam_side"],
        help="Image feature keys",
    )
    parser.add_argument(
        "--mask-keys",
        nargs="*",
        default=[],
        help="Mask feature keys to exclude from input features",
    )

    return parser.parse_args()


def build_lr_lambda(warmup_steps: int, training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def main():
    args = parse_args()

    output_directory = Path(args.output_dir)
    data_dir = Path(args.data_path)
    checkpoints_dir = output_directory / f"checkpoints_{time.strftime('%Y-%m-%d_%H:%M')}"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_directory / f"runs_{time.strftime('%Y-%m-%d_%H:%M')}"))

    # Auto-detect device if needed
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Auto-detected device: {device}")
    else:
        device = torch.device(args.device)

    # For datasets in lerobot cache format: root/repo_id (e.g., /path/to/lerobot/qian1dqs/so100-pusht)
    # We need to find the correct repo_id and root from the data_path
    # The repo_id should be the last two parts of the path (org/dataset_name)
    # and root should be everything before that

    # Check if this looks like a cached lerobot dataset (has org/dataset format)
    parts = data_dir.parts
    if len(parts) >= 2:
        # Assume last two parts form the repo_id: org/dataset_name
        repo_id = f"{parts[-2]}/{parts[-1]}"
        root_path = Path(*parts[:-2])
    else:
        # Fallback: use directory name as repo_id
        repo_id = data_dir.name
        root_path = data_dir.parent

    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root_path)
    features = dataset_to_policy_features(dataset_metadata.features)

    image_keys = args.image_keys
    mask_keys = args.mask_keys

    for key in image_keys:
        if key in features:
            features[key].shape = (3, 224, 224)

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features and k not in mask_keys}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        vision_backbone=args.vision_backbone,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dim_model=args.dim_model,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        kl_weight=args.kl_weight,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        device=str(device),  # Set device in config so preprocessor uses correct device
    )

    # 打印关键训练参数
    print("=" * 60)
    print("ACT Training Configuration:")
    print(f"  chunk_size: {args.chunk_size}")
    print(f"  n_action_steps: {args.n_action_steps}")
    print(f"  kl_weight: {args.kl_weight}")
    print(f"  temporal_ensemble_coeff: {args.temporal_ensemble_coeff}")
    print(f"  training_steps: {args.training_steps}")
    print(f"  lr: {args.lr}")
    print("=" * 60)

    # Build delta_timestamps for dataset
    # ACT's observation_delta_indices returns None, so we use [0] for current observation
    # ACT's action_delta_indices returns list(range(chunk_size))
    delta_timestamps = {}
    for k in input_features.keys():
        delta_timestamps[k] = [0]  # Current observation only
    for k in output_features.keys():
        delta_timestamps[k] = [i / dataset_metadata.fps for i in cfg.action_delta_indices]

    dataset = LeRobotDataset(repo_id=repo_id, root=root_path, delta_timestamps=delta_timestamps)

    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(args.warmup_steps, args.training_steps),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    step = 0
    done = False
    print(f"Training ACT started. Saving to {output_directory}")

    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            batch = preprocessor(batch)

            # For ACT, squeeze the time dimension from observation features
            # since n_obs_steps=1 and delta_timestamps returns shape (B, 1, *)
            # but ACT expects (B, *)
            for key in batch:
                if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 3:
                    # Check if this is an observation feature (not action)
                    if key not in output_features and not key.endswith("_is_pad"):
                        # Squeeze the time dimension if it's 1
                        if batch[key].shape[1] == 1:
                            batch[key] = batch[key].squeeze(1)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("LR/train", current_lr, step)
                print(f"step: {step} loss: {loss.item():.3f} lr: {current_lr:.6f}")

            if step > 0 and step % args.save_freq == 0:
                step_ckpt_dir = checkpoints_dir / f"step_{step}"
                step_ckpt_dir.mkdir(parents=True, exist_ok=True)

                policy.save_pretrained(step_ckpt_dir)
                preprocessor.save_pretrained(step_ckpt_dir)
                postprocessor.save_pretrained(step_ckpt_dir)
                print(f"Checkpoint saved at step {step}")

            step += 1
            if step >= args.training_steps:
                done = True
                break

    final_dir = checkpoints_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    writer.close()
    print(f"Training finished. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
