from pathlib import Path
import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import math
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLA Policy on LeRobot dataset")

    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/baqian/qba/pusht/NewData3.9-ee-2d-pos",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pusht_smolvla",
        help="Directory to save logs and checkpoints",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--training-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=5000, help="Checkpoint save frequency")

    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")

    parser.add_argument("--n-obs-steps", type=int, default=1, help="Number of observation steps")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk size / prediction horizon")
    parser.add_argument("--n-action-steps", type=int, default=50, help="Number of action steps to execute per inference")

    # SmolVLA specific arguments
    parser.add_argument("--optimizer-lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-10, help="Weight decay")
    parser.add_argument("--optimizer-grad-clip-norm", type=float, default=10.0, help="Gradient clipping norm")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--scheduler-decay-steps", type=int, default=30000, help="Decay steps")
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6, help="Decay learning rate")

    parser.add_argument("--vlm-model-name", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", help="VLM model name")
    parser.add_argument("--load-vlm-weights", action="store_true", help="Load VLM weights")
    parser.add_argument("--freeze-vision-encoder", action="store_true", default=True, help="Freeze vision encoder")
    parser.add_argument("--train-expert-only", action="store_true", default=True, help="Train expert only")
    parser.add_argument("--train-state-proj", action="store_true", default=True, help="Train state projection")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--use-peft", action="store_true", help="Use PEFT for fine-tuning")

    parser.add_argument("--max-state-dim", type=int, default=32, help="Maximum state dimension")
    parser.add_argument("--max-action-dim", type=int, default=32, help="Maximum action dimension")
    parser.add_argument("--resize-imgs-with-padding", type=int, nargs=2, default=[512, 512], help="Resize images with padding (height, width)")
    parser.add_argument("--num-expert-layers", type=int, default=-1, help="Number of expert layers (-1 for default)")
    parser.add_argument("--num-vlm-layers", type=int, default=16, help="Number of VLM layers")
    parser.add_argument("--expert-width-multiplier", type=float, default=0.75, help="Expert width multiplier")
    parser.add_argument("--attention-mode", type=str, default="cross_attn", help="Attention mode")
    parser.add_argument("--tokenizer-max-length", type=int, default=48, help="Tokenizer max length")

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


def build_lr_lambda(warmup_steps: int, decay_steps: int, decay_lr: float, base_lr: float):
    """Build learning rate lambda function for SmolVLA schedule."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
        # Linear decay to decay_lr
        return max(decay_lr / base_lr, 1.0 - progress * (1.0 - decay_lr / base_lr))

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

    # Resize images to SmolVLA expected size
    resize_imgs = tuple(args.resize_imgs_with_padding)
    for key in image_keys:
        if key in features:
            features[key].shape = (3, resize_imgs[0], resize_imgs[1])

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features and k not in mask_keys}

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        device=str(device),
        use_amp=args.use_amp,
        use_peft=args.use_peft,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        resize_imgs_with_padding=resize_imgs,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=args.train_expert_only,
        train_state_proj=args.train_state_proj,
        optimizer_lr=args.optimizer_lr,
        optimizer_weight_decay=args.optimizer_weight_decay,
        optimizer_grad_clip_norm=args.optimizer_grad_clip_norm,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_decay_steps=args.scheduler_decay_steps,
        scheduler_decay_lr=args.scheduler_decay_lr,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=args.load_vlm_weights,
        num_expert_layers=args.num_expert_layers,
        num_vlm_layers=args.num_vlm_layers,
        expert_width_multiplier=args.expert_width_multiplier,
        attention_mode=args.attention_mode,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    # Print key training parameters
    print("=" * 60)
    print("SmolVLA Training Configuration:")
    print(f"  chunk_size: {args.chunk_size}")
    print(f"  n_action_steps: {args.n_action_steps}")
    print(f"  optimizer_lr: {args.optimizer_lr}")
    print(f"  scheduler_warmup_steps: {args.scheduler_warmup_steps}")
    print(f"  scheduler_decay_steps: {args.scheduler_decay_steps}")
    print(f"  training_steps: {args.training_steps}")
    print(f"  vlm_model_name: {args.vlm_model_name}")
    print(f"  freeze_vision_encoder: {args.freeze_vision_encoder}")
    print(f"  train_expert_only: {args.train_expert_only}")
    print(f"  use_amp: {args.use_amp}")
    print(f"  use_peft: {args.use_peft}")
    print("=" * 60)

    # Build delta_timestamps for dataset
    # SmolVLA uses n_obs_steps=1 and chunk_size for action prediction
    delta_timestamps = {}
    for k in input_features.keys():
        delta_timestamps[k] = [0]  # Current observation only
    for k in output_features.keys():
        delta_timestamps[k] = [i / dataset_metadata.fps for i in range(cfg.chunk_size)]

    dataset = LeRobotDataset(repo_id=repo_id, root=root_path, delta_timestamps=delta_timestamps)

    policy = SmolVLAPolicy(cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    # SmolVLA uses its own optimizer configuration
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.optimizer_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.optimizer_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(
            args.scheduler_warmup_steps,
            args.scheduler_decay_steps,
            args.scheduler_decay_lr,
            args.optimizer_lr,
        ),
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
    print(f"Training SmolVLA started. Saving to {output_directory}")

    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == "cuda" else None

    while not done:
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            batch = preprocessor(batch)

            # For SmolVLA, squeeze the time dimension from observation features
            # since n_obs_steps=1 and delta_timestamps returns shape (B, 1, *)
            # but SmolVLA expects (B, *)
            for key in batch:
                if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 3:
                    # Check if this is an observation feature (not action)
                    if key not in output_features and not key.endswith("_is_pad"):
                        # Squeeze the time dimension if it's 1
                        if batch[key].shape[1] == 1:
                            batch[key] = batch[key].squeeze(1)

            if args.use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss, _ = policy.forward(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.optimizer_grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _ = policy.forward(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.optimizer_grad_clip_norm)
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
