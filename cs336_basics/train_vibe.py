import argparse
import contextlib
import glob
import math
import os
import time

import config
import NeuralNets
import torch
import numpy as np

# Run `bash cs336_basics/train_runs.sh`


try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-basics")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1)
    parser.add_argument("--wandb_watch_freq", type=int, default=10)
    parser.add_argument("--no_wandb_watch", action="store_true")

    # Profiler
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_dir", type=str, default="wandb_profiler")
    parser.add_argument("--profile_wait", type=int, default=1)
    parser.add_argument("--profile_warmup", type=int, default=1)
    parser.add_argument("--profile_active", type=int, default=3)
    parser.add_argument("--profile_repeat", type=int, default=1)

    # Common shortcuts
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train_file_path", type=str, default=None)
    parser.add_argument("--out_checkpoint_path", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=None)

    args, unknown = parser.parse_known_args()
    return args, unknown


def coerce(value, old_value):
    if isinstance(old_value, bool):
        return value.lower() in {"1", "true", "yes", "y"}
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(value)
    if isinstance(old_value, float):
        return float(value)
    if isinstance(old_value, torch.device):
        return torch.device(value)
    return value


def set_cfg_attr(cfg, name, value):
    if not hasattr(cfg, name):
        raise ValueError(f"Unknown config field: {name}")
    setattr(cfg, name, coerce(value, getattr(cfg, name)))


def apply_cli(args, unknown, model_cfg, optimizer_cfg, training_cfg):
    if args.steps is not None:
        training_cfg.steps = args.steps
    if args.batch_size is not None:
        training_cfg.batch_size = args.batch_size
    if args.context_length is not None:
        model_cfg.context_length = args.context_length
    if args.device is not None:
        model_cfg.device = args.device
    if args.lr is not None:
        optimizer_cfg.lr = args.lr
    if args.weight_decay is not None:
        optimizer_cfg.weight_decay = args.weight_decay
    if args.train_file_path is not None:
        training_cfg.train_file_path = args.train_file_path
    if args.out_checkpoint_path is not None:
        training_cfg.out_checkpoint_path = args.out_checkpoint_path
    if args.log_every is not None:
        training_cfg.log_every = args.log_every

    groups = {
        "model": model_cfg,
        "optim": optimizer_cfg,
        "optimizer": optimizer_cfg,
        "train": training_cfg,
        "training": training_cfg,
    }

    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected CLI token: {token}")

        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
        else:
            i += 1
            if i >= len(unknown):
                raise ValueError(f"Missing value for --{key}")
            value = unknown[i]

        if "." not in key:
            raise ValueError(
                f"Unknown arg --{key}. Use --model.x, --optimizer.x, or --training.x"
            )

        group, field = key.split(".", 1)
        if group not in groups:
            raise ValueError(f"Unknown config group: {group}")

        set_cfg_attr(groups[group], field, value)
        i += 1


def cfg_dict(model_cfg, optimizer_cfg, training_cfg):
    out = {}
    for prefix, cfg in [
        ("model", model_cfg),
        ("optimizer", optimizer_cfg),
        ("training", training_cfg),
    ]:
        for k, v in vars(cfg).items():
            if isinstance(v, (int, float, str, bool, type(None))):
                out[f"{prefix}.{k}"] = v
            else:
                out[f"{prefix}.{k}"] = str(v)
    return out


def hardware_dict(device):
    info = {
        "torch.version": torch.__version__,
        "torch.cuda": torch.version.cuda,
        "torch.cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "cuda.available": torch.cuda.is_available(),
    }

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "cuda.gpu_name": props.name,
            "cuda.capability": f"{props.major}.{props.minor}",
            "cuda.total_memory_gb": props.total_memory / 1e9,
            "cuda.multi_processor_count": props.multi_processor_count,
        })

    return info


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def model_stats(model):
    device = next(model.parameters()).device

    grad_sq = torch.zeros((), device=device)
    grad_abs_sum = torch.zeros((), device=device)
    grad_abs_max = torch.zeros((), device=device)
    grad_count = 0

    param_sq = torch.zeros((), device=device)
    param_abs_sum = torch.zeros((), device=device)
    param_abs_max = torch.zeros((), device=device)
    param_count = 0

    for p in model.parameters():
        pf = p.detach().float()
        param_sq += (pf * pf).sum()
        param_abs_sum += pf.abs().sum()
        param_abs_max = torch.maximum(param_abs_max, pf.abs().max())
        param_count += p.numel()

        if p.grad is not None:
            gf = p.grad.detach().float()
            grad_sq += (gf * gf).sum()
            grad_abs_sum += gf.abs().sum()
            grad_abs_max = torch.maximum(grad_abs_max, gf.abs().max())
            grad_count += gf.numel()

    return {
        "grad/global_norm": grad_sq.sqrt().item(),
        "grad/abs_mean": (grad_abs_sum / max(1, grad_count)).item(),
        "grad/abs_max": grad_abs_max.item(),
        "param/global_norm": param_sq.sqrt().item(),
        "param/abs_mean": (param_abs_sum / max(1, param_count)).item(),
        "param/abs_max": param_abs_max.item(),
    }


def cuda_mem_stats(device):
    if device.type != "cuda":
        return {}

    total = torch.cuda.get_device_properties(device).total_memory

    return {
        "cuda/memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "cuda/memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "cuda/max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "cuda/max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "cuda/memory_allocated_pct": 100 * torch.cuda.memory_allocated(device) / total,
    }


def current_lr(optimizer, optimizer_cfg):
    if hasattr(optimizer, "param_groups"):
        return optimizer.param_groups[0].get("lr", None)
    return getattr(optimizer_cfg, "lr", None)


def make_profiler(args, device):
    if not args.profile:
        return contextlib.nullcontext(None)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=args.profile_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def log_profiler_to_wandb(run, profile_dir):
    traces = glob.glob(os.path.join(profile_dir, "**", "*.pt.trace.json"), recursive=True)
    if not traces:
        return

    artifact = wandb.Artifact("torch-profiler-traces", type="profile")
    for trace in traces:
        artifact.add_file(trace)
    run.log_artifact(artifact)


if __name__ == "__main__":
    args, unknown = parse_args()

    model_cfg = config.ModelConfig()
    optimizer_cfg = config.OptimizerConfig()
    training_cfg = config.TrainingConfig()
    apply_cli(args, unknown, model_cfg, optimizer_cfg, training_cfg)

    device = torch.device(model_cfg.device)

    model = NeuralNets.transformer_lm(**model_cfg.__dict__)
    model.to(device=device, dtype=torch.float32)

    optimizer = NeuralNets.AdamW(model.parameters(), **optimizer_cfg.__dict__)
    dataset = np.load(training_cfg.train_file_path, mmap_mode="r")
    loss_fn = NeuralNets.cross_entropy_loss

    total_params, trainable_params = count_params(model)
    tokens_per_step = training_cfg.batch_size * model_cfg.context_length

    run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("Install W&B first: pip install wandb")

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                **cfg_dict(model_cfg, optimizer_cfg, training_cfg),
                **hardware_dict(device),
                "model.total_params": total_params,
                "model.trainable_params": trainable_params,
                "benchmark.tokens_per_step": tokens_per_step,
            },
        )

        if not args.no_wandb_watch:
            wandb.watch(
                model,
                log="all",
                log_freq=args.wandb_watch_freq,
                log_graph=False,
            )

    with make_profiler(args, device) as prof:
        for step in range(1, training_cfg.steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            sync_if_cuda(device)
            t0 = time.perf_counter()

            x_batch, y_batch = NeuralNets.get_batch(
                dataset=dataset,
                batch_size=training_cfg.batch_size,
                context_len=model_cfg.context_length,
                device=model_cfg.device,
            )

            logits = model(x_batch)
            loss = loss_fn(logits=logits, targets=y_batch)

            optimizer.zero_grad()
            loss.backward()

            stats = model_stats(model)

            optimizer.step()

            sync_if_cuda(device)
            step_time = time.perf_counter() - t0

            metrics = {
                "train/loss": loss.item(),
                "train/perplexity": math.exp(min(loss.item(), 20)),
                "train/lr": current_lr(optimizer, optimizer_cfg),
                "perf/step_time_s": step_time,
                "perf/tokens_per_s": tokens_per_step / step_time,
                "perf/samples_per_s": training_cfg.batch_size / step_time,
                "perf/tokens_seen": step * tokens_per_step,
                **stats,
                **cuda_mem_stats(device),
            }

            print(
                f"step={step} "
                f"loss={loss.item():.4f} "
                f"time={step_time:.3f}s "
                f"tok/s={tokens_per_step / step_time:.0f}"
            )

            if run is not None and step % args.wandb_log_every == 0:
                wandb.log(metrics, step=step)

            if step % training_cfg.log_every == 0:
                NeuralNets.save_checkpoint(
                    model,
                    optimizer,
                    step,
                    training_cfg.out_checkpoint_path + f"{step}.pt",
                )

            if prof is not None:
                prof.step()

    if run is not None:
        run.summary["final_loss"] = loss.item()
        run.summary["total_tokens_seen"] = training_cfg.steps * tokens_per_step
        run.summary["total_params"] = total_params
        run.summary["trainable_params"] = trainable_params

        if args.profile:
            log_profiler_to_wandb(run, args.profile_dir)

        wandb.finish()