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
import wandb

# Run `bash run.sh`

# Training script, flexible on arguments: config default + parse_arguments() for CLI override

# What do we need in the training environment?

# Num training tokens: num_steps * batch_size * context_length
# Weights budget / model configuration: d_model, context_len, vocab_size, num_layers
# -> Together gives FLOPs budget

# Optimizer hyperparameters: lr (max and min if lr_scheduler), weight decay, betas for AdamW

# Initialize model and optimizer (attached to model parameters()) and load them in HBM

# Training Loop: get_batch to GPU -> forward() -> zero_grad() + backward() -> optmizer step() -> log batch_loss & gradients -> save model & optimizer checkpoints if num_steps % log_step == 0 -> repeat

# ---

# Config is the default, replace with CLI arguments if provided: def parse_agrs() ...


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
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--lr_max", type=float, default=None)
    parser.add_argument("--T_warmup", type=float, default=None)
    parser.add_argument("--T_c", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--train_file_path", type=str, default=None)
    parser.add_argument("--val_file_path", type=str, default=None)
    parser.add_argument("--out_checkpoint_path", type=str, default=None)
    parser.add_argument('--with_cosine_lr', action='store_true')
    parser.add_argument('--with_gradient_clipping', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--torch_compile', action='store_true')

    args = parser.parse_args()
    return args

def apply_cli(args, model_cfg, optimizer_cfg, training_cfg):
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
    if args.lr_min is not None:
        training_cfg.lr_min = args.lr_min
    if args.lr_max is not None:
        training_cfg.lr_max = args.lr_max
    if args.T_warmup is not None:
        training_cfg.T_warmup = args.T_warmup
    if args.T_c is not None:
        training_cfg.T_c = args.T_c
    if args.weight_decay is not None:
        optimizer_cfg.weight_decay = args.weight_decay
    if args.train_file_path is not None:
        training_cfg.train_file_path = args.train_file_path
    if args.val_file_path is not None:
        training_cfg.val_file_path = args.val_file_path
    if args.out_checkpoint_path is not None:
        training_cfg.out_checkpoint_path = args.out_checkpoint_path

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
    args = parse_args()

    model_cfg = config.ModelConfig()
    optimizer_cfg = config.OptimizerConfig()
    training_cfg = config.TrainingConfig()
    apply_cli(args, model_cfg, optimizer_cfg, training_cfg)

    device = torch.device(model_cfg.device)

    model_cfg.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    model = NeuralNets.transformer_lm(**model_cfg.__dict__)

    if args.torch_compile:
        model = torch.compile(model)

    optimizer = NeuralNets.AdamW(model.parameters(), **optimizer_cfg.__dict__)

    train_dataset = np.load(training_cfg.train_file_path, mmap_mode="r")
    val_dataset = np.load(training_cfg.val_file_path, mmap_mode='r')

    loss_fn = NeuralNets.cross_entropy_loss
    if args.with_gradient_clipping:
        gradient_clipping = NeuralNets.gradient_clipping
    if args.with_cosine_lr:
        cosine_lr_schedule = NeuralNets.cosine_lr_schedule

    total_params = sum(torch.numel(p) for p in model.parameters())

    tokens_per_step = training_cfg.batch_size * model_cfg.context_length

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                **model_cfg.__dict__, 
                **optimizer_cfg.__dict__, 
                **training_cfg.__dict__,
                **hardware_dict(device),
                "model.total_params": total_params,
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
        start_wall_clock_time = time.perf_counter()
        
        for step in range(1, training_cfg.steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            if device.type == 'mps':
                torch.mps.synchronize()

            t0 = time.perf_counter()

            x_batch, y_batch = NeuralNets.get_batch(
                dataset=train_dataset,
                batch_size=training_cfg.batch_size,
                context_len=model_cfg.context_length,
                device=device,
            )

            logits = model(x_batch)
            loss = loss_fn(logits=logits, targets=y_batch)

            optimizer.zero_grad()
            loss.backward()
            if args.with_gradient_clipping:
                gradient_clipping(model.parameters(), max_grad=training_cfg.max_grad)
            if args.with_cosine_lr:
                for group in optimizer.param_groups:
                    group['lr'] = cosine_lr_schedule(t=step, lr_min=training_cfg.lr_min, lr_max=training_cfg.lr_max, T_warmup=training_cfg.T_warmup, T_c=training_cfg.T_c)

            optimizer.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            if device.type == 'mps':
                torch.mps.synchronize()
            step_time = time.perf_counter() - t0

            if run is not None and step % args.wandb_log_every == 0:
                stats = model_stats(model)

            print(
                f"step={step} "
                f"loss={loss.item():.4f} "
                f"time={step_time:.3f}s "
                f"tok/s={tokens_per_step / step_time:.0f}"
            )

            if run is not None and step % args.wandb_log_every == 0:
                metrics = {
                "train/loss": loss.item(),
                "train/perplexity": math.exp(min(loss.item(), 20)),
                "train/lr": optimizer.param_groups[0]['lr'],
                "perf/step_time_s": step_time,
                "perf/tokens_per_s": tokens_per_step / step_time,
                "perf/samples_per_s": training_cfg.batch_size / step_time,
                "perf/tokens_seen": step * tokens_per_step,
                "perf/wall_clock_time": time.perf_counter() - start_wall_clock_time,
                **stats,
                **cuda_mem_stats(device)}
                wandb.log(metrics, step=step)

            if step % training_cfg.val_and_log_every == 0:
                to_save = getattr(model, "_orig_mod", model)
                NeuralNets.save_checkpoint(
                    to_save,
                    optimizer,
                    step,
                    training_cfg.out_checkpoint_path + f"{step}.pt",
                )

                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    num_val_steps = 50

                    for i in range(num_val_steps):
                        x_batch, y_batch = NeuralNets.get_batch(dataset=val_dataset, batch_size=training_cfg.batch_size, context_len=model_cfg.context_length, device=device)
                        val_loss += loss_fn(model(x_batch), y_batch).item()

                val_loss = val_loss / num_val_steps

                model.train()

                if run is not None:
                    wandb.log({'validation/loss': val_loss}, step=step)

            if prof is not None:
                prof.step()

    if run is not None:
        run.summary["final_loss"] = loss.item()
        run.summary["total_tokens_seen"] = training_cfg.steps * tokens_per_step
        run.summary["total_params"] = total_params
        run.summary['total_wall_clock_time'] = time.perf_counter() - start_wall_clock_time
        if args.profile:
            log_profiler_to_wandb(run, args.profile_dir)

        wandb.finish()