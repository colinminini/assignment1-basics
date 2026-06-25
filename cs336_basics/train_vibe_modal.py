import modal

app = modal.App("cs336-train")

# IMAGE: ship your local modules in as the top layer. Pin torch to your local
# version for reproducibility (e.g. "torch==2.6.0").
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("config", "NeuralNets", "train_vibe")
)

# VOLUMES: data is read-only-ish, checkpoints must persist past the container.
data_vol = modal.Volume.from_name("cs336-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("cs336-ckpt", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",                              # CUDA hardware attached at run
    secrets=[modal.Secret.from_name("wandb")],    # -> WANDB_API_KEY env var
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    timeout=24 * 60 * 60,
)
def train(argv: list[str]):
    import sys, runpy

    sys.argv = ["train_vibe.py", *argv]           # feed flags to your parse_args()
    runpy.run_module("train_vibe", run_name="__main__")  # runs your script unchanged
    ckpt_vol.commit()                             # flush /ckpt writes to the volume


@app.local_entrypoint()
def main(args: str = ""):
    # `args` is your full CLI string, forwarded to the container untouched.
    train.remote(args.split())


@app.local_entrypoint()
def sweep():
    ctx, tokens = 256, 10_000_000          # fixed token budget => fair across bs
    base = ("--device cuda --with_gradient_clipping --wandb --profile --no_wandb_watch --with_torch_compile "
            "--train_file_path /data/TinyStoriesV2-GPT4-train.npy "
            "--val_file_path /data/TinyStoriesV2-GPT4-valid.npy "
            "--out_checkpoint_path /ckpt/run_")
    grid = []
    for bs in (8, 16, 32, 64, 128):
        for lr in (1e-4, 5e-4, 1e-3, 5e-3):
            steps = tokens // (bs * ctx)
            grid.append(f"{base} --batch_size {bs} --lr {lr:g} --context_length {ctx} "
                        f"--steps {steps} --wandb_run_name bs{bs}_lr{lr:g}".split())
    list(train.map(grid))                    # one GPU container per (bs, lr)

# For the hyperparameters sweep lr/batch_size at fixed training tokens (small run): run `modal run cs336_basics/train_vibe_modal.py::sweep``
