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