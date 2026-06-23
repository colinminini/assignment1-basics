import config
import NeuralNets
import torch
import numpy as np
import time

# Training script, flexible on arguments: config default + parse_arguments() for CLI override
# `uv run cs336_basics/train.py`

# What do we need in the training environment?

# Num training tokens: num_steps * batch_size * context_length
# Weights budget / model configuration: d_model, context_len, vocab_size, num_layers
# -> Together gives FLOPs budget

# Optimizer hyperparameters: lr (max and min if lr_scheduler), weight decay, betas for AdamW

# Initialize model and optimizer (attached to model parameters()) and load them in HBM

# Training Loop: get_batch to GPU -> forward() -> zero_grad() + backward() -> optmizer step() -> log batch_loss & gradients -> save model & optimizer checkpoints if num_steps % log_step == 0 -> repeat

# ---

# Config is the default, replace with CLI arguments if provided: def parse_agrs() ...

# Log every step with weights and biases ...

@torch.no_grad
def metrics_logging(model, optimizer, step, out_checkpoint_path):
    global_grad_norm = sum(torch.square(p.grad).sum() for p in model.parameters() if p.grad is not None).sqrt.item()
    stats = {
        'gloab_grad_norm': global_grad_norm,
    }
    return stats

if __name__ == '__main__':

    model_cfg = config.ModelConfig()
    optimizer_cfg = config.OptimizerConfig()
    training_cfg = config.TrainingConfig()
    tokens_per_step = training_cfg.batch_size * model_cfg.context_length

    model = NeuralNets.transformer_lm(**model_cfg.__dict__)

    num_total_param = sum(torch.numel(p) for p in model.parameters())

    optimizer = NeuralNets.AdamW(model.parameters(), **optimizer_cfg.__dict__)
    dataset = np.load(training_cfg.train_file_path, mmap_mode='r')
    loss_fn = NeuralNets.cross_entropy_loss
    gardient_clipping = NeuralNets.gradient_clipping
    cosine_lr_schedule = NeuralNets.cosine_lr_schedule

    print('Total Parameter Count:', f'{num_total_param}')

    start_time = time.perf_counter()
    for step in range(1, training_cfg.steps + 1):
        x_batch, y_batch = NeuralNets.get_batch(dataset=dataset, batch_size=training_cfg.batch_size, context_len=model_cfg.context_length, device=model_cfg.device) # get_batch() to GPU
        logits = model(x_batch) # forward() pass
        # print('logits:', f'{logits}')
        loss = loss_fn(logits=logits, targets=y_batch) # Cost scalar
        print('loss:', f'{loss.item():.4f}')
        optimizer.zero_grad() # zeroing out the parameters gradients. Autograd graph ready for backprop()
        loss.backward() # backward pass (note: Flash-attention re-computes n square attention scores instead of savings the activations)
        gardient_clipping(model.parameters(), max_grad=training_cfg.max_grad)
        optimizer.step() # AdamW step, weights get updated
        if step % training_cfg.val_and_log_every == 0: # save model & optimizer states (& iteration step)
            with torch.no_grad():
                NeuralNets.save_checkpoint(model, optimizer, step, training_cfg.out_checkpoint_path + f'{step}' + '.pt')
        # ... log relevant metrics into weights and biases
        end_time = time.perf_counter() - start_time
        print('time_spent_on_batch:', f'{end_time:.2f}', 'tokens_per_second:', f'{int(tokens_per_step / end_time)}')
        start_time = time.perf_counter()
