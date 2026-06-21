import config
import NeuralNets
import torch
import numpy as np
import time

# Training script, flexible on arguments: config default + parse_arguments() for CLI override
# `uv run cs336_basics/train_and_log.py`

# What do we need in the training environment?

# num training tokens (num_steps * batch_size * context_length)
# Weights budget / model configuration: d_model, context_len, vocab_size, num_layers
# -> Together gives FLOPs budget

# Optimizer hyperparameters: lr (max and min if lr_scheduler), weight decay, betas for AdamW

# Initialize model and optimizer (attached to model parameters()) and load them in HBM

# Training Loop: get_batch to GPU -> forward() -> zero_grad() + backward() -> optmizer step() -> log batch_loss & gradients -> save model & optimizer checkpoints if num_steps % log_step == 0 -> repeat

# ---

# Config is the default, replace with CLI arguments if provided: def parse_agrs() ...

model_cfg = config.ModelConfig()
optimizer_cfg = config.OptimizerConfig()
training_cfg = config.TrainingConfig()

model = NeuralNets.transformer_lm(**model_cfg.__dict__)
model.to(dtype=torch.float32)

optimizer = NeuralNets.AdamW(model.parameters(), **optimizer_cfg.__dict__)
dataset = np.load(training_cfg.train_file_path, mmap_mode='r')
loss_fn = NeuralNets.cross_entropy_loss

# Log every step with weights and biases ...

start_time = time.time()

if __name__ == '__main__':
    for step in range(1, training_cfg.steps + 1):
        x_batch, y_batch = NeuralNets.get_batch(dataset=dataset, batch_size=training_cfg.batch_size, context_len=model_cfg.context_length, device=model_cfg.device) # get_batch() to GPU
        logits = model(x_batch) # forward() pass
        #print('logits:', f'{logits}')
        loss = loss_fn(logits=logits, targets=y_batch) # Cost scalar
        print('loss:', f'{loss.item()}')
        optimizer.zero_grad() # zeroing out the parameters gradients. Autograd graph ready for backprop()
        loss.backward() # backward pass (note: Flash-attention re-computes n square attention scores instead of savings the activations)
        #print('params_before:', f'{list(model.parameters())[4]}')
        optimizer.step() # AdamW step, weights get updated
        #print('params_after:', f'{list(model.parameters())[4]}')
        if step % training_cfg.log_every == 0: # save model & optimizer states (& iteration step)
            NeuralNets.save_checkpoint(model, optimizer, step, training_cfg.out_checkpoint_path + f'{step}' + '.pt')
        # ... log relevant metrics into weights and biases
        print('time_spent_on_batch:', f'{time.time() - start_time}')
        start_time = time.time()
