
import jax
import optax
import jax.numpy as jnp

# ====================================================================
# Custom optimizers or schedulers should be in this section
# ====================================================================

def linear_warmup_cosine_decay(max_lr, warmup_steps, decay_steps, alpha=0.0):
    linear_schedule = optax.linear_schedule(init_value=1e-12, end_value=max_lr, transition_steps=warmup_steps)
    cosine_decay_schedule = optax.cosine_decay_schedule(init_value=max_lr, decay_steps=decay_steps, alpha=alpha)

    def schedule(step):
        if step <= warmup_steps:
            return linear_schedule(step)
        else:
            return cosine_decay_schedule(step)
    return schedule

# ==============++++++++++++++++++++++++++++++++++++++++++++++++++++++

OPTIMIZERS = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "rmsprop": optax.rmsprop,
    "lamb": optax.lamb
}

SCHEDULERS = {
    "constant": optax.constant_schedule,
    "cosine": optax.cosine_decay_schedule,
    "multistep": optax.piecewise_constant_schedule,
    "warmup_cosine": linear_warmup_cosine_decay
}

def get_optimizer(name, learning_rate, **kwargs):
    assert name in OPTIMIZERS.keys(), f"Invalid optimizer {name}, should be one of {list(OPTIMIZERS.keys())}"
    optimizer = OPTIMIZERS[name](learning_rate=learning_rate, **kwargs)
    return optimizer

def get_scheduler(name, epochs, warmup_epochs, steps_per_epoch, **kwargs):
    assert name in SCHEDULERS.keys(), f"Invalid scheduler {name}, should be one of {list(SCHEDULERS.keys())}"
    if name in ["constant", "multistep"]:
        lr_scheduler = SCHEDULERS[name](**kwargs)
    elif name == "cosine":
        lr_scheduler = SCHEDULERS[name](decay_steps=epochs*steps_per_epoch, **kwargs)
    elif name == "warmup_cosine":
        warmup_steps = warmup_epochs * steps_per_epoch
        decay_steps = (epochs - warmup_epochs) * steps_per_epoch
        lr_scheduler = SCHEDULERS[name](warmup_steps=warmup_steps, decay_steps=decay_steps, **kwargs)
    return lr_scheduler
