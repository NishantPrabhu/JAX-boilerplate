
import optax 

def cosine_lr_schedule(base_lr, total_epochs, warmup_epochs, steps_per_epoch):
    warmup_fn = optax.linear_schedule(0, base_lr, warmup_epochs * steps_per_epoch)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(base_lr, cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules = [warmup_fn, cosine_fn],
        boundaries = [warmup_epochs * steps_per_epoch]
    ) 
    return schedule_fn

def multistep_lr_schedule(base_lr, lr_decay, milestones, warmup_epochs, steps_per_epoch):
    milestones = [int(m) * steps_per_epoch for m in milestones.split(',')]
    b_and_s = {m: lr_decay for m in milestones}
    warmup_fn = optax.linear_schedule(0, base_lr, warmup_epochs * steps_per_epoch)
    piecewise_fn = optax.piecewise_constant_schedule(base_lr, b_and_s)
    schedule_fn = optax.join_schedules(
        schedules = [warmup_fn, piecewise_fn],
        boundaries = [warmup_epochs * steps_per_epoch]
    )
    return schedule_fn