
# This script will contain the function called by the accelerator for distributed training
# Meant for use with TPUs as of now

import jax
import jmp
import optax
import wandb
import argparse
import networks
import functools
import haiku as hk
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple
from datetime import datetime as dt
from utils import data_utils, optim_utils, expt_utils


class TrainState(NamedTuple):
    params: hk.params
    state: hk.State
    opt_state: optax.OptState

def _forward_pass(model, batch, input_keys, is_training):
    input_vals = [batch[key] for key in input_keys]
    return model(*input_vals, is_training=is_training)

def initialize_network(rng, fwd_func, optimizer, batch):
    params, state = fwd_func.init(rng, batch, is_training=True)
    opt_state = optimizer.init(params)
    return TrainState(params=params, state=state, opt_state=opt_state)

@functools.partial(jax.pmap, axis_name="i", donate_args=(0,))
def train_step(train_state, batch, loss_fn, optimizer):
    params, state, opt_state = train_state
    grads, (loss, metrics, new_state) = jax.grad(loss_fn, has_aux=True)(params, state, batch)
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    scalars = {"loss": loss} if metrics is not None else {"loss": loss, **metrics}
    scalars = jax.lax.pmean(scalars, axis_name="i")
    scalars = jax.tree_map(lambda v: np.mean(v).item(), jax.device_get(scalars))
    train_state = TrainState(params=new_params, state=new_state, opt_state=new_opt_state)
    return train_state, scalars

@jax.jit
def eval_step(fwd_func, params, state, batch):
    logits, _ = fwd_func.apply(params, state, None, batch, is_training=False)
    preds = jnp.argmax(logits, axis=-1)
    eval_metrics = compute_metrics(preds, batch["labels"])                                      # TODO: Metrics computation
    return eval_metrics

def train():
    # Get CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file")
    ap.add_argument("-o", "--output", type=str, default=dt.now().strftime("%d-%m-%Y_%H-%M"), help="Name of output directory")
    ap.add_argument("-l", "--load", type=str, default=None, help="Path to directory from which best model is to be loaded")
    args = vars(ap.parse_args())
    config, output_dir, logger = expt_utils.initialize_experiment(args, output_root="./outputs/test", seed=420)

    device_count = jax.local_device_count()
    fwd_func = hk.transform_with_state(_forward_pass)

    # Datasets and loaders
    train_dset, test_dset = data_utils.prepare_standard_dataset(name=config["data"]["name"], transforms=config["data"]["transforms"])
    train_loader, test_loader = data_utils.get_multi_epoch_loaders(train_dset, test_dset, config["data"]["batch_size"], config["data"]["num_workers"])

    # Model, optimizer, loss function
    rng = jax.random.PRNGKey(420)
    rng = jnp.broadcast_to(rng, (device_count,) + rng.shape)
    batch = next(train_loader)

    lr_schedule = optim_utils.get_scheduler(**config["scheduler"])
    optimizer = optim_utils.get_optimizer(name=config["optim"].pop("name"), learning_rate=lr_schedule, **config["optim"])
    train_state = jax.pmap(initialize_network)(rng, fwd_func, optimizer, batch)
    loss_fn = None
    ckpt_metric, best_metric_val = config["checkpoint"]["metric"], config["checkpoint"]["worst_value"]

    # Summary and wandb
    run = wandb.init(project=config["wandb"]["project"])
    logger.write("Wandb run: {}".format(run.get_url()), mode="info")
    summary = hk.experimental.tabulate(train_step)(train_state, batch)
    for line in summary.split("\n"):
        logger.record(line, mode="info")

    # Begin training/evaluation loop
    for epoch in range(1, config["epochs"]):
        desc = "[TRAIN] Epoch {:4d}/{:4d}".format(epoch, config["epochs"])
        avg_meter = expt_utils.AverageMeter()

        for step, batch in enumerate(train_loader):
            train_state, metrics = train_step(train_state, batch, loss_fn, optimizer)
            avg_meter.add(metrics)
            if (step+1) % config["log_every"] == 0 and (config["log_every"] > 0):
                expt_utils.progress_bar(progress=(step+1)/len(train_loader), desc=desc, status=avg_meter.return_msg())
                wandb.log({"Train loss": metrics["loss"], "Step": step+1})
        print()
        train_metrics = avg_meter.return_avg()
        metric_log = {f"Train {key}": train_metrics[value] for key, value in train_metrics.items() if key != "loss"}
        logger.write("Epoch {:4d}/{:4d} - {}".format(epoch, config["epochs"], avg_meter.return_msg()), mode="train")
        wandb.log({**metric_log, "Epoch": epoch})

        if epoch % config["eval_every"] == 0:
            desc = "{}[VALID] Epoch {:4d}/{:4d}{}".format(expt_utils.COLORS["blue"], epoch, config["epochs"], expt_utils.COLORS["end"])
            avg_meter = expt_utils.AverageMeter()

            for step, batch in enumerate(test_loader):
                metrics = eval_step(fwd_func, train_state.params, train_state.state, batch)
                avg_meter.add(metrics)
                if (step+1) % config["log_every"] == 0 and (config["log_every"] > 0):
                    expt_utils.progress_bar(progress=(step+1)/len(test_loader), desc=desc, status=avg_meter.return_msg())
            print()
            val_metrics = avg_meter.return_avg()
            metric_log = {f"Val {key}": val_metrics[value] for key, value in val_metrics.items()}
            logger.write("Epoch {:4d}/{:4d} - {}".format(epoch, config["epochs"], avg_meter.return_msg()), mode="val")
            wandb.log({**metric_log, "Epoch": epoch})

            if val_metrics.get(ckpt_metric) > best_metric_val:
                best_metric_val == val_metrics.get(ckpt_metric)
                # TODO Checkpointing mechanism
    print()
    logger.record("Training complete!", mode="info")
