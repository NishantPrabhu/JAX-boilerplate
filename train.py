
import os 
import time 
import flax 
import logging
import argparse
import functools
from clu import platform 
from flax import jax_utils 
from flax import optim 
from networks import resnet 
from flax.training import checkpoints 
from flax.training import common_utils
from flax.training import train_state 
from utils import data_utils, expt_utils
from clu import metric_writers
from clu import periodic_actions
from torchvision import transforms, datasets
from datetime import datetime as dt
from typing import Any

import jax 
from jax import lax 
from jax import random 
import jax.numpy as jnp 
import optax 
import tensorflow as tf

NETWORKS = {
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101
}

DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}

# Experimental 
# os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


class TrainState(train_state.TrainState):
    batch_stats: Any 
    dynamic_scale: flax.optim.DynamicScale
    

def create_model(name, half_precision, num_classes, pre_conv):
    assert name in NETWORKS, f'Unrecognized network {name}'
    platform = jax.local_devices()[0].platform 
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16 
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return NETWORKS[name](num_classes=num_classes, dtype=model_dtype, pre_conv=pre_conv)

def initialize(key, image_shape, model):
    input_shape = (1, *image_shape)
    @jax.jit 
    def init(*args):
        return model.init(*args)
    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
    return variables['params'], variables['batch_stats']

def cross_entropy_loss(logits, labels):
    onehot_labels = common_utils.onehot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=onehot_labels)
    return jnp.mean(loss)

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics 

def create_lr_schedule(total_epochs, warmup_epochs, base_lr, steps_per_epoch):
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_lr, 
        transition_steps=warmup_epochs * steps_per_epoch
    )
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=cosine_epochs * steps_per_epoch
    ) 
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch]
    )
    return schedule_fn 

def train_step(state, batch, lr_func, weight_decay):
    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats']
        )
        loss = cross_entropy_loss(logits, batch['label'])
        weight_penalty_params = jax.tree_leaves(params)
        weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2 
        loss = loss + weight_penalty 
        return loss, (new_model_state, logits)
    
    step = state.step 
    dynamic_scale = state.dynamic_scale 
    lr = lr_func(step)
    
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        grads = lax.pmean(grads, axis_name='batch')
    
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])
    metrics['learning_rate'] = lr 
    
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state = jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state, 
                state.opt_state
            ),
            params = jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.partial,
                state.params
            )
        )
        metrics['scale'] = dynamic_scale.scale 
    return new_state, metrics

def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
    return compute_metrics(logits, batch['label'])

def create_dataloaders(data_name, transform, loader_cfg):
    train_transform = data_utils.DataTransform(transform['train'])
    val_transform = data_utils.DataTransform(transform['val'])
    train_dset = DATASETS[data_name](root=f'~/Datasets/{data_name}', train=True, transform=train_transform, download=True)
    val_dset = DATASETS[data_name](root=f'~/Datasets/{data_name}', train=False, transform=val_transform, download=True)
    train_loader = data_utils.JaxDataLoader(train_dset, shuffle=True, **loader_cfg)
    val_loader = data_utils.JaxDataLoader(val_dset, shuffle=False, **loader_cfg)
    return train_loader, val_loader

def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)

def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)

cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')

def sync_batch_stats(state):
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def create_train_state(rng, half_precision, model, img_shape, lr_func, momentum):
    dynamic_scale = None 
    platform = jax.local_devices()[0].platform 
    
    if half_precision and platform == 'gpu':
        dynamic_scale = optim.DynamicScale()
    else:
        dynamic_scale = None 
        
    params, batch_state = initialize(rng, img_shape, model)
    tx = optax.sgd(learning_rate=lr_func, momentum=momentum, nesterov=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_state,
        dynamic_scale=dynamic_scale
    )
    return state 

def run(args, workdir):
    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)
    rng = jax.random.PRNGKey(args.seed)
    
    if args.batch_size % jax.device_count() > 0:
        raise ValueError(f"Batch size {args.batch_size} not divisible by device count {jax.device_count()}")
    local_batch_size = args.batch_size // jax.process_count()
    platform = jax.local_devices()[0].platform 
    
    transform = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])
    }
    loader_cfg = {
        'batch_size': args.batch_size,
        'num_workers': 0,
        'pin_memory': args.pin_memory,
    }
    train_loader, val_loader = create_dataloaders(args.data_name, transform, loader_cfg)
    train_steps_per_epoch, val_steps_per_epoch = len(train_loader), len(val_loader)
    
    # Model
    assert args.net in NETWORKS, f'Network {args.net} is not available'
    model = NETWORKS[args.net](num_classes=args.n_classes, pre_conv=args.pre_conv)
    lr_func = create_lr_schedule(args.epochs, args.warmup_epochs, args.lr, train_steps_per_epoch)
    
    state = create_train_state(rng, args.half_precision, model, args.img_shape, lr_func, args.momentum)
    state = restore_checkpoint(state, workdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state)
    
    p_train_step = jax.pmap(functools.partial(train_step, lr_func=lr_func, weight_decay=args.weight_decay), axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')
    
    train_metrics = []
    hooks = []        
    train_metrics_last_t = time.time()
    best_val = 0
    
    # Main loop
    print('\n[info] beginning training...\n')
    for epoch in range(1, args.epochs+1):
        
        for step, batch in enumerate(train_loader):
            batch = {'image': batch[0], 'label': batch[1]}
            state, metrics = p_train_step(state, batch)
            train_metrics.append(metrics)
            expt_utils.progress_bar((step+1)/len(train_loader), desc=f'train epoch {epoch}', status="")
        print()
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {f'train {k}': v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()}
        summary['epoch_time'] = time.time() - train_metrics_last_t
        
        print('Train epoch {} | {}'.format(epoch, ' | '.join(['{}: {:.4f}'.format(k, v) for k, v in summary.items()])))
        train_metrics = []
        train_metrics_last_t = time.time()
        
        if epoch % args.eval_every == 0:
            eval_metrics = []
            state = sync_batch_stats(state)
            
            for step, batch in enumerate(val_loader):
                batch = {'image': batch[0], 'label': batch[1]}
                metrics = p_eval_step(state, batch)
                eval_metrics.append(metrics)
                expt_utils.progress_bar((step+1)/len(val_loader), desc=f'eval epoch {epoch}', status="")
            print()
            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            print('Eval epoch {} | Loss: {:.4f} | Accuracy: {:.2f}'.format(
                epoch, summary['loss'], summary['accuracy'] * 100.0
            ))
            writer.write_scalars(
                epoch, {f'eval_{key}': val for key, val in summary.items()}
            )
            writer.flush()
            
            if summary['accuracy'] > best_val:
                best_val = summary['accuracy']
                state = sync_batch_stats(state)
                save_checkpoint(state, workdir)
        
        print('-----------------------------------------------------------------')
                
    jax.random.normal(jax.random.PRNGKey(args.seed), ()).block_until_ready()
    return state


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default=1, type=int, help='random seed')
    ap.add_argument('--net', default='resnet18', type=str, help='model architecture')
    ap.add_argument('--pre_conv', default='small', type=str, choices=['small', 'full'], help='reduced or full preconv for resnet')
    ap.add_argument('--img_shape', default=(32, 32, 3), type=tuple, help='image shape')
    ap.add_argument('--data_name', default='cifar10', type=str, choices=list(DATASETS.keys()), help='dataset')
    ap.add_argument('--n_classes', default=10, type=int, help='number of classes')
    ap.add_argument('--batch_size', default=128, type=int, help='batch size')
    ap.add_argument('--n_workers', default=1, type=int, help='dataloading worker count')
    ap.add_argument('--pin_memory', action='store_true', help='pin memory')
    ap.add_argument('--half_precision', action='store_true', help='float16 precision training')
    ap.add_argument('--lr', default=0.1, type=float, help='learning rate')
    ap.add_argument('--momentum', default=0.9, type=float, help='optimizer sgd momentum')
    ap.add_argument('--epochs', default=100, type=int, help='training epochs')
    ap.add_argument('--weight_decay', default=1e-06, type=float, help='weight decay')
    ap.add_argument('--warmup_epochs', default=0, type=int, help='linear LR warmup epochs')
    ap.add_argument('--log_interval', default=10, type=int, help='number of steps to log after')
    ap.add_argument('--eval_every', default=1, type=int, help='number of epochs to eval after')
    args = ap.parse_args()

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                        f'process_count: {jax.process_count()}')
    run(args, 'tmp/cifar10')