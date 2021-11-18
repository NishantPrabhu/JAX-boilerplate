
import os 
import time 
import torch
import logging
import argparse
import functools
import numpy as np
from typing import Any
from datetime import datetime as dt
from utils import data_utils, expt_utils
from torchvision import transforms, datasets
from networks import resnet 

import jax 
import flax 
import optax 
import losses
import jax.numpy as jnp 
import tensorflow as tf
from jax import lax 
from jax import random 
from flax import optim 
from clu import platform 
from flax import jax_utils 
from clu import metric_writers
from flax.training import checkpoints
from flax.training import train_state 
from flax.training import common_utils

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

NETWORKS = {
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101
}

TORCH_DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}

class TrainState(train_state.TrainState):
    batch_stats: Any 
    dynamic_scale: flax.optim.DynamicScale


class Trainer:
    
    def __init__(self, args):
        self.args = args 
        self.main_thread = (jax.process_index() == 0)
        self.out_dir = os.path.join('output', args.out_dir) 
        os.makedirs(self.out_dir, exist_ok=True)
        
        if self.main_thread:
            expt_utils.print_args(self.args)
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.jax_rng = jax.random.PRNGKey(args.seed)
        
        if self.args.half_precision:
            if platform == 'tpu':
                self.dtype = jnp.bfloat16
            else:
                self.dtype = jnp.float16
        else:
            self.dtype = jnp.float32 
            
        # Transforms and dataloaders
        if 'cifar' in args.data_name:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                data_utils.ToArray(),
                data_utils.ArrayNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            val_transform = transforms.Compose([
                transforms.Resize(32),
                data_utils.ToArray(),
                data_utils.ArrayNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dset = TORCH_DATASETS[args.data_name](root=args.data_root, train=True, transform=train_transform, download=True)
            val_dset = TORCH_DATASETS[args.data_name](root=args.data_root, train=False, transform=val_transform, download=True)
            self.n_classes = 10
            
        elif 'imagenet' in args.data_name:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                data_utils.ToArray(),
                data_utils.ArrayNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                data_utils.ToArray(),
                data_utils.ArrayNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_dset = datasets.ImageFolder(root=os.path.join(args.data_root, 'train'), transform=train_transform)
            val_dset = datasets.ImageFolder(root=os.path.join(args.data_root, 'val'), transform=val_transform)
            self.n_classes = 1000
            
        else:
            raise ValueError(f'Dataset {args.data_name} is not available')
        
        self.train_loader = data_utils.JaxDataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        self.val_loader = data_utils.JaxDataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
        self.lr_func = self.multistep_lr_schedule(steps_per_epoch=len(self.train_loader))
        
        assert args.net in NETWORKS, f'Network {args.net} is not available'
        self.model = self.create_model()
        self.state = self.create_train_state(self.model)
        if args.resume:
            self.state = restore_checkpoint(self.out_dir, self.state)
        self.state = jax_utils.replicate(self.state)                            # Required for multi-core training
        
        self.p_train_step = jax.pmap(self.train_step, axis_name='batch')
        self.p_eval_step = jax.pmap(self.eval_step, axis_name='batch')
        self.best_train, self.best_eval = 0, 0
        
        if args.batch_size % jax.local_device_count() > 0:
            raise ValueError(f'Batch size {args.batch_size} is not divisible by device count {jax.local_device_count()}')
        local_batch_size = args.batch_size // jax.process_count()
        platform = jax.local_devices()[0].platform
        
        # Logging and wandb 
        if self.main_thread:
            self.logger = expt_utils.Logger(self.out_dir)
            self.logger.print('JAX local devices: {}'.format(jax.local_devices()), 'info')
            self.log_wandb = False
            if args.wandb:
                run = wandb.init()
                self.logger.write('wandb url: {}'.format(run.get_url()))
                self.log_wandb = True
            
    def create_model(self):
        assert self.args.net in NETWORKS, f'Network {self.args.net} is not available'
        platform = jax.local_devices()[0].platform 
        
        if 'resnet' in self.args.net:
            model = NETWORKS[self.args.net](
                num_classes=self.n_classes, 
                dtype=self.dtype,
                pre_conv=self.args.pre_conv
            )
        else:
            raise NotImplementedError()
        return model
    
    def initialize_model(self, model):
        input_shape = (self.args.input_size, self.args.input_size, 3)
        @jax.jit 
        def init(*args):
            return model.init(*args)
        variables = init({'params': self.jax_rng}, jnp.ones((1, *input_shape), self.dtype))
        return variables['params'], variables['batch_stats']
    
    def create_train_state(self, model):
        dynamic_scale = None 
        platform = jax.local_devices()[0].platform 
        dynamic_scale = optim.DynamicScale() if (self.args.half_precision and platform == 'gpu') else None
        
        params, batch_stats = self.initialize_model(model)
        tx = optax.sgd(learning_rate=self.lr_func, momentum=self.args.momentum, nesterov=args.nesterov)
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
            dynamic_scale=dynamic_scale
        )
        return state
    
    def compute_metrics(self, logits, labels):
        loss = losses.cross_entropy_loss(logits, labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {'loss': loss, 'accuracy': accuracy}
        metrics = lax.pmean(metrics, axis_name='batch')
        return metrics 
    
    def cosine_lr_schedule(self, steps_per_epoch):
        warmup_fn = optax.linear_schedule(0, self.args.lr, self.args.warmup_epochs * steps_per_epoch)
        cosine_epochs = max(self.args.epochs - self.args.warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(self.args.lr, cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(
            schedules = [warmup_fn, cosine_fn],
            boundaries = [self.args.warmup_epochs * steps_per_epoch]
        ) 
        return schedule_fn
    
    def multistep_lr_schedule(self, steps_per_epoch):
        milestones = [int(m) * steps_per_epoch for m in self.args.multistep_milestones.split(',')]
        b_and_s = {m: self.args.lr_decay for m in milestones}
        warmup_fn = optax.linear_schedule(0, self.args.lr, self.args.warmup_epochs * steps_per_epoch)
        piecewise_fn = optax.piecewise_constant_schedule(self.args.lr, b_and_s)
        schedule_fn = optax.join_schedules(
            schedules = [warmup_fn, piecewise_fn],
            boundaries = [self.args.warmup_epochs * steps_per_epoch]
        )
        return schedule_fn

    def train_step(self, state, batch):
        imgs, labels = batch
        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                imgs,
                mutable=['batch_stats']
            )        
            loss = losses.cross_entropy_loss(logits, labels)
            weight_penalty_params = jax.tree_leaves(params)
            weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
            weight_penalty = self.args.weight_decay * 0.5 * weight_l2 
            loss = loss + weight_penalty 
            return loss, (new_model_state, logits)
        
        step = state.step
        dynamic_scale = state.dynamic_scale
        lr = self.lr_func(step)
        
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
            dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        else:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            aux, grads = grad_fn(state.params)
            grads = lax.pmean(grads, axis_name='batch')
            
        new_model_state, logits = aux[1]
        metrics = self.compute_metrics(logits, labels)
        metrics['lr'] = lr 
        
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
                    new_state.params,
                    state.params,
                )
            )
            metrics['scale'] = dynamic_scale.scale 
        return new_state, metrics 
    
    def eval_step(self, state, batch):
        imgs, labels = batch
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logits = state.apply_fn(variables, imgs, train=False, mutable=False)
        return self.compute_metrics(logits, labels)
    
    def sync_batch_stats(self, state):
        cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
        return state.replace(batch_stats=cross_replica_mean(state.batch_stats))
    
    def save(self):
        state = jax.device_get(jax.tree_map(lambda x: x[0], self.state))
        step = int(self.state.step)
        checkpoints.save_checkpoint(self.out_dir, self.state, step, keep=3)
            
    def load(self, out_dir):
        checkpoints.restore_checkpoint(out_dir, self.state)
        
    def run(self):
        train_metrics, val_metrics = [], []
        train_last_t = time.time()
        
        for epoch in range(1, self.args.epochs+1):
            print()
            # Training loop
            for step, batch in enumerate(self.train_loader):
                batch = data_utils.shard(batch)
                self.state, metrics = self.p_train_step(self.state, batch)

                if self.main_thread and self.log_wandb and (step+1) % self.args.log_interval == 0:
                    wandb.log({'step': step+1, 'train loss': metrics['loss']})
                train_metrics.append(metrics)
                expt_utils.progress_bar((step+1)/len(self.train_loader), desc='train progress')
            print()
            train_metrics = common_utils.get_metrics(train_metrics)
            train_summary = {f'train {k}': v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()}
            train_summary['epoch time'] = time.time() - train_last_t
            
            if self.main_thread:
                print('train epoch: {:3d} | {}'.format(epoch, ' | '.join(['{}: {:.4f}'.format(k, v) for k, v in train_summary.items()]))) 
                if self.log_wandb:
                    wandb.log({'epoch': epoch, **{k: v for k, v in train_summary.items() if k != 'train loss'}})
            
                if train_summary['train accuracy'] > self.best_train:
                    self.logger.print('Train accuracy improved from {:.4f} -> {:.4f}'.format(
                        self.best_train, train_summary['train accuracy']), 'train'
                    )
                    self.best_train = train_summary['train accuracy']
            
            # Evaluation loop
            if self.main_thread and epoch % self.args.eval_every == 0 or epoch == self.args.epochs:
                print()
                self.state = self.sync_batch_stats(self.state)
                
                for step, batch in enumerate(self.val_loader):
                    batch = data_utils.shard(batch)
                    metrics = self.p_eval_step(self.state, batch)
                    val_metrics.append(metrics)
                    expt_utils.progress_bar((step+1)/len(self.val_loader), desc='eval  progress')
                print()
                val_metrics = common_utils.get_metrics(val_metrics)
                val_summary = {f'val {k}': v for k, v in jax.tree_map(lambda x: x.mean(), val_metrics).items()}
                
                print('eval  epoch: {:3d} | {}'.format(epoch, ' | '.join(['{}: {:.4f}'.format(k, v) for k, v in val_summary.items()])))
                if self.log_wandb:
                    wandb.log({'epoch': epoch, **val_summary})
                
                if val_summary['val accuracy'] > self.best_eval:
                    self.logger.print('Val accuracy improved from {:.4f} -> {:.4f}'.format(
                        self.best_eval, val_summary['val accuracy']), 'val'
                    )
                    self.best_eval = val_summary['val accuracy']
                    self.save()
                
            train_metrics, val_metrics = [], []
            train_last_t = time.time() 
            print()
            self.logger.record('Epoch: {:3d} | Best train: {:.4f} | Best eval: {:.4f}'.format(epoch, self.best_train, self.best_eval), 'info')
            print('--------------------------------------------------------------')
        
        # Wait until all processes have completed
        jax.random.normal(jax.random.PRNGKey(args.seed), ()).block_until_ready()
        
        
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default=1, type=int, help='random seed')
    ap.add_argument('--out_dir', default=dt.now().strftime('%d-%m-%Y_%H-%M'), type=str, help='working directory path')
    ap.add_argument('--data_name', default='cifar10', type=str, help='dataset')
    ap.add_argument('--data_root', default='~/Datasets/cifar10', type=str, help='directory where dataset is stored')
    ap.add_argument('--batch_size', default=128, type=int, help='batch size')
    ap.add_argument('--n_workers', default=4, type=int, help='dataloading worker count')
    ap.add_argument('--net', default='resnet18', type=str, help='model architecture')
    ap.add_argument('--pre_conv', action='store_true', help='reduced or full preconv for resnet')
    ap.add_argument('--input_size', default=32, type=int, help='image shape')
    ap.add_argument('--half_precision', action='store_true', help='float16 precision training')
    ap.add_argument('--lr', default=0.1, type=float, help='learning rate')
    ap.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay for multistep')
    ap.add_argument('--multistep_milestones', default='30,60', type=str, help='milestones at which multistep lr decay happens')
    ap.add_argument('--momentum', default=0.9, type=float, help='optimizer sgd momentum')
    ap.add_argument('--nesterov', action='store_true', help='sgd nesterov accelerated gradients')
    ap.add_argument('--epochs', default=100, type=int, help='training epochs')
    ap.add_argument('--weight_decay', default=1e-06, type=float, help='weight decay')
    ap.add_argument('--warmup_epochs', default=0, type=int, help='linear LR warmup epochs')
    ap.add_argument('--log_interval', default=10, type=int, help='number of steps to log after')
    ap.add_argument('--eval_every', default=1, type=int, help='number of epochs to eval after')
    ap.add_argument('--resume', action='store_true', help='whether to resume training from working directory')
    ap.add_argument('--wandb', action='store_true', help='enable wandb logging for experiment')
    args = ap.parse_args()

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Train
    trainer = Trainer(args)
    trainer.run()
    
    