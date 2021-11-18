
import jax
import torch
import numpy as np
import jax.numpy as jnp
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

    
class ToArray:
    
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        x /= 255.0 
        return x 
    
    
class ArrayNormalize:
    
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, x):
        mean = np.asarray(self.mean, dtype=x.dtype)
        std = np.asarray(self.std, dtype=x.dtype)
        if mean.ndim == 1:
            mean = mean.reshape(1, 1, -1)
        if std.ndim == 1:
            std = std.reshape(1, 1, -1)
        x -= mean
        x /= std
        return x
    
    
class JaxDataLoader(DataLoader):
    
    def __init__(
        self, 
        dataset, 
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=jax_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )
        
def jax_collate(batch):
    imgs, targets = zip(*batch)
    return np.stack(imgs), np.array(targets)
    
def shard(xs):
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]) if len(x.shape) != 0 else x, xs)