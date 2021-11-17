
import torch
import numpy as np
import jax.numpy as jnp
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def jax_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [jax_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)
    
    
class DataTransform:
    
    def __init__(self, transform=None):
        self.transform = transform
    
    def __call__(self, img):
        img = self.transform(img).permute(1, 2, 0).numpy()
        return jnp.asarray(img, dtype=jnp.float32)
    
    
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