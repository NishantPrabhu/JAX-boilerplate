
import torch
import jax.numpy as jnp
from torchvision import datasets
from vision_augs import get_transform
from torch.utils.data import Dataset, DataLoader

DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def collate_func(batch):
    """ Converts images/labels to NHWC jax.numpy DeviceArrays and returns batch items as dict """
    images, labels = zip(*batch)
    images, labels = torch.stack(images, 0).permute(0, 2, 3, 1).numpy(), torch.tensor(labels).long().numpy()
    images, labels = jnp.asarray(images), jnp.asarray(labels)
    return {"img": images, "label": labels}

def prepare_standard_dataset(name, transforms, download_root=None):
    root = f"data/{name}" if download_root is None else download_root
    assert name in DATASETS.keys(), f"Only {list(DATASETS.keys())} are available as of now"
    train_dset = DATASETS.get(name)(root=root, train=True, transform=get_transform(transforms["train"]), download=True)
    test_dset = DATASETS.get(name)(root=root, train=False, transform=get_transform(transforms["test"]), download=True)
    return train_dset, test_dset

def get_multi_epoch_loaders(train_dset, test_dset, batch_size, num_workers):
    train_loader = MultiEpochsDataLoader(train_dset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, collate_fn=collate_func)
    test_loader = MultiEpochsDataLoader(test_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True, collate_fn=collate_func)
    return train_loader, test_loader
