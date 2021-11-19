
import jax 
import optax
import jax.numpy as jnp 
from flax.training import common_utils


def cross_entropy_loss(logits, labels):
    onehot_labels = common_utils.onehot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=onehot_labels)
    return jnp.mean(loss)

def mean_squared_loss(logits, labels):
    loss = optax.l2_loss(logits, labels)
    return jnp.mean(loss)

def smooth_l1_loss(logits, labels):
    loss = optax.huber_loss(logits, labels)
    return jnp.mean(loss)