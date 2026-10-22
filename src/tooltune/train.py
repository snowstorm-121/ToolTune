import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset


def training_dataset(examples, tokenizer):
    return CacheDataset(ChatDataset(examples, tokenizer, mask_prompt=True))


def assistant_loss(model, batch, lengths):
    targets = batch[:, 1:]
    logits = model(batch[:, :-1])
    positions = mx.arange(1, batch.shape[1])
    # lengths = [prompt boundary, unpadded token count]. Targets use shifted indices.
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
    n = mask.sum()
    loss = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / n
    return loss, n
