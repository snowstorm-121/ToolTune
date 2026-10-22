import mlx.core as mx
import mlx.nn as nn

from tooltune.train import assistant_loss


def test_padding_and_prompt_excluded_from_shifted_loss():
    batch = mx.array([[1, 2, 3, 4, 0, 0, 0]])
    lengths = mx.array([[2, 4]])
    logits = mx.zeros((1, 6, 8))
    loss, n = assistant_loss(lambda x: logits, batch, lengths)
    assert n.item() == 2
    assert abs(loss.item() - 2.0794415) < 1e-5


def test_original_upstream_loss_includes_extra_padding_token():
    from mlx_lm.tuner.trainer import default_loss
    _, n = default_loss(lambda x: mx.zeros((1, 6, 8)),
                        mx.array([[1, 2, 3, 4, 0, 0, 0]]), mx.array([[2, 4]]))
    assert n.item() == 3  # Pins the observed upstream behavior at mlx-lm 0.32.2.


def test_only_target_positions_contribute_to_loss():
    batch = mx.array([[1, 2, 3, 4, 0, 0]])
    logits = mx.arange(40, dtype=mx.float32).reshape(1, 5, 8) / 10
    loss, _ = assistant_loss(lambda x: logits, batch, mx.array([[2, 4]]))
    expected = nn.losses.cross_entropy(logits[:, 1:3], mx.array([[3, 4]])).mean()
    assert abs(loss.item() - expected.item()) < 1e-6
