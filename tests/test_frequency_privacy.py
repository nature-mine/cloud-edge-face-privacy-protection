import cv2
import numpy as np
import pytest
import torch
from torch import nn

from frequency_privacy import AttackConfig, block_dct, block_idct, frequency_budget, protect_face
from privacy_demo import inverse_warp_delta, quantize_protected


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(17)
    torch.set_num_threads(2)


def test_dct_matches_opencv_and_roundtrips_rgb():
    image = torch.rand(2, 3, 16, 24, dtype=torch.float64, requires_grad=True)
    coefficients = block_dct(image)
    reference = cv2.dct(image.detach().numpy()[0, 1, 8:16, 16:24])
    np.testing.assert_allclose(coefficients.detach().numpy()[0, 1, 1, 2], reference, atol=1e-12)
    torch.testing.assert_close(block_idct(coefficients), image, atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda x: block_idct(block_dct(x)), (image[:, :, :8, :8],))


@pytest.mark.parametrize("bands", ["high", "mid-high"])
def test_attack_reduces_similarity_and_preserves_constraints(bands):
    image = torch.rand(1, 3, 16, 16) * 0.7 + 0.15
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 2:14, 2:14] = 0.8
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 32))
    config = AttackConfig(steps=10, bands=bands)
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    result = protect_face(image, mask, model, config)
    assert result.cosine_after < result.cosine_before - 1e-5
    assert (result.image - image).abs().max() <= config.epsilon + 1e-7
    assert torch.equal(result.image[mask.expand_as(image) == 0], image[mask.expand_as(image) == 0])
    budget = frequency_budget(config, image).expand_as(result.delta)
    assert torch.count_nonzero(result.delta[budget == 0]) == 0
    assert torch.all(result.delta.abs() <= budget + 1e-7)
    assert result.image.min() >= 0 and result.image.max() <= 1
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], atol=0, rtol=0)
    repeated = protect_face(image, mask, model, config)
    torch.testing.assert_close(result.image, repeated.image, atol=0, rtol=0)


@pytest.mark.parametrize("case", ["empty-mask", "zero-steps", "zero-budget"])
def test_noop_cases_are_exact(case):
    image = torch.rand(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8) if case == "empty-mask" else torch.ones(1, 1, 8, 8)
    config = AttackConfig(steps=0 if case == "zero-steps" else 2, epsilon=0 if case == "zero-budget" else 8 / 255)
    model = nn.Sequential(nn.Flatten(), nn.Linear(192, 8))
    result = protect_face(image, mask, model, config)
    assert torch.equal(result.image, image)
    assert torch.count_nonzero(result.delta) == 0


def test_inverse_warp_does_not_resample_original_image():
    original = np.random.default_rng(17).integers(0, 256, (35, 29, 3), dtype=np.uint8)
    transform = np.array([[0.9, 0.1, 2.3], [-0.1, 0.9, 1.7]], dtype=np.float32)
    zero_delta = np.zeros((16, 16, 3), dtype=np.float32)
    warped = inverse_warp_delta(original, zero_delta, transform)
    np.testing.assert_array_equal(quantize_protected(original, warped, 8 / 255), original)
    changed = quantize_protected(original, np.ones_like(original) * 10.0, 8 / 255)
    assert np.abs(changed.astype(np.int16) - original.astype(np.int16)).max() <= 8


def test_invalid_config_and_shapes_are_rejected():
    with pytest.raises(ValueError):
        AttackConfig(epsilon=-1)
    with pytest.raises(ValueError):
        AttackConfig(step_size=float("nan"))
    with pytest.raises(ValueError):
        block_dct(torch.zeros(1, 3, 9, 8))
    with pytest.raises(ValueError):
        block_idct(torch.zeros(1, 3, 8, 8))
