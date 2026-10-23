import pytest
import torch

from diagnostics import compute_energy_distance, compute_ess, compute_mmd_rbf


def test_ess_equal_weights_is_n():
    theta = torch.randn(200, 3)
    log_w = torch.zeros(200)
    ess, ess_n = compute_ess(theta, log_w, log_w)
    assert ess == pytest.approx(200)
    assert ess_n == pytest.approx(1.0)


def test_ess_single_dominant_weight_is_one():
    theta = torch.randn(200, 3)
    log_w = torch.full((200,), -1e3)
    log_w[0] = 0.0
    ess, _ = compute_ess(theta, log_w, torch.zeros(200))
    assert ess == pytest.approx(1.0)


def test_mmd_and_energy_distance_vanish_on_identical_samples():
    x = torch.randn(300, 4)
    assert compute_mmd_rbf(x, x) == pytest.approx(0.0, abs=1e-6)
    assert compute_energy_distance(x, x) == pytest.approx(0.0, abs=1e-5)


def test_mmd_and_energy_distance_detect_a_shift():
    torch.manual_seed(0)
    x = torch.randn(500, 4)
    y_same = torch.randn(500, 4)
    y_shift = torch.randn(500, 4) + 1.0
    assert compute_mmd_rbf(x, y_shift) > 10 * compute_mmd_rbf(x, y_same)
    assert compute_energy_distance(x, y_shift) > 10 * compute_energy_distance(x, y_same)
