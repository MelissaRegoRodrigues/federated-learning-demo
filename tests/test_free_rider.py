import numpy as np
import pytest

from flapp.free_rider import assign_behavior, apply_behavior


def test_assign_behavior_cycles_silent_noisy_lazy_then_honest():
    assert assign_behavior(0, num_free_riders=3) == "silent"
    assert assign_behavior(1, num_free_riders=3) == "noisy"
    assert assign_behavior(2, num_free_riders=3) == "lazy"
    assert assign_behavior(3, num_free_riders=3) == "honest"


def test_assign_behavior_zero_free_riders_is_all_honest():
    assert assign_behavior(0, num_free_riders=0) == "honest"


def test_apply_behavior_honest_is_identity():
    params = [np.array([1.0, 2.0, 3.0])]
    result = apply_behavior(params, "honest")
    np.testing.assert_array_equal(result[0], params[0])


def test_apply_behavior_silent_is_all_zeros():
    params = [np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]])]
    result = apply_behavior(params, "silent")
    for arr in result:
        assert np.all(arr == 0)


def test_apply_behavior_noisy_changes_values_same_shape():
    rng = np.random.default_rng(0)
    params = [np.ones((4,))]
    result = apply_behavior(params, "noisy", rng=rng)
    assert result[0].shape == params[0].shape
    assert not np.array_equal(result[0], params[0])


def test_apply_behavior_lazy_scales_down():
    params = [np.array([10.0, 20.0])]
    result = apply_behavior(params, "lazy")
    np.testing.assert_allclose(result[0], params[0] * 0.3)


def test_apply_behavior_unknown_raises():
    with pytest.raises(ValueError):
        apply_behavior([np.array([1.0])], "malicious")
