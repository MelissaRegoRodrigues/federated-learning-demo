from flapp.free_rider import is_free_rider


def test_first_clients_are_free_riders():
    assert is_free_rider(0, num_free_riders=3)
    assert is_free_rider(2, num_free_riders=3)
    assert not is_free_rider(3, num_free_riders=3)


def test_zero_free_riders_means_everyone_is_honest():
    assert not is_free_rider(0, num_free_riders=0)
