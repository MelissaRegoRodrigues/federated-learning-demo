import pytest

from flapp import economics

MODEL_SIZE = 18406


def make_clients(count=4):
    params = {}
    sizes = {}
    for client_id in range(count):
        params[client_id] = economics.client_parameters(42, client_id)
        sizes[client_id] = 3000
    return params, sizes


def test_client_parameters_are_inside_the_table_ranges():
    params = economics.client_parameters(42, 3)
    assert 3000 <= params["computing_power"] <= 5000
    assert 30 <= params["transfer_limit"] <= 60
    assert 0.3 <= params["energy_efficiency"] <= 0.5
    assert 0.05 <= params["preprocessing"] <= 0.07


def test_client_parameters_are_reproducible():
    assert economics.client_parameters(42, 1) == economics.client_parameters(42, 1)


def test_unconstrained_datasize_is_where_the_marginal_utility_is_zero():
    params = economics.client_parameters(42, 0)
    datasize = economics.unconstrained_datasize(2.0, 1, params, MODEL_SIZE)
    step = 1e-3
    left = economics.client_utility(datasize - step, 2.0, 1, params, MODEL_SIZE)
    right = economics.client_utility(datasize + step, 2.0, 1, params, MODEL_SIZE)
    middle = economics.client_utility(datasize, 2.0, 1, params, MODEL_SIZE)
    assert middle >= left and middle >= right


def test_best_datasize_is_zero_when_the_price_does_not_pay_the_costs():
    params = economics.client_parameters(42, 0)
    assert economics.best_datasize(0.2, 1, params, MODEL_SIZE, 5000) == 0


def test_best_datasize_never_exceeds_the_local_dataset():
    params = economics.client_parameters(42, 0)
    assert economics.best_datasize(50.0, 1, params, MODEL_SIZE, 100) == 100


def test_solved_price_satisfies_the_first_order_condition():
    params, sizes = make_clients()
    average = economics.average_parameters(list(params.values()))
    price = economics.solve_price(4, 1.0e5, average, params[0], MODEL_SIZE)
    assert economics.price_condition(price, 4, 1.0e5, average, params[0], MODEL_SIZE) == pytest.approx(0.0, abs=1e-3)


def test_calibrated_time_gives_exactly_the_requested_number_of_rounds():
    params, sizes = make_clients()
    total_time = economics.calibrate_total_time(10, params, sizes, MODEL_SIZE)
    assert economics.time_for_rounds(total_time, 10, params, sizes, MODEL_SIZE) <= total_time
    assert economics.time_for_rounds(total_time, 11, params, sizes, MODEL_SIZE) > total_time
