from iebot.iebot_v2 import *


def test_process_interval():
    assert process_interval(np.array([0] * 42), list(range(42))) == []

    assert process_interval(np.array([0, 2, 2, 2]), list(range(4))) == [{'free_space_id': 0, 'is_my_advantage': False}]

    assert process_interval(np.array([0, 1, 1, 1]), list(range(4))) == [{'free_space_id': 0, 'is_my_advantage': True}]

    assert process_interval(np.array([1, 0, 1, 1]), list(range(4))) == [{'free_space_id': 1, 'is_my_advantage': True}]

    assert process_interval(np.array([0, 0, 1, 1, 1]), list(range(5))) == [
        {'free_space_id': 1, 'is_my_advantage': True}]

    assert process_interval(np.array([0, 0, 1, 1, 1, 0]), list(range(2, 8))) == [
        {'free_space_id': 3, 'is_my_advantage': True},
        {'free_space_id': 7, 'is_my_advantage': True}]
