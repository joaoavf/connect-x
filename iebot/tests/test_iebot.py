from iebot.iebot_v1 import *


def test_get_vertical_summary():
    # Output is a tuple containing (top_piece, counter, num_spaces)
    assert get_vertical_summary([2, 2, 2, 1, 1, 2, 2]) == (2, 3, 0)
    assert get_vertical_summary([0, 2, 2, 1, 1, 2, 2]) == (2, 2, 1)
    assert get_vertical_summary([0, 0, 1, 1, 1, 2, 2]) == (1, 3, 2)
