from iebot.iebot_v1 import *
import numpy as np


def test_get_column_summary():
    # Output is a tuple containing (counter, num_spaces)
    assert get_column_summary([2, 2, 2, 1, 2, 2]) == (3, 0)
    assert get_column_summary([0, 2, 2, 1, 2, 2]) == (2, 1)
    assert get_column_summary([0, 0, 1, 1, 1, 2]) == (3, 2)


def test_get_vertical_summary():
    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 1, 1, 0, 1, 1],
                             [0, 0, 1, 2, 0, 2, 2],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 0, 1, 2, 0, 1, 2]])

    sample_result = np.array([[2, 4], [0, 6], [3, 1], [1, 0], [0, 6], [1, 1], [1, 1]])

    result = get_vertical_summary(sample_board)

    np.testing.assert_array_equal(result, sample_result)


def test_get_row_summary():
    assert get_row_summary([0, 0, 0, 0, 0, 0, 0]) == [0, 0]
    assert get_row_summary([0, 0, 0, 0, 1, 0, 0]) == [1, 4]
    assert get_row_summary([1, 1, 1, 0, 1, 1, 0]) == [3, 2]
    assert get_row_summary([2, 2, 2, 0, 1, 1, 0]) == [3, 2]
    assert get_row_summary([2, 2, 2, 0, 2, 2, 0]) == [3, 2]
    assert get_row_summary([2, 1, 2, 0, 2, 2, 0]) == [2, 5]


def test_get_horizontal_summary():
    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 1],
                             [0, 0, 1, 2, 0, 2, 2],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 1, 1, 1, 0, 1, 2]])

    sample_result = np.array([[1, 3], [2, 6], [2, 6], [2, 3], [2, 3], [3, 3]])
    result = get_horizontal_summary(sample_board)

    np.testing.assert_array_equal(result, sample_result)


def test_horizontal_play():
    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 1],
                             [0, 0, 2, 2, 0, 2, 2],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 1, 1, 1, 0, 1, 2]])

    horizontal_summary = get_horizontal_summary(sample_board)
    vertical_summary = get_vertical_summary(sample_board)

    assert horizontal_play(horizontal_summary=horizontal_summary, vertical_summary=vertical_summary) == 4


def test_vertical_play():
    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 2],
                             [0, 0, 2, 2, 0, 2, 2],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 0, 1, 1, 0, 1, 2]])

    vertical_summary = get_vertical_summary(sample_board)

    assert vertical_play(vertical_summary=vertical_summary) == 6

    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 2, 2, 0, 2, 0],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 0, 1, 1, 0, 1, 2]])

    vertical_summary = get_vertical_summary(sample_board)

    assert vertical_play(vertical_summary=vertical_summary) == 0

    sample_board = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 2, 2, 0, 2, 0],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 0, 1, 1, 0, 1, 2]])

    vertical_summary = get_vertical_summary(sample_board)

    assert vertical_play(vertical_summary=vertical_summary) == 3
