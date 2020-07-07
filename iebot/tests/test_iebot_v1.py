from iebot.iebot_v1 import *


def test_get_column_summary():
    # Output is a tuple containing (top_piece, counter, num_spaces)
    assert get_column_summary([2, 2, 2, 1, 2, 2]) == (2, 3, 0)
    assert get_column_summary([0, 2, 2, 1, 2, 2]) == (2, 2, 1)
    assert get_column_summary([0, 0, 1, 1, 1, 2]) == (1, 3, 2)


def test_get_vertical_summary():
    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 1, 1, 0, 1, 1],
                             [0, 0, 1, 2, 0, 2, 2],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 0, 1, 2, 0, 1, 2]])

    sample_result = [{'top_piece': 2, 'counter': 2, 'num_spaces': 4},
                     {'top_piece': 0, 'counter': 0, 'num_spaces': 6},
                     {'top_piece': 1, 'counter': 3, 'num_spaces': 1},
                     {'top_piece': 2, 'counter': 1, 'num_spaces': 0},
                     {'top_piece': 0, 'counter': 0, 'num_spaces': 6},
                     {'top_piece': 1, 'counter': 1, 'num_spaces': 1},
                     {'top_piece': 1, 'counter': 1, 'num_spaces': 1}]

    assert get_vertical_summary(sample_board) == sample_result


def test_get_row_summary():
    assert get_row_summary([0, 0, 0, 0, 0, 0, 0]) == [0, 0]
    assert get_row_summary([0, 0, 0, 0, 1, 0, 0]) == [1, 4]
    assert get_row_summary([1, 1, 1, 0, 1, 1, 0]) == [3, 2]
    assert get_row_summary([2, 2, 2, 0, 1, 1, 0]) == [3, 2]
    assert get_row_summary([2, 2, 2, 0, 2, 2, 0]) == [3, 2]
    assert get_row_summary([2, 1, 2, 0, 2, 2, 0]) == [2, 5]


def test_get_horizontal_summary():
    sample_board = np.array([[0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 1, 1, 0, 1, 1],
                             [0, 0, 1, 2, 0, 2, 2],
                             [0, 0, 1, 1, 0, 1, 2],
                             [2, 0, 2, 2, 0, 1, 1],
                             [2, 0, 1, 2, 0, 1, 2]])

    sample_result = [[1, 3], [2, 3], [2, 6], [2, 3], [2, 3], [1, 0]]

    assert get_horizontal_summary(sample_board) == sample_result
