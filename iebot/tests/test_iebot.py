from iebot.iebot import *


def test_translate_board():
    assert translate_board([0] * 42).shape == (6, 7)


def test_analyze_column():
    # Output is a tuple containing (top_piece, counter, num_spaces)
    assert analyze_column([2, 2, 2, 1, 1, 2, 2]) == (2, 3, 0)
    assert analyze_column([0, 2, 2, 1, 1, 2, 2]) == (2, 2, 1)
    assert analyze_column([0, 0, 1, 1, 1, 2, 2]) == (1, 3, 2)
