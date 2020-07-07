from iebot.utils import translate_board


def test_translate_board():
    assert translate_board([0] * 42).shape == (6, 7)
