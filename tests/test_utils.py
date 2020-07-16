from engine.utils import *


def test_translate_board():
    assert translate_board([0] * 42).shape == (6, 7)


def test_get_position_mask_bitmap():
    board_0 = np.array([[0, 0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1],
                        [0, 0, 2, 2, 0, 2, 2],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 1, 1, 1, 0, 1, 2]])

    assert get_position_mask_bitmap(board=board_0, player=1) == (79955155304576, 137404726100099)
    assert get_position_mask_bitmap(board=board_0, player=2) == (57449570795523, 137404726100099)

    board_1 = np.array([[0, 0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 2],
                        [0, 0, 2, 2, 0, 2, 2],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 0, 1, 1, 0, 1, 2]])

    assert get_position_mask_bitmap(board=board_1, player=1) == (9586411126784, 137404726099971)
    assert get_position_mask_bitmap(board=board_1, player=2) == (127818314973187, 137404726099971)

    board_2 = np.array([[0, 0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 2, 2, 0, 2, 0],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 0, 1, 1, 0, 1, 2]])

    assert get_position_mask_bitmap(board=board_2, player=1) == (9586411126784, 31851609833475)
    assert get_position_mask_bitmap(board=board_2, player=2) == (22265198706691, 31851609833475)

    board_3 = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 2, 2, 0, 2, 0],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 0, 1, 1, 0, 1, 2]])

    assert get_position_mask_bitmap(board=board_3, player=1) == (9586411126784, 31851542724611)
    assert get_position_mask_bitmap(board=board_3, player=2) == (22265131597827, 31851542724611)

    board_4 = np.array([[0, 0, 0, 0, 0, 0, 0]] * 6)

    assert get_position_mask_bitmap(board=board_4, player=1) == (0, 0)
    assert get_position_mask_bitmap(board=board_4, player=2) == (0, 0)


def test_connected_four():
    board_0 = np.array([[0, 0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1],
                        [0, 0, 2, 2, 0, 2, 2],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 1, 1, 1, 0, 1, 2]])

    assert connected_four(get_position_mask_bitmap(board=board_0, player=1)[0]) is False
    assert connected_four(get_position_mask_bitmap(board=board_0, player=2)[0]) is False

    board_1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 2],
                        [0, 0, 2, 1, 2, 2, 2],
                        [0, 0, 1, 2, 2, 1, 2],
                        [2, 0, 2, 2, 1, 1, 1],
                        [2, 2, 1, 1, 2, 1, 2]])

    assert connected_four(get_position_mask_bitmap(board=board_1, player=1)[0]) is False
    assert connected_four(get_position_mask_bitmap(board=board_1, player=2)[0]) is True

    board_2 = np.array([[0, 0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 1, 0, 2, 0],
                        [0, 0, 2, 2, 0, 1, 0],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 0, 1, 1, 0, 1, 2]])

    assert connected_four(get_position_mask_bitmap(board=board_2, player=1)[0]) is True
    assert connected_four(get_position_mask_bitmap(board=board_2, player=2)[0]) is False

    board_3 = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 2, 2, 0, 2, 0],
                        [0, 0, 1, 1, 0, 1, 2],
                        [2, 0, 2, 2, 0, 1, 1],
                        [2, 0, 1, 1, 1, 1, 2]])

    assert connected_four(get_position_mask_bitmap(board=board_3, player=1)[0]) is True
    assert connected_four(get_position_mask_bitmap(board=board_3, player=2)[0]) is False

    board_4 = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0],
                        [0, 2, 2, 2, 0, 2, 0],
                        [0, 2, 2, 1, 0, 1, 2],
                        [2, 1, 2, 2, 0, 1, 1],
                        [2, 2, 1, 1, 2, 1, 2]])

    assert connected_four(get_position_mask_bitmap(board=board_4, player=1)[0]) is False
    assert connected_four(get_position_mask_bitmap(board=board_4, player=2)[0]) is True

    board_5 = np.array([[0, 0, 0, 0, 0, 0, 0]] * 6)

    assert connected_four(get_position_mask_bitmap(board=board_5, player=1)[1]) is False
    assert connected_four(get_position_mask_bitmap(board=board_5, player=2)[1]) is False
