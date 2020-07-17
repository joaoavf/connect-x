import numpy as np


def translate_board(board):
    """Translate a 42 items flat list into a 6x7 numpy array.

    Parameters:
    board (list): 42 items (not nested) mapping the board (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)"""

    return np.array(board).reshape(6, 7).tolist()


def connected_four(bit_board):
    """Evaluates if player bit board has made a connect 4.

    Parameters:
    bit_board (int): bit board representation of player pieces the game

    Returns:
    bool : True if the board has achieved a connect 4"""

    # Horizontal check
    m = bit_board & (bit_board >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = bit_board & (bit_board >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = bit_board & (bit_board >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = bit_board & (bit_board >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False


def get_position_mask_bitmap(board, player):
    """Transform a 6x7 board representation into bit boards.

    Parameters:
    board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    (int, int) : (bit board of player pieces, bit board of all pieces)"""

    player_pieces, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel
        player_pieces += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i][j] != 0]
            player_pieces += [b'0', b'1'][board[i][j] == player]

    return int(player_pieces, 2), int(mask, 2)


def generate_plays(mask, order_by_mid=False):
    """Generate a list with all the possible plays in a given round.

    Parameters:
    mask (int): binary representation (bit board) of all pieces

    Returns:
    List : bit value of all available plays"""

    position_map = [2 ** i for i in range(49)]  # List of a binary representation of individual pieces in the board

    available_plays = []
    for column_number in range(7):
        column_values = position_map[7 * column_number: 7 * column_number + 6]  # Minus extra cell on top of the board
        for value in column_values:
            if mask & value == 0:
                available_plays.append(value)
                break

    if order_by_mid:
        available_plays = [available_plays.pop(i // 2) for i in reversed(range(len(available_plays)))]

    return available_plays


def transform_play_to_column(play):
    """Return position of the column where the play was made.

    Parameters:
    play         (int): bit board representation of a piece

    Returns:
    int : column position"""

    return [2 ** i for i in range(49)].index(play) // 7
