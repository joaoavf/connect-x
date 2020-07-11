import numpy as np


def translate_board(board):
    """Translate a 42 items flat list into a 6x7 numpy array.

    Parameters:
    board (list): 42 items (not nested) mapping the board (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)"""
    return np.array(board).reshape(6, 7)


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

    # TODO fix deprecation warnings
    position, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel # TODO understand why?
        position += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i, j] != 0]
            position += [b'0', b'1'][board[i, j] == player]

    return int(position, 2), int(mask, 2)


def generate_columns_map(mask):
    """Generates all valid moves per column on the Connect4 Board.

    Parameters:
    mask (int): binary representation (bit board) of all pieces

    Returns:
    List[List[int, ...]] : each nested listed correspond to column and its valid pieces left in binary representation"""

    position_map = [2 ** i for i in range(49)]  # List of a binary representation of individual pieces in the board

    columns_map = []
    for column_number in range(7):
        column_values = position_map[7 * column_number: 7 * column_number + 6]  # Minus extra cell on top of the board
        column_values = [value for value in column_values if mask & value == 0]  # Removing full columns
        columns_map.append(column_values)

    return columns_map


def transform_play_to_column(play, columns_map):
    """Return position of the column where the play was made.

    Parameters:
    play         (int): bit board representation of next play
    columns_map (list): maps bit board cells to their respective columns and position

    Returns:
    int : column position"""

    for index, column in enumerate(columns_map):
        if play in column:
            return index


def can_play(mask, column):
    """Evaluate if there is an available play to be made in the given column.

    Parameters:
    mask         (int): binary representation (bit board) of all pieces
    column      (list): bit board cell representations for a given column

    Returns:
    int : 0 if no element was found else the element number"""

    for element in column:
        if element & mask == 0:
            return element
    return 0


def generate_plays(mask, columns_map):
    """Generate a list with all the possible plays in a given round.

    Parameters:
    mask          (int): binary representation (bit board) of all pieces
    columns_map  (list): maps bit board cells to their respective columns and position

    Returns:
    list : column position"""

    plays = [can_play(mask, column) for column in columns_map]
    return [play for play in plays if play]  # Remove cases of play = 0
