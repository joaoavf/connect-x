# cythonize -a -i utils_cy.pyx
import numpy as np

cpdef translate_board(board):
    """Translate a 42 items flat list into a 6x7 numpy array.

    Parameters:
    board (list): 42 items (not nested) mapping the board (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)"""

    return np.array(board).reshape(6, 7)

cpdef bint connected_four(unsigned long long bit_board):
    """Evaluates if player bit board has made a connect 4.

    Parameters:
    bit_board (int): bit board representation of player pieces the game

    Returns:
    bool : True if the board has achieved a connect 4"""

    cdef unsigned long long m
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

cpdef get_position_mask_bitmap(board, player):
    """Transform a 6x7 board representation into bit boards.

    Parameters:
    board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    (int, int) : (bit board of player pieces, bit board of all pieces)"""

    # TODO fix deprecation warnings
    player_pieces, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel
        player_pieces += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i, j] != 0]
            player_pieces += [b'0', b'1'][board[i, j] == player]

    return int(player_pieces, 2), int(mask, 2)

cpdef list generate_plays(unsigned long long mask):
    """Generate a list with all the possible plays in a given round.

    Parameters:
    mask (int): binary representation (bit board) of all pieces

    Returns:
    List : bit value of all available plays"""
    cdef list valid_plays
    cdef list column_values
    cdef int column_number
    cdef unsigned long long value
    cdef list POSSIBLE_MOVES

    POSSIBLE_MOVES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
                      262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728,
                      268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368,
                      68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552,
                      4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328,
                      281474976710656]

    valid_plays = []
    for column_number in range(7):
        column_values = POSSIBLE_MOVES[7 * column_number: 7 * column_number + 6]  # Minus extra cell on top of the board
        for value in column_values:
            if mask & value == 0:
                valid_plays.append(value)
                break
    return valid_plays

cpdef unsigned long long transform_play_to_column(unsigned long long play):
    """Return position of the column where the play was made.

    Parameters:
    play         (int): bit board representation of a piece

    Returns:
    int : column position"""
    cdef unsigned long long i
    POSSIBLE_MOVES = [2 ** i for i in range(49)]

    return POSSIBLE_MOVES.index(play) // 7
