"""
This version is intended to implement transposition tables.

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""
import numpy as np

from iebot.tranposition_table_8_ply import tranposition_table


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
    bit_board, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel
        bit_board += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i, j] != 0]
            bit_board += [b'0', b'1'][board[i, j] == player]

    return int(bit_board, 2), int(mask, 2)


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


def negamax_ab(node, max_depth, columns_map, alpha=-float('inf'), beta=float('inf'), root=False):
    if max_depth == 0 or node.value != 0:
        return [-node.value - (0.01 * max_depth), node.play]  # Giving higher score to shallow nodes
    elif not root and (node.bit_board, node.mask) in tranposition_table:
        return [-tranposition_table[(node.bit_board, node.mask)], node.play]

    max_value, play = -float('inf'), -1

    for child in node.create_children(columns_map=columns_map):

        result = negamax_ab(node=child, max_depth=max_depth - 1, columns_map=columns_map, alpha=-beta, beta=-alpha)

        if -result[0] > max_value:
            max_value = -result[0]
            play = child.play
            alpha = max(alpha, max_value)

        if alpha >= beta:
            break

    if play == -1:  # Happens only when there are no more pieces left and game is tied
        tranposition_table[(node.bit_board ^ node.mask, node.mask)] = max_value
        return [0, play]

    elif max_value != 0:
        tranposition_table[(node.bit_board ^ node.mask, node.mask)] = max_value

    return [max_value, play]


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)

    def create_children(self, columns_map):
        plays = generate_plays(self.mask, columns_map)
        plays = [plays.pop(i // 2) for i in reversed(range(len(plays)))]  # Order by the center

        for play in plays:
            new_bit_board = (self.mask ^ self.bit_board) | play
            new_mask = self.mask | play
            node = Node(bit_board=new_bit_board,
                        mask=new_mask,
                        play=play)
            yield node


def my_agent(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)
    columns_map = generate_columns_map(mask)

    node = Node(bit_board ^ mask, mask)

    _, play = negamax_ab(node=node, max_depth=8, columns_map=columns_map, root=True)
    return transform_play_to_column(play=play, columns_map=columns_map)
