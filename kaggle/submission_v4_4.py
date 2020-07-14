import numpy as np


def translate_board(board):
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
        column_values = [cell_value for cell_value in column_values if mask & cell_value == 0]  # Removing full columns
        columns_map.append(column_values)

    return columns_map


def transform_play_to_column(play, columns_map):
    """Return position of the column where the play was made..

       Returns:
       int : column position"""

    for index, column in enumerate(columns_map):
        if play in column:
            return index


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def generate_plays(mask, columns_map):
    plays = [can_play(mask, column) for column in columns_map]
    return [play for play in plays if play]  # Remove cases of play = 0


def negamax_ab(node, depth, columns_map, alpha=-float('inf'), beta=float('inf')):
    if depth == 0 or node.value != 0:
        return [-node.value - (0.01 * depth), node.play]  # Giving higher score to higher depth (less shallow)
    # TODO rewrite code so more depth != more shallow

    max_value, play = -float('inf'), -1

    for child in node.create_children(columns_map=columns_map):

        result = negamax_ab(node=child, depth=depth - 1, columns_map=columns_map, alpha=-beta, beta=-alpha)

        if -result[0] > max_value:
            max_value = -result[0]
            play = child.play
            alpha = max(alpha, max_value)

        if alpha >= beta:
            break

    if play == -1:  # If ran out of pieces in the board and game is tied
        return [0, play]
    else:
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

    _, play = negamax_ab(node=node, depth=7, columns_map=columns_map)
    return transform_play_to_column(play=play, columns_map=columns_map)
