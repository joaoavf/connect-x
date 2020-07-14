"""
This version is intended to implement transposition tables.
@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import shuffle
from time import time
import math
import numpy as np

POSSIBLE_MOVES = [2 ** i for i in range(49)]


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
    player_pieces, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel
        player_pieces += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i, j] != 0]
            player_pieces += [b'0', b'1'][board[i, j] == player]

    return int(player_pieces, 2), int(mask, 2)


def generate_plays(mask):
    """Generate a list with all the possible plays in a given round.

    Parameters:
    mask (int): binary representation (bit board) of all pieces

    Returns:
    List : bit value of all available plays"""

    valid_plays = []
    for column_number in range(7):
        column_values = POSSIBLE_MOVES[7 * column_number: 7 * column_number + 6]  # Minus extra cell on top of the board
        for value in column_values:
            if mask & value == 0:
                valid_plays.append(value)
                break
    return valid_plays


def transform_play_to_column(play):
    """Return position of the column where the play was made.

    Parameters:
    play         (int): bit board representation of a piece

    Returns:
    int : column position"""

    return [2 ** i for i in range(49)].index(play) // 7


external_dict = {}


def initialize_node(bit_board, mask):
    if (bit_board, mask) in external_dict.keys():
        return external_dict[(bit_board, mask)]
    else:
        return 0, 0


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)
        self.children = []

        self.plays = generate_plays(self.mask)
        self.score, self.count = initialize_node(bit_board, mask)

        shuffle(self.plays)  # Inplace list shuffle

    def explore_or_exploit(self):
        if self.plays:
            return self.new_child()
        else:
            scores = [ucb1(child, self.count, exploration_parameter=math.sqrt(2)) for child in self.children]
            return self.children[scores.index(max(scores))]

    def new_child(self):
        play = self.plays.pop(0)

        new_bit_board = (self.mask ^ self.bit_board) | play
        new_mask = self.mask | play
        node = Node(bit_board=new_bit_board,
                    mask=new_mask,
                    play=play)
        self.children.append(node)
        return node


def manager(current_node, max_time):
    t0 = time()

    while time() - t0 < max_time:
        tree_search(current_node)

    scores = [child.score / child.count for child in current_node.children]

    save_data_into_dict(current_node)

    return current_node.children[scores.index(max(scores))].play


def save_data_into_dict(current_node):
    for child in current_node.children:
        external_dict[(child.bit_board, child.mask)] = (child.score, child.count)
        save_data_into_dict(child)


def tree_search(node):
    if node.value != 0 or node.mask == 279258638311359:  # Find terminal nodes
        node.score += int(node.value)
        node.count += 1
        return [-node.value, node.play]  # Giving higher score to shallow nodes

    child = node.explore_or_exploit()
    result = tree_search(node=child)

    node.score += int(result[0] == 1)
    node.count += 1

    return -result[0], child.play


def ucb1(child, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child.count
    return (child.score / child.count) + exploration_parameter * math.sqrt(e1)


def iebot_v6(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    global global_node
    global external_dict

    try:
        try:
            node_selection = [child.bit_board == (bit_board ^ mask) and child.mask == mask for child in
                              global_node.children]

            global_node = global_node.children[node_selection.index(True)]
        except:
            scores = [child.score / child.count for child in global_node.children]
            global_node = global_node.children[scores.index(max(scores))]

            node_selection = [child.bit_board == (bit_board ^ mask) and child.mask == mask for child in
                              global_node.children]
            global_node = global_node.children[node_selection.index(True)]
    except:
        global_node = Node(bit_board ^ mask, mask)

    play = manager(current_node=global_node, max_time=1)

    return transform_play_to_column(play=play)
