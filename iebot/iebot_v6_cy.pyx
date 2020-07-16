from time import time
import math
import numpy as np
from random import shuffle

cpdef translate_board(board):
    return np.array(board).reshape(6, 7).tolist()

cpdef bint connected_four(unsigned long long bit_board):
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
    player_pieces, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel
        player_pieces += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i][j] != 0]
            player_pieces += [b'0', b'1'][board[i][j] == player]

    return int(player_pieces, 2), int(mask, 2)

cpdef list generate_plays(unsigned long long mask):
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

cpdef int transform_play_to_column(unsigned long long play):
    cdef unsigned long long i
    POSSIBLE_MOVES = [2 ** i for i in range(49)]

    return POSSIBLE_MOVES.index(play) // 7

cpdef unsigned long long manager(current_node, double max_time, dict external_dict):
    cdef double t0
    cdef list scores

    t0 = time()
    while time() - t0 < max_time:
        tree_search(current_node, external_dict)

    scores = [child.score for child in current_node.children]

    save_data_into_dict(current_node, external_dict)

    return current_node.children[scores.index(max(scores))].play

cdef void save_data_into_dict(current_node, dict external_dict):
    for child in current_node.children:
        external_dict[(child.bit_board, child.mask)] = (child.score, child.count)
        save_data_into_dict(child, external_dict)

cdef tuple tree_search(node, dict external_dict):
    cdef tuple result

    if node.value != 0 or node.mask == 279258638311359:  # Find terminal nodes
        node.score += node.value
        node.count += 1
        return -node.value, node.play

    child = node.explore_or_exploit(external_dict)
    result = tree_search(child, external_dict)

    node.score += int(result[0] == 1)
    node.count += 1

    return -result[0], child.play

def ucb1(child, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child.count
    return (child.score / child.count) + exploration_parameter * math.sqrt(e1)

def initialize_node(bit_board, mask, external_dict):
    if (bit_board, mask) in external_dict.keys():
        return external_dict[(bit_board, mask)]
    else:
        return 0, 0


class Node:
    def __init__(self, bit_board, mask, external_dict, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = int(connected_four(self.bit_board))
        self.children = []

        self.plays = generate_plays(self.mask)
        self.score, self.count = initialize_node(bit_board, mask, external_dict)

        shuffle(self.plays)

    def explore_or_exploit(self, external_dict):
        if self.plays:
            return self.new_child(external_dict)
        else:
            scores = [ucb1(child, self.count, exploration_parameter=math.sqrt(2)) for child in self.children]
            return self.children[scores.index(max(scores))]

    def new_child(self, external_dict):
        play = self.plays.pop(0)

        new_bit_board = self.mask - self.bit_board + play
        new_mask = self.mask + play
        node = Node(bit_board=new_bit_board,
                    mask=new_mask,
                    external_dict=external_dict,
                    play=play)

        self.children.append(node)
        return node
