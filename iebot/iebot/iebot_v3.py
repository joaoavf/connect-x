# Inspired by https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
# Author: joaoavf - joao.avf@gmail.com

"""
def run_a(a):
    a = np.array(a)

    b = a == 1
    b = b.dot(1 << np.arange(b.size)[::-1])

    c = a > 0
    c = c.dot(1 << np.arange(c.size)[::-1])

    np.flipud(np.array(range(42)).reshape(7, 6).transpose())

    return b, c"""

import numpy as np


class Node:
    def __init__(self, bit_board, mask, position_map, column_map, player_mark, recursiveness, is_origin):
        self.bit_board = bit_board
        self.mask = mask
        self.position_map = position_map
        self.column_map = column_map
        self.player_mark = player_mark
        self.opp_mark = 2 if player_mark == 1 else 1
        self.is_tree = False
        self.children = []
        self.recursiveness = recursiveness
        self.create_trees()
        self.is_origin = is_origin
        self.value = 0

    def create_trees(self):
        for column in columns:
            play = can_play(self.mask, column)
            if play:
                new_bit_board = self.bit_board | play
                new_mask = self.mask | play
                self.children.append(Tree(bit_board=new_bit_board,
                                          mask=new_mask,
                                          play=play,
                                          position_map=self.position_map,
                                          column_map=self.column_map,
                                          player_mark=self.player_mark,
                                          recursiveness=self.recursiveness))


class Tree:
    def __init__(self, bit_board, mask, play, position_map, column_map, player_mark, recursiveness):
        self.node = []
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.position_map = position_map
        self.column_map = column_map
        self.player_mark = player_mark
        self.opp_mark = 2 if player_mark == 1 else 1
        self.connect4 = connected_four(bit_board)
        self.is_tree = True
        self.recursiveness = recursiveness

        self.create_node()

    def create_node(self):
        if self.recursiveness > 1 and not self.connect4:
            self.node = Node(bit_board=self.bit_board ^ self.mask,
                             mask=self.mask,
                             position_map=self.position_map,
                             column_map=self.column_map,
                             player_mark=self.opp_mark,
                             recursiveness=self.recursiveness - 1,
                             is_origin=False)


pos_map = [2 ** i for i in range(49)]

columns = []
for col_number in range(7):
    columns.append(pos_map[7 * col_number: 7 * col_number + 6])


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def column_routine(bit_board, mask):
    result = []

    for column in columns:
        play = can_play(mask, column)
        if play:
            new_bit_board = bit_board | play
            new_mask = mask | play
            result.append({'bit_board': new_bit_board,
                           'mask': new_mask,
                           'connect_4': connected_four(new_bit_board)})
    return result


def translate_board(board):
    return np.array(board).reshape(6, 7)


def get_position_mask_bitmap(board, player):
    position, mask = b'', b''
    # Start with right-most column
    for j in range(6, -1, -1):
        # Add 0-bits to sentinel
        mask += b'0'
        position += b'0'
        # Start with bottom row
        for i in range(0, 6):
            mask += [b'0', b'1'][board[i, j] != 0]
            position += [b'0', b'1'][board[i, j] == player]
    return int(position, 2), int(mask, 2)


def connected_four(position):
    # Horizontal check
    m = position & (position >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False


def iebot_v3(obs, config):
    board = translate_board(obs.board)

    opp_mark = 2 if obs.mark == 1 else 1

    play = play_highest_column(board, opp_mark)

    if board.sum() == 0:
        return int(3)

    return int(play)
