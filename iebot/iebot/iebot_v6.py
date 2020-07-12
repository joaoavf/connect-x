"""
This version is intended to implement transposition tables.

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import choice
from time import time
import pandas as pd
from iebot.utils import *
from iebot.tranposition_table_8_ply import tranposition_table


def manager(current_node, max_time):
    t0 = time()
    results = []
    while time() - t0 < max_time:
        results.append(tree_search(current_node))
    df = pd.DataFrame(results, columns=['value', 'play'])
    return df.groupby('play').mean().idxmax()[0]


def tree_search(node):
    if node.value != 0 or node.mask == 279258638311359:
        return [-node.value, node.play]  # Giving higher score to shallow nodes
    elif node.play != 0 and (node.bit_board, node.mask) in tranposition_table:
        return [-tranposition_table[(node.bit_board, node.mask)], node.play]

    child = node.random_child()
    result = tree_search(node=child)

    return -result[0], child.play


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)

    def random_child(self):
        plays = generate_plays(self.mask)
        play = choice(plays)

        new_bit_board = (self.mask ^ self.bit_board) | play
        new_mask = self.mask | play
        node = Node(bit_board=new_bit_board,
                    mask=new_mask,
                    play=play)
        return node


def iebot_v6(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    node = Node(bit_board ^ mask, mask)

    play = manager(current_node=node, max_time=1)

    return transform_play_to_column(play=play)
