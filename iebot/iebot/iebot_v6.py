"""
This version is intended to implement transposition tables.
@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import shuffle
from time import time
import math
import pickle

global external_dict

with open('external_dict.pickle', 'rb') as handle:
    external_dict = pickle.load(handle)

from iebot.utils import *


def manager(current_node, max_time):
    t0 = time()
    results = []
    while time() - t0 < max_time:
        tree_search(current_node)

    scores = [child.score for child in current_node.children]

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
    # elif node.play != 0 and (node.bit_board, node.mask) in tranposition_table:
    # return [-tranposition_table[(node.bit_board, node.mask)], node.play]

    child = node.explore_or_exploit()
    result = tree_search(node=child)

    node.score += int(result[0] == 1)
    node.count += 1

    return -result[0], child.play


def ucb1(child, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child.count
    return (child.score / child.count) + exploration_parameter * math.sqrt(e1)


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

    def best_node(self):
        scores = [child.score for child in self.children]
        return self.children[scores.index(max(scores))]

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


def iebot_v6(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    node = Node(bit_board ^ mask, mask)

    play = manager(current_node=node, max_time=3)

    return transform_play_to_column(play=play)
