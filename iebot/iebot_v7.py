"""
Monte carlo forests

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import shuffle
from time import time
import pandas as pd
from iebot.utils import *
import math
from copy import deepcopy


def manager(node, max_time, trees=10):
    results = []
    for i in range(trees):
        results.extend(tree(deepcopy(node), max_time=max_time / trees))
    df = pd.DataFrame(results, columns=['value', 'play'])
    return df.groupby('play').mean().idxmax()[0]


def tree(node, max_time):
    t0 = time()
    results = []
    while time() - t0 < max_time:
        results.append(tree_search(node))
    return results


def tree_search(node):
    if node.value != 0 or node.mask == 279258638311359:  # Find terminal nodes
        node.score += node.value
        node.count += 1
        return [-node.value, node.play]  # Giving higher score to shallow nodes
    # elif node.play != 0 and (node.bit_board, node.mask) in tranposition_table:
    # return [-tranposition_table[(node.bit_board, node.mask)], node.play]

    child = node.explore_or_exploit()
    result = tree_search(node=child)

    node.score += result[0] == 1
    node.count += 1

    return -result[0], child.play


def ucb1(child, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child.count
    return (child.score / child.count) + exploration_parameter * math.sqrt(e1)


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)
        self.children = []
        self.plays = generate_plays(self.mask)
        self.score = 0
        self.count = 0
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


def iebot_v7(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    node = Node(bit_board ^ mask, mask)

    play = manager(node=node, max_time=1)

    return transform_play_to_column(play=play)
