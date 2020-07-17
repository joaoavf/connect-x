"""
This version is intended to implement transposition tables.
@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import shuffle
from time import time
import math
from utils import *


def manager(current_node, max_time):
    t0 = time()

    while time() - t0 < max_time:
        tree_search(current_node)

    scores = [child.score for child in current_node.children]

    return current_node.children[scores.index(max(scores))].play


def tree_search(node):
    if node.value != 0 or node.mask == 279258638311359:  # Find terminal nodes
        node.score += int(node.value)
        node.count += 1
        return -node.value  # Giving higher score to shallow nodes

    child = node.explore_or_exploit()
    result = tree_search(node=child)

    node.score += int(result == 1)
    node.count += 1

    return -result


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
        self.score, self.count = 0, 0

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


def my_agent(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    node = Node(bit_board ^ mask, mask)

    play = manager(current_node=node, max_time=1)

    return transform_play_to_column(play=play)
