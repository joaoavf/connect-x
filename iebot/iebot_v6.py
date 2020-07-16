"""
This version is intended to implement transposition tables.
@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from random import shuffle
from time import time
import math
from iebot.utils import *


class Manager:
    def __init__(self, bit_board, mask, max_time):
        self.node = Node(bit_board ^ mask, mask)
        self.max_time = max_time

    def run(self, bit_board, mask):
        t0 = time()
        while time() - t0 < self.max_time:
            tree_search(self.node)

        if self.node.bit_board != bit_board ^ mask:  # Single bot
            self.node = [child for child in self.node.children if child.bit_board == (bit_board ^ mask)][0]

        scores = [child.score for child in self.node.children]

        play = self.node.children[scores.index(max(scores))].play

        self.node = self.node.children[scores.index(max(scores))]

        return play


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


def iebot_v6(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    global manager

    if 'manager' not in globals() or (board <= 1).sum():
        manager = Manager(bit_board=bit_board, mask=mask, max_time=1)

    play = manager.run(bit_board, mask)

    return transform_play_to_column(play=play)
