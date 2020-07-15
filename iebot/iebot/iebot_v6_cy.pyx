from random import shuffle
from time import time
import math
import numpy as np
from iebot.utils import *
import cython


from iebot.utils_cy import generate_plays

def manager(current_node, max_time, external_dict):
    t0 = time()
    while time() - t0 < max_time:
        tree_search(current_node, external_dict)

    scores = [child.count for child in current_node.children]

    save_data_into_dict(current_node, external_dict)

    return current_node.children[scores.index(max(scores))].play

def save_data_into_dict(current_node, external_dict):
    for child in current_node.children:
        external_dict[(child.bit_board, child.mask)] = (child.score, child.count)
        save_data_into_dict(child, external_dict)

def tree_search(node, external_dict):
    if node.value != 0 or node.mask == 279258638311359:  # Find terminal nodes
        node.score += int(node.value)
        node.count += 1
        return [-node.value, node.play]  # Giving higher score to shallow nodes

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
        self.value = connected_four(self.bit_board)
        self.children = []

        self.plays = generate_plays(self.mask)
        self.score, self.count = initialize_node(bit_board, mask, external_dict)

        shuffle(self.plays)  # Inplace list shuffle

    def explore_or_exploit(self, external_dict):
        if self.plays:
            return self.new_child(external_dict)
        else:
            scores = [ucb1(child, self.count, exploration_parameter=math.sqrt(2)) for child in self.children]
            return self.children[scores.index(max(scores))]

    def new_child(self, external_dict):
        play = self.plays.pop(0)

        new_bit_board = (self.mask ^ self.bit_board) | play
        new_mask = self.mask | play
        node = Node(bit_board=new_bit_board,
                    mask=new_mask,
                    play=play,
                    external_dict=external_dict)
        self.children.append(node)
        return node
