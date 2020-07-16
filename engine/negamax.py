"""
This version was refactored and implemented alpha beta pruning, which increase nodes considered by the bot from 5 to 7.

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""
from engine.utils import *


def negamax_ab(node, max_depth, alpha=-float('inf'), beta=float('inf')):
    if max_depth == 0 or node.value != 0:
        return [-node.value - (0.01 * max_depth), node.play]  # Giving higher score to shallow nodes

    max_value, play = -float('inf'), -1

    for child in node.create_children():

        result = negamax_ab(node=child, max_depth=max_depth - 1, alpha=-beta, beta=-alpha)

        if -result[0] > max_value:
            max_value = -result[0]
            play = child.play
            alpha = max(alpha, max_value)

        if alpha >= beta:
            break

    if play == -1:  # Happens only when there are no more pieces left and game is tied
        return [0, play]
    else:
        return [max_value, play]


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)

    def create_children(self):
        plays = generate_plays(self.mask)
        plays = [plays.pop(i // 2) for i in reversed(range(len(plays)))]  # Order by the center

        for play in plays:
            new_bit_board = (self.mask ^ self.bit_board) | play
            new_mask = self.mask | play
            node = Node(bit_board=new_bit_board,
                        mask=new_mask,
                        play=play)
            yield node


def iebot_v4(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)

    node = Node(bit_board ^ mask, mask)

    _, play = negamax_ab(node=node, max_depth=8)
    return transform_play_to_column(play=play)
