from iebot.utils import *


def negamax_ab(node, depth, columns_map, alpha=-float('inf'), beta=float('inf')):
    if depth == 0 or node.value != 0:
        return [node.value, node.play]  # TODO add heuristic value of the node

    value, play = -float('inf'), -1
    for child in node.create_children(mask=node.mask, columns_map=columns_map):

        result = negamax_ab(node=child, depth=depth - 1, columns_map=columns_map, alpha=-beta, beta=-alpha)
        # TODO rewrite this to make more elegant the use of a list here
        result[0] = -result[0]
        value = max(value, result[0])

        play = node.play
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return [value, play]


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)

    def create_children(self, mask, columns_map):
        plays = generate_plays(mask, columns_map)
        for play in plays:
            new_bit_board = self.bit_board | play
            new_mask = self.mask | play
            yield Node(bit_board=new_bit_board ^ new_mask,
                       mask=new_mask,
                       play=play)


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def generate_plays(mask, columns_map):
    plays = [can_play(mask, column) for column in columns_map]
    return [play for play in plays if play]  # Remove cases of play = 0


def iebot_v4(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)
    columns_map = generate_columns_map(mask)

    node = Node(bit_board, mask)
    # TODO rewrite this to make more elegant the use of a list here
    _, play = negamax_ab(node=node, depth=5, columns_map=columns_map)

    return transform_play_to_column(play=play, columns_map=columns_map)
