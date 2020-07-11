from iebot.utils import *


def negamax_ab(node, depth, columns_map, alpha=-float('inf'), beta=float('inf'), on=True):
    if depth == 0 or node.value != 0:
        return [-node.value * (-0.01 * depth), node.play]  # Giving higher score to higher depth (less shallow)
    # TODO rewrite code so more depth != more shallow

    max_value, play = -float('inf'), -1

    for child in node.create_children(columns_map=columns_map):

        result = negamax_ab(node=child, depth=depth - 1, columns_map=columns_map, alpha=-beta, beta=-alpha)

        if -result[0] > max_value:
            max_value = -result[0]
            play = child.play
            alpha = max(alpha, max_value)

        if alpha >= beta:
            break

    if play == -1:  # If ran out of pieces in the board and game is tied
        return [0, play]
    else:
        return [max_value, play]


class Node:
    def __init__(self, bit_board, mask, play=0):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.value = connected_four(self.bit_board)

    def create_children(self, columns_map):
        plays = generate_plays(self.mask, columns_map)
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
    columns_map = generate_columns_map(mask)

    node = Node(bit_board ^ mask, mask)

    _, play = negamax_ab(node=node, depth=8, columns_map=columns_map)
    return transform_play_to_column(play=play, columns_map=columns_map)
