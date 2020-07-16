"""
This is a MiniMax Connect 4 agent.

Inspired by:
https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
http://blog.gamesolver.org/

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from iebot.utils import *


class Node:
    def __init__(self, bit_board, mask, columns_map, play=None, recursiveness=5):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.columns_map = columns_map
        self.children = []
        self.recursiveness = recursiveness

        self.value = self.run()

    def generate_plays(self):
        plays = [can_play(self.mask, column) for column in self.columns_map]
        return [play for play in plays if play]  # Remove cases of play = 0

    def create_children(self, plays):
        for play in plays:
            if self.recursiveness > 0:
                new_bit_board = self.bit_board | play
                new_mask = self.mask | play
                self.children.append(Node(bit_board=new_bit_board ^ new_mask,
                                          mask=new_mask,
                                          columns_map=self.columns_map,
                                          recursiveness=self.recursiveness - 1,
                                          play=play))

    def evaluate_node(self, plays):
        for play in plays:  # This is where the alpha-beta pruning happens, as it prevents the creation of new children

            new_bit_board = self.bit_board | play
            if connected_four(new_bit_board):
                return [1, play]  # No need for children if there is a connect 4 available
        else:
            self.create_children(plays)
            return [0, -1]  # Result set in case there is no children and we need to propagate this

    def negamax(self):
        local_max, play = -100, -1

        for child in self.children:
            if -child.value[0] > local_max:
                local_max = -child.value[0]
                play = child.play

        return [local_max, play]

    def run(self):
        """First it checks all the possible moves in the game, then it creates the children of nodes that will ever be
        possibly played.

        Returns:
        List[int, int] : [is_game_won   : 1=won | 0=tie | -1=loss,
                          play_position : position in binary board representation]"""

        plays = self.generate_plays()  # Calculate all possible moves for the player in this board position
        is_game_won, play_position = self.evaluate_node(plays)  # Has children if node is not final

        if is_game_won or not self.children:  # If is game won or if last node due to recursiveness limit
            return [is_game_won, play_position]
        else:
            return self.negamax()  # NegaMax on children output


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def calculate_recursiveness(board):
    pieces = (board > 0).sum()

    if pieces < 12:
        return 5
    elif pieces < 18:
        return 6
    elif pieces < 22:
        return 7
    elif pieces < 26:
        return 8
    else:
        return 9


def iebot_v3(obs, config):
    board = translate_board(obs.board)
    bit_board, mask = get_position_mask_bitmap(board, obs.mark)
    columns_map = generate_columns_map(mask)
    recursiveness = calculate_recursiveness(board)

    node = Node(bit_board, mask, recursiveness=recursiveness, columns_map=columns_map)
    play = node.value[1]

    # TODO implement heuristics
    return transform_play_to_column(play, columns_map)
