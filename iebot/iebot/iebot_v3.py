"""
This is a MiniMax Connect 4 agent.

Inspired by:
https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
http://blog.gamesolver.org/

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

from iebot.iebot_v2 import iebot_v2
from iebot.utils import translate_board


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
        return [can_play(self.mask, column) for column in self.columns_map]

    def create_children(self, plays, bit_board_map):
        for play in plays:
            if play and self.recursiveness > 0:
                new_mask = self.mask | play
                self.children.append(Node(bit_board=bit_board_map[play] ^ new_mask,
                                          mask=new_mask,
                                          columns_map=self.columns_map,
                                          recursiveness=self.recursiveness - 1,
                                          play=play))

    def evaluate_current_node(self, plays):
        bit_board_map = {}
        for play in plays:
            if play:
                bit_board_map[play] = self.bit_board | play
                if connected_four(bit_board_map[play]):
                    return [1, 0, play]  # No need for children if there is a connect 4 available
        else:
            self.create_children(plays, bit_board_map)
            return [0, 0, -1]  # Result set in case there is no children and we need to propagate this

    def evaluate_children_results(self):
        local_max, local_min, play = -100, 100, -1

        for child in self.children:
            if -child.value[0] > local_max:
                local_max = -child.value[0]
                play = child.play

            if local_min > -child.value[0]:
                local_min = -child.value[0]

        # TODO evaluate if local_min is necessary after we finish implementation (It currently depends on IEBot V2)
        return [local_max, local_min, play]

    def run(self):
        plays = self.generate_plays()
        result = self.evaluate_current_node(plays)  # Has children if node is not final and within recursiveness limit

        if result[0] or not self.children:  # If connect 4 or if last node to be evaluated due to recursiveness limit
            return result

        return self.evaluate_children_results()  # This is where it runs the MiniMax part of the algorithm


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def connected_four(bit_board):
    """Evaluates if player bit board has made a connect 4.

   Parameters:
   bit_board (int): bit board representation of player pieces the game

   Returns:
   bool : True if the board has achieved a connect 4"""

    # Horizontal check
    m = bit_board & (bit_board >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = bit_board & (bit_board >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = bit_board & (bit_board >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = bit_board & (bit_board >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False


def get_position_mask_bitmap(board, player):
    """Transform a 6x7 board representation into bit boards.

       Parameters:
       board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

       Returns:
       (int, int) : (bit board of player pieces, bit board of all pieces)"""

    # TODO fix deprecation warnings
    position, mask = b'', b''

    for j in range(6, -1, -1):  # Start with right-most column
        mask += b'0'  # Add 0-bits to sentinel
        position += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i, j] != 0]
            position += [b'0', b'1'][board[i, j] == player]

    return int(position, 2), int(mask, 2)


def generate_columns_map(mask):
    position_map = [2 ** i for i in range(49)]

    columns_map = []
    for column_number in range(7):
        column_values = position_map[7 * column_number: 7 * column_number + 6]
        column_values = [cell_value for cell_value in column_values if mask & cell_value == 0]
        columns_map.append(column_values)

    return columns_map


def column_from_play(play, columns_map):
    for i, column in enumerate(columns_map):
        if play in column:
            return i


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

    if node.value[0] < 0:
        return int(iebot_v2(obs, config))

    if node.value[0] > 0 or node.value[1] < 0:
        play = column_from_play(node.value[2], columns_map)
    else:
        play = iebot_v2(obs, config)

    return int(play)
