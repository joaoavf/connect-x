from iebot import iebot
import numpy as np


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def connected_four(position):
    # Horizontal check
    m = position & (position >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False


def translate_board(board):
    return np.array(board).reshape(6, 7)


def get_position_mask_bitmap(board, player):
    position, mask = b'', b''
    # Start with right-most column
    for j in range(6, -1, -1):
        # Add 0-bits to sentinel
        mask += b'0'
        position += b'0'
        # Start with bottom row
        for i in range(0, 6):
            mask += [b'0', b'1'][board[i, j] != 0]
            position += [b'0', b'1'][board[i, j] == player]
    return int(position, 2), int(mask, 2)


pos_map = [2 ** i for i in range(49)]

columns = []
for col_number in range(7):
    columns.append(pos_map[7 * col_number: 7 * col_number + 6])


class Node:
    def __init__(self, bit_board, mask, play=None, recursiveness=5):
        self.bit_board = bit_board
        self.mask = mask
        self.play = play
        self.children = []
        self.recursiveness = recursiveness
        self.value = self.run()

    def generate_plays(self):
        return [can_play(self.mask, column) for column in columns]

    def run(self):
        plays = self.generate_plays()
        for play in plays:
            if play:
                new_bit_board = self.bit_board | play
                new_mask = self.mask | play
                if connected_four(new_bit_board):
                    return [1, 0, play]

        for play in plays:
            if play:
                new_bit_board = self.bit_board | play
                new_mask = self.mask | play
                if self.recursiveness > 0:
                    self.children.append(Node(bit_board=new_bit_board ^ new_mask,
                                              mask=new_mask,
                                              recursiveness=self.recursiveness - 1,
                                              play=play))

        if not self.children:
            return [0, 0, -1]

        local_max = -100
        local_min = 100
        play = -1

        for child in self.children:
            if local_max < -child.value[0]:
                local_max = -child.value[0]
                play = child.play

            if local_min > -child.value[0]:
                local_min = -child.value[0]

        return [local_max, local_min, play]


def column_from_play(play):
    for i, column in enumerate(columns):
        if play in column:
            return i


def iebot_v3(obs, config):
    bit_board, mask = get_position_mask_bitmap(translate_board(obs.board), obs.mark)

    node = Node(bit_board, mask)

    if node.value[0] > 0 or node.value[1] < 0:
        play = column_from_play(node.value[2])
    else:
        play = iebot(obs, config)

    return int(play)
