# Inspired by https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
# Author: joaoavf - joao.avf@gmail.com

"""
def run_a(a):
    a = np.array(a)

    b = a == 1
    b = b.dot(1 << np.arange(b.size)[::-1])

    c = a > 0
    c = c.dot(1 << np.arange(c.size)[::-1])

    np.flipud(np.array(range(42)).reshape(7, 6).transpose())

    return b, c"""

import numpy as np

pos_map = [2 ** i for i in range(49)]

columns = []
for col_number in range(7):
    columns.append(pos_map[7 * col_number: 7 * col_number + 6])


def can_play(bit_board, column):
    for element in column:
        if element & bit_board == 0:
            return element
    return False


def column_routine(bit_board, mask):
    result = []

    for column in columns:
        play = can_play(mask, column)
        if play:
            new_bit_board = bit_board | play
            new_mask = mask | play
            result.append({'bit_board': new_bit_board,
                           'mask': new_mask,
                           'connect_4': connected_four(new_bit_board)})
    return result


def leaf(bit_board, mask, recursiveness=1):
    results = {'bit_board': bit_board, 'mask': mask}

    results['tree'] = column_routine(bit_board=bit_board, mask=mask)

    if recursiveness > 1:
        if any([result['connect_4'] for result in results['tree']]):
            return 'aaaa'
        for result in results['tree']:
            if not result['connect_4']:
                if recursiveness > 1:
                    result['tree'] = leaf(bit_board ^ mask, mask, recursiveness - 1)

    return results


def search_tree(tree, start):
    for result in tree['tree']:
        if result['connect_4']:
            return start + [result['bit_board'] ^ tree['bit_board']]
        else:
            try:
                return search_tree(result['tree'], [result['bit_board'] ^ tree['bit_board']])
            except:
                return start + [result['bit_board'] ^ tree['bit_board']]


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


def iebot_v3(obs, config):
    board = translate_board(obs.board)

    opp_mark = 2 if obs.mark == 1 else 1

    play = play_highest_column(board, opp_mark)

    if board.sum() == 0:
        return int(3)

    return int(play)
