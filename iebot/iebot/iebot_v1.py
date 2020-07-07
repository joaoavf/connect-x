"""
This code is a Connect 4 agent based on simple procedural technology that searches for 3 connected pieces and play the
4th, either to block its opponent or to make its own Connect 4 Sequence.

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

import numpy as np
from iebot.utils import translate_board


def get_column_summary(column):
    """Analyses the status of a given vertical column.

    Parameters:
    column (List): Column of a Connect4 Game mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    Tuple: (top_piece (int)   : top_piece identifier to player 1 or 2,
            count (int)       : count of top_piece sequence,
            free_spaces (int) : number of free cell slots in that column)"""

    count, top_piece, free_spaces = 0, 0, 0

    for cell_value in column:
        if cell_value > 0:  # Cell is not empty
            if count == 0:
                top_piece = cell_value
            if cell_value == top_piece:
                count += 1
            else:
                break
        else:
            free_spaces += 1
    # TODO remove top_piece, it is irrelevant in the code logic
    return top_piece, count, free_spaces


def get_vertical_summary(board):
    """Analyses the vertical status of a board.

    Parameters:
    board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    List: [Tuple: (top_piece (int)   : top_piece identifier to player 1 or 2,
                   count (int)       : count of top_piece sequence,
                   free_spaces (int) : number of free cell slots in that column)]"""

    vertical_summary = []

    for column_position in range(board.shape[1]):
        top_piece, counter, num_spaces = get_column_summary(column=board[:, column_position])
        vertical_summary.append({'top_piece': top_piece, 'counter': counter, 'num_spaces': num_spaces})

    return vertical_summary


def get_row_summary(row):
    """Analyses the status of a given horizontal row.

    Parameters:
    row (List): Row of a Connect4 Game mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    List: [max_count (int)              : max count achieved by horizontal sequence in this row,
           max_end_position (int)       : end position for the max count find]"""

    previous_cell_value, max_count, max_end_position, current_count = 0, 0, 0, 0

    for cell_position, cell_value in enumerate(row):
        if cell_value <= 0:
            current_count = 0
        elif cell_value == previous_cell_value:
            current_count += 1
        else:
            previous_cell_value = cell_value
            current_count = 1

        if current_count > max_count:
            max_count, max_end_position = current_count, cell_position

    return [max_count, max_end_position]


def get_horizontal_summary(board):
    """Analyses the horizontal status of a board.

    Parameters:
     board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    List[List[max_count (int)              : max count achieved by horizontal sequence in this row,
              max_end_position (int)       : end position for the max count find]]"""

    return [get_row_summary(row=board[i, :]) for i in range(board.shape[0])]


def play_highest_column(board, opp_mark):
    top_pieces = get_vertical_summary(board)
    # lines = pd.DataFrame(reversed(get_lines(board)), columns=['counter', 'end_pos'])
    lines = np.array(list(reversed(get_horizontal_summary(board))))

    # max_lines = lines['counter'].max()
    max_lines = lines[:, 0].max()

    # counters = pd.Series([c['counter'] for c in top_pieces]).sort_values(ascending=False)
    counters = np.array([[i, c['counter']] for i, c in enumerate(top_pieces)])
    counters = counters[list(reversed(counters[:, 1].argsort()))]


def horizontal_play(horizontal_summary, num_spaces):
    end_pos = horizontal_summary[horizontal_summary[:, 0].argmax(), 1]
    start_pos = end_pos - horizontal_summary['counter'].max() + 1
    level = num_spaces[end_pos]
    if end_pos < 6:

        if level == num_spaces[end_pos + 1] - 1:
            if num_spaces[end_pos + 1]:
                return True, int(end_pos + 1)

    if start_pos > 0:

        if level == num_spaces[start_pos - 1] - 1:
            if num_spaces[start_pos - 1]:
                return True, int(start_pos - 1)

    return False, -1


def vertical_play(counters, num_spaces):
    for i, count_value in counters:
        if count_value < 3:
            if num_spaces[3]:
                return 3

        if num_spaces[i]:
            if count_value + num_spaces[i] >= 4:
                return i

    for i, count_value in counters:
        if num_spaces[i]:
            return i


def play(board):
    vertical_summary = get_vertical_summary(board)

    horizontal_summary = np.array(list(reversed(get_horizontal_summary(board))))

    # TODO make sure this is working, I have tossed out reversed()
    max_horizontal_counter = horizontal_summary[:, 0].max()

    counters = np.array([[i, c['counter']] for i, c in enumerate(vertical_summary)])
    counters = counters[list(reversed(counters[:, 1].argsort()))]

    max_vertical_counter = counters.max()

    num_spaces = [c['num_spaces'] for c in vertical_summary]

    if max_horizontal_counter > max_vertical_counter:
        is_play, play_position = horizontal_play(horizontal_summary=horizontal_summary, num_spaces=num_spaces)
        if is_play:
            return play_position

    return vertical_play(counters=counters, num_spaces=num_spaces)


def iebot_v1(obs, config):
    board = translate_board(obs.board)

    play_position = play(board)

    if board.sum() == 0:
        return int(3)

    return int(play_position)
