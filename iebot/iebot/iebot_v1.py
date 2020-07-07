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
    Tuple: (count (int)       : count of top_piece sequence,
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

    return count, free_spaces


def get_vertical_summary(board):
    """Analyses the vertical status of a board.

    Parameters:
    board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    List: [List: (count (int)       : count of top_piece sequence,
                  free_spaces (int) : number of free cell slots in that column)]"""

    vertical_summary = []

    for column_position in range(board.shape[1]):
        vertical_summary.append(get_column_summary(column=board[:, column_position]))

    return np.array(vertical_summary)


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

    horizontal_summary = [get_row_summary(row=board[i, :]) for i in range(board.shape[0])]

    return np.array(horizontal_summary)


def horizontal_play(horizontal_summary, vertical_summary):
    # TODO figure out how to deal with current ordering (non-ordering)
    number_free_spaces = vertical_summary[:, 1]
    end_pos = horizontal_summary[horizontal_summary[:, 0].argmax(), 1]

    # TODO fix start_pos routine
    start_pos = end_pos - horizontal_summary['counter'].max() + 1
    level = number_free_spaces[end_pos]
    if end_pos < 6:

        if level == number_free_spaces[end_pos + 1] - 1:
            if number_free_spaces[end_pos + 1]:
                return end_pos + 1

    if start_pos > 0:

        if level == number_free_spaces[start_pos - 1] - 1:
            if number_free_spaces[start_pos - 1]:
                return start_pos - 1

    return vertical_play(vertical_summary)


def vertical_play(vertical_summary):
    # TODO how to deal with the order and indexes
    counters = vertical_summary[:, 0]
    number_free_spaces = vertical_summary[:, 1]

    for i, count_value in counters:
        if count_value < 3:
            if number_free_spaces[3]:
                return 3

        if number_free_spaces[i]:
            if count_value + number_free_spaces[i] >= 4:
                return i

    for i, count_value in counters:
        if number_free_spaces[i]:
            return i


def play(board):
    vertical_summary = get_vertical_summary(board)
    horizontal_summary = get_horizontal_summary(board)

    if horizontal_summary[:, 0].max() > vertical_summary[:, 0].max():  # Is sequence count higher on the horizontal?
        return horizontal_play(horizontal_summary=horizontal_summary, vertical_summary=vertical_summary)

    return vertical_play(vertical_summary)


def iebot_v1(obs, config):
    board = translate_board(obs.board)

    play_position = play(board)

    if board.sum() == 0:
        return int(3)

    return int(play_position)
