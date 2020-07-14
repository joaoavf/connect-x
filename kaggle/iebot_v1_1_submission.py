"""
This code is a Connect 4 agent based on if and else, hence If-Else Bot (IEBOT) that searches for 3 connected  pieces and
play the 4th, either to block its opponent or to make its own Connect 4 Sequence.

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

import numpy as np


def translate_board(board):
    return np.array(board).reshape(6, 7)


def get_column_summary(column):
    """Analyses the status of a given vertical column.

    Parameters:
    column (List): Column of a Connect4 Game mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    Tuple(count (int)       : count of top_piece sequence,
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
    List[List[count (int)       : count of top_piece sequence,
              free_spaces (int) : number of free cell slots in that column]]"""

    vertical_summary = []

    for column_position in range(board.shape[1]):
        vertical_summary.append(get_column_summary(column=board[:, column_position]))

    return np.array(vertical_summary)


def get_row_summary(row):
    """Analyses the status of a given horizontal row.

    Parameters:
    row (List): Row of a Connect4 Game mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    List[max_count (int)              : max count achieved by horizontal sequence in this row,
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
    """Analyses the horizontal status of a board.

    Parameters:
    horizontal_summary (List[int, int]): Contains max_count and max_position by line
    vertical_summary   (List[int, int]): Contains count and free_spaces by line

    Returns:
    int : column number to be played"""

    number_free_spaces = vertical_summary[:, 1]
    end_position = horizontal_summary[horizontal_summary[:, 0].argmax(), 1]
    start_position = end_position - horizontal_summary[:, 0].max() + 1
    level = horizontal_summary[:, 0].argmax()

    possible_moves = [start_position - 1, end_position + 1]
    valid_moves = [position for position in possible_moves if position in list(range(7))]

    for position in valid_moves:
        if level == number_free_spaces[position] - 1:
            return position

    return vertical_play(vertical_summary)  # vertical_summary contains all other plays


def vertical_play(vertical_summary):
    """Analyses the vertical status of a board.

    Parameters:
    vertical_summary   (List[int, int]): Contains count and free_spaces by line

    Returns:
    int : column number to be played"""

    index = np.flipud(vertical_summary[:, 0].argsort())  # Sorts columns indexes by descending order of sequence count
    vertical_summary = vertical_summary[index]
    counters, number_free_spaces = vertical_summary[:, 0], vertical_summary[:, 1]

    for i, count_value in enumerate(counters):
        if number_free_spaces[i]:
            if count_value < 3 and number_free_spaces[index.tolist().index(3)]:  # If not Connect4, always play middle
                return 3
            elif count_value + number_free_spaces[i] >= 4:  # Continues or blocks valid sequence
                return index[i]

    for i, count_value in counters:
        if number_free_spaces[i]:
            return index[i]


def play(board):
    """Decides which will be the next play given the board status.

    Parameters:
    board (np.array): 6x7 board mapped by (0: Empty, 1: Player 1, 2: Player 2)

    Returns:
    int : column number to be played"""

    vertical_summary = get_vertical_summary(board)
    horizontal_summary = get_horizontal_summary(board)

    if horizontal_summary[:, 0].max() > vertical_summary[:, 0].max():  # Is sequence count higher on the horizontal?
        return horizontal_play(horizontal_summary=horizontal_summary, vertical_summary=vertical_summary)

    return vertical_play(vertical_summary)


def my_agent(obs, config):
    """Transform received data into the necessary data types and calls the next play.

    Parameters:
    obs (?)    : All observed data from the game is contained here
    config (?) : Mandatory field from Kaggle that contains setting for the game
    Returns:
    int : column number to be played"""

    board = translate_board(obs.board)

    play_position = play(board)

    return int(play_position)
