import numpy as np

def get_top_pieces(board):
    top_pieces = []

    for c in range(board.shape[1]):

        col = board[:, c]

        top_piece, counter, is_space = 0, 0, 0

        for v in col:

            if v > 0:

                if top_piece == 0:
                    top_piece = v
                    counter += 1

                elif v == top_piece:
                    counter += 1

                else:
                    break

            else:
                is_space += 1

        top_pieces.append({'top_piece': top_piece,
                           'counter': counter,
                           'is_space': is_space})

    return top_pieces


def get_lines(board):
    max_list = []

    for c in range(board.shape[0]):

        row = board[c, :]

        last = -1

        max_last_counter = 0

        max_pos = 0

        for i, v in enumerate(row):

            if v == last and v > 0:
                last_counter += 1

            elif v > 0:
                last = v
                last_counter = 1

            else:
                last_counter = 0

            if last_counter > max_last_counter:
                max_last_counter = last_counter
                max_pos = i

        max_list.append([max_last_counter, max_pos])

    return max_list


def play_highest_column(board, opp_mark):
    top_pieces = get_top_pieces(board)
    #lines = pd.DataFrame(reversed(get_lines(board)), columns=['counter', 'end_pos'])
    lines = np.array(list(reversed(get_lines(board))))

    #max_lines = lines['counter'].max()
    max_lines = lines[:, 0].max()

    #counters = pd.Series([c['counter'] for c in top_pieces]).sort_values(ascending=False)
    counters = np.array([[i, c['counter']] for i, c in enumerate(top_pieces)])
    counters = counters[list(reversed(counters[:, 1].argsort()))]

    max_columns = counters.max()

    free_slots = [c['is_space'] for c in top_pieces]

    top = [c['top_piece'] for c in top_pieces]

    if max_lines > max_columns:
        #end_pos = lines.loc[lines[:, 1].idxmax()]['end_pos']
        end_pos = lines[lines[:, 0].argmax(), 1]
        start_pos = end_pos - lines['counter'].max() + 1
        level = free_slots[end_pos]
        if end_pos < 6:

            if level == free_slots[end_pos + 1] - 1:
                if free_slots[end_pos + 1]:
                    return int(end_pos + 1)

        if start_pos > 0:

            if level == free_slots[start_pos - 1] - 1:
                if free_slots[start_pos - 1]:
                    return int(start_pos - 1)

    for i, _ in counters:
        if free_slots[i]:
            if free_slots[i] < 4 and opp_mark == top[i]:
                continue
            else:
                return int(i)


def translate_board(board):
    return np.array(board).reshape(6, 7)


def my_agent(obs, config):
    board = translate_board(obs.board)

    opp_mark = 2 if obs.mark == 1 else 1

    play = play_highest_column(board, opp_mark)

    return int(play)
