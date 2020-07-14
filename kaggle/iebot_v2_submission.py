import numpy as np


def translate_board(board):
    return np.array(board).reshape(6, 7)


def analyze_column(column):
    top_piece, counter, num_spaces = 0, 0, 0

    for value in column:

        if value > 0:

            if top_piece == 0:
                top_piece = value
                counter += 1

            elif value == top_piece:
                counter += 1

            else:
                break
        else:
            num_spaces += 1

    return top_piece, counter, num_spaces


def contextualize_vertical(board):
    top_pieces = []

    for c in range(board.shape[1]):
        top_piece, counter, num_spaces = analyze_column(column=board[:, c])

        top_pieces.append({'top_piece': top_piece,
                           'counter': counter,
                           'num_spaces': num_spaces})

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
    top_pieces = contextualize_vertical(board)
    # lines = pd.DataFrame(reversed(get_lines(board)), columns=['counter', 'end_pos'])
    lines = np.array(list(reversed(get_lines(board))))

    # max_lines = lines['counter'].max()
    max_lines = lines[:, 0].max()

    # counters = pd.Series([c['counter'] for c in top_pieces]).sort_values(ascending=False)
    counters = np.array([[i, c['counter']] for i, c in enumerate(top_pieces)])
    counters = counters[list(reversed(counters[:, 1].argsort()))]

    max_columns = counters.max()

    num_spaces = [c['num_spaces'] for c in top_pieces]

    top = [c['top_piece'] for c in top_pieces]

    if max_lines > max_columns:
        # end_pos = lines.loc[lines[:, 1].idxmax()]['end_pos']
        end_pos = lines[lines[:, 0].argmax(), 1]
        start_pos = end_pos - lines['counter'].max() + 1
        level = num_spaces[end_pos]
        if end_pos < 6:

            if level == num_spaces[end_pos + 1] - 1:
                if num_spaces[end_pos + 1]:
                    return int(end_pos + 1)

        if start_pos > 0:

            if level == num_spaces[start_pos - 1] - 1:
                if num_spaces[start_pos - 1]:
                    return int(start_pos - 1)

    for i, count_value in counters:
        if count_value < 3:
            if num_spaces[3]:
                return int(3)

        if num_spaces[i]:
            if count_value + num_spaces[i] >= 4:  # and opp_mark == top[i]:
                return int(i)

    for i, count_value in counters:
        if num_spaces[i]:
            return int(i)


def iebot(obs, config):
    board = translate_board(obs.board)

    opp_mark = 2 if obs.mark == 1 else 1

    play = play_highest_column(board, opp_mark)

    if board.sum() == 0:
        return int(3)

    return int(play)


def process_interval(values, ids, player_mark, opp_mark):
    output = []
    for i in range(len(values) - 3):
        interval = values[i: i + 4]
        player_points = (interval == player_mark).sum()
        opp_points = (interval == opp_mark).sum()
        free_spaces = (interval == 0)

        is_left_1 = player_points == 3 or opp_points == 3

        if is_left_1 and free_spaces.sum():
            summary = {'free_space_id': ids[free_spaces.argmax() + i],
                       'is_my_advantage': player_points == 3}
            output.append(summary)
    return output


class Board:
    def __init__(self, board, player_mark):
        self.values = np.array(board).reshape(6, 7)
        self.ids = np.array(range(42)).reshape(6, 7)
        self.player_mark = player_mark
        self.opp_mark = 2 if player_mark == 1 else 1

    def play(self):
        results = self.process_all()
        for item in results:
            if item['is_my_advantage']:
                if item['free_space_id'] > 34:
                    return item['free_space_id'] % 7
                elif self.values.flatten()[item['free_space_id'] + 7] > 0:
                    return item['free_space_id'] % 7

        for item in results:
            if not item['is_my_advantage']:
                if item['free_space_id'] > 34:
                    return item['free_space_id'] % 7
                elif self.values.flatten()[item['free_space_id'] + 7] > 0:
                    return item['free_space_id'] % 7

    def process_all(self):
        results = []
        results.extend(self.process_diagonals())
        results.extend(self.process_columns())
        results.extend(self.process_rows())
        return results

    def process_diagonals(self):
        results = []
        for offset in range(-2, 4):
            left_to_right_diagonal = Diagonal(self.values, self.ids, left_to_right=True, offset=offset)
            results.extend(process_interval(left_to_right_diagonal.values, left_to_right_diagonal.ids,
                                            player_mark=self.player_mark, opp_mark=self.opp_mark))
            right_to_left_diagonal = Diagonal(self.values, self.ids, left_to_right=False, offset=offset)
            results.extend(process_interval(right_to_left_diagonal.values, right_to_left_diagonal.ids,
                                            player_mark=self.player_mark, opp_mark=self.opp_mark))
        return results

    def process_columns(self):
        results = []
        for i in range(self.values.shape[1]):
            results.extend(process_interval(self.values[:, i], self.ids[:, i], player_mark=self.player_mark,
                                            opp_mark=self.opp_mark))
        return results

    def process_rows(self):
        results = []
        for i in range(self.values.shape[0]):
            results.extend(process_interval(self.values[i, :], self.ids[i, :], player_mark=self.player_mark,
                                            opp_mark=self.opp_mark))
        return results


class Diagonal:
    def __init__(self, board_values, ids, left_to_right, offset):
        if left_to_right:
            self.values = board_values.diagonal(offset)
            self.ids = ids.diagonal(offset)
        else:
            self.values = np.diagonal(np.fliplr(board_values), offset=offset)
            self.ids = np.diagonal(np.fliplr(ids), offset=offset)

        self.offset = offset
        self.left_to_right = left_to_right


def my_agent(obs, config):
    board = Board(obs.board, obs.mark)

    play = board.play()

    if play is None:
        play = iebot(obs, config)

    return int(play)