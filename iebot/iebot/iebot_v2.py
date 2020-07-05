import numpy as np
from iebot import iebot

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
            self.values = np.diagonal(np.fliplr(board_values))
            self.ids = np.diagonal(np.fliplr(ids))

        self.offset = offset
        self.left_to_right = left_to_right


def my_agent(obs, config):
    board = Board(obs.board, obs.mark)

    play = board.play()

    if play is None:
        play = iebot(obs, config)

    return int(play)