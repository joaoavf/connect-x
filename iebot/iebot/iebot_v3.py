"""
This is a MiniMax Connect 4 agent.

Inspired by:
https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
http://blog.gamesolver.org/

@author: João Alexandre Vaz Ferreira - joao.avf@gmail.com
"""

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

    def create_children(self, plays):
        for play in plays:
            if play and self.recursiveness > 0:
                new_bit_board = self.bit_board | play
                new_mask = self.mask | play
                self.children.append(Node(bit_board=new_bit_board ^ new_mask,
                                          mask=new_mask,
                                          columns_map=self.columns_map,
                                          recursiveness=self.recursiveness - 1,
                                          play=play))

    def alpha_beta_pruning(self, plays):
        for play in plays:  # This is where the alpha-beta pruning happens, as it prevents the creation of new children
            if play:
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
        is_game_won, play_position = self.alpha_beta_pruning(plays)  # Has children if node is not final

        if is_game_won or not self.children:  # If is game won or if last node due to recursiveness limit
            return [is_game_won, play_position]
        else:
            return self.negamax()  # NegaMax on children output

    def get_column_to_play(self):
        """Return position of the column where the play was made..

           Returns:
           int : column position"""

        for index, column in enumerate(self.columns_map):
            if self.play in column:
                return index


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
        mask += b'0'  # Add 0-bits to sentinel # TODO understand why?
        position += b'0'

        for i in range(0, 6):  # Start with bottom row
            mask += [b'0', b'1'][board[i, j] != 0]
            position += [b'0', b'1'][board[i, j] == player]

    return int(position, 2), int(mask, 2)


def generate_columns_map(mask):
    """Generates all valid moves per column on the Connect4 Board.

    Parameters:
    mask (int): binary representation (bit board) of all pieces

    Returns:
    List[List[int, ...]] : each nested listed correspond to column and its valid pieces left in binary representation"""

    position_map = [2 ** i for i in range(49)]  # List of a binary representation of individual pieces in the board

    columns_map = []
    for column_number in range(7):
        column_values = position_map[7 * column_number: 7 * column_number + 6]  # Minus extra cell on top of the board
        column_values = [cell_value for cell_value in column_values if mask & cell_value == 0]  # Removing full columns
        columns_map.append(column_values)

    return columns_map


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

    # TODO implement heuristics
    return node.get_column_to_play()
