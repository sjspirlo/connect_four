import numpy as np
from dataclasses import dataclass, field
from src.game.game_constants import BoardProperties
from typing import TypeAlias
from numpy.typing import NDArray
from src.utils import FloatArray, IntArray

BOARD_PROPERTIES = BoardProperties()

@dataclass
class BoardState:
    state: IntArray # N_ROWS x N_COLS array of -1, 0, 1 values representing the board.

    def __post_init__(self):
        assert self.state.shape == (6, 7), 'State array has incorrect shape'
        assert np.issubdtype(self.state.dtype, np.integer), 'State array needs to contain values of integer type' 

    @classmethod
    def as_new_board(cls) -> 'BoardState':
        return BoardState(
            state = np.zeros((BOARD_PROPERTIES.N_ROWS, BOARD_PROPERTIES.N_COLS), dtype = np.int8)
        )

    @property
    def board_is_full(
        self
    ) -> bool:
        return np.count_nonzero(self.state) == BOARD_PROPERTIES.N_FIELDS
    
    def col_is_full(
        self,
        col: int
    ) -> bool:
        return np.count_nonzero(self.state[:, col]) == BOARD_PROPERTIES.N_ROWS
    
    def get_legal_moves(
        self
    ) -> IntArray:
        legal_moves = [int(col) for col in range(BOARD_PROPERTIES.N_COLS) if not self.col_is_full(col=col)]
        return np.array(legal_moves, dtype=np.int8)
    
    def get_next_row(
        self,
        col: int
    ) -> int:
        return int(BOARD_PROPERTIES.N_ROWS - 1 - np.sum(self.state[:, col] != 0))

@dataclass
class ConnectFourGame:
    board_state: BoardState
    whose_turn: int # 1 for white, -1 for black
    result: int | None = field(default=None)
    verbose: bool = field(default=False)

    def __post_init__(self):
        assert self.whose_turn in [1, -1]

    def copy(self) -> 'ConnectFourGame':
        return ConnectFourGame(
            board_state=BoardState(state=self.board_state.state.copy()),
            whose_turn=self.whose_turn,
            result=self.result,
            verbose=False
        )
    
    @classmethod
    def as_new_game(
        cls, 
        verbose: bool = False
    ) -> 'ConnectFourGame':
        return cls(
            board_state = BoardState.as_new_board(),
            whose_turn = 1,
            verbose = verbose,
            result = None
        )
    
    @property
    def player(self) -> str:
        return 'White' if self.whose_turn == 1 else 'Black'
    
    def is_legal(
        self,
        move: int, 
    ) -> bool:
        assert isinstance(move, (int, np.integer)), 'Provide move as zero based integer column index.'
        return move in self.board_state.get_legal_moves()
    
    def _check_for_winning_move(
        self, 
        row: int, 
        col: int
    ) -> bool:
        player = self.whose_turn
        axes = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in axes:
            count = 1
            for sign in [1, -1]:
                step = 1
                while True:
                    r = row + sign * step * dr
                    c = col + sign * step * dc
                    if not (0 <= r < BOARD_PROPERTIES.N_ROWS and 0 <= c < BOARD_PROPERTIES.N_COLS):
                        break
                    if self.board_state.state[r, c] != player:
                        break
                    count += 1
                    step += 1
            if count >= 4:
                return True
        return False

    def make_move(
        self, 
        move: int
    ) -> None:

        if self.result is not None:
            raise ValueError('Game is already over')
        
        if not self.is_legal(move = move):
            if self.verbose: print(f'Provided move ({move}) is not legal, no move made.')
            return None

        next_row = self.board_state.get_next_row(col = move)
        self.board_state.state[next_row, move] = self.whose_turn

        if self._check_for_winning_move(row = next_row, col = move):
            if self.verbose: print(f'GAME OVER - {self.player} won!')
            self.result = self.whose_turn
            return None

        if self.board_state.board_is_full:
            if self.verbose: print('GAME OVER - Board is full; no winner.')
            self.result = 0
            return None

        self.whose_turn = int(self.whose_turn*-1)

    def print_board(self) -> None:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in range(BOARD_PROPERTIES.N_ROWS):
            print(' '.join(symbols[cell] for cell in self.board_state.state[row]))
        print(' '.join(str(col) for col in range(BOARD_PROPERTIES.N_COLS)))
