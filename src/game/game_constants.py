from dataclasses import dataclass


@dataclass
class BoardProperties:
    N_ROWS: int = 6
    N_COLS: int = 7

    @property
    def N_FIELDS(self) -> int:
        return int(self.N_COLS * self.N_ROWS)
