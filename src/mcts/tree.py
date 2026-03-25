import numpy as np
from src.game.connect_four import ConnectFourGame, BoardState
from dataclasses import dataclass, field
from src.utils import FloatArray, IntArray
from copy import deepcopy
from itertools import count
_id_counter = count()

@dataclass
class Action:
    move: int
    prior: float = 0.0 # Placeholder for neural net later

@dataclass
class Node:
    id: int
    game: ConnectFourGame
    n_visits: int
    n_wins: int
    untried_actions: list[Action]
    parent: 'Node | None' = None
    parent_action: Action | None = None
    children: list['Node'] = field(default_factory=list)

    @classmethod
    def as_root_node_from_game(
        cls, 
        game: ConnectFourGame
    ) -> 'Node':
        
        board_state: BoardState = game.board_state

        return Node(
            id = 0,
            game = game,
            n_visits = 0,
            n_wins = 0,
            untried_actions= [Action(move = m) for m in board_state.get_legal_moves()],
            parent = None,
            parent_action= None
        )
    
    def make_child_from_action(
        self,
        action: Action
    ) -> 'Node':
        new_game = self.game.copy()
        new_game.make_move(move = action.move)
        new_board_state = new_game.board_state

        child_node = Node(
            id = next(_id_counter),
            game = new_game,
            n_visits = 0,
            n_wins = 0,
            untried_actions= [Action(move = m) for m in new_board_state.get_legal_moves()],
            parent = self,
            parent_action= action
        )

        self.children.append(child_node)
        self.untried_actions.remove(action)

        return child_node


@dataclass
class Tree:
    root_node: Node
    nodes: list[Node] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes.append(Node)