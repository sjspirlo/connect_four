import numpy as np
from src.mcts.tree import Node, Action, Tree
from dataclasses import dataclass
from src.utils import FloatArray, IntArray
from src.game.connect_four import ConnectFourGame

@dataclass
class MCTS:
    root_node: Node
    rng: np.random.Generator
    neural_net_mode: bool = False
    ucb1_exploration_param: float = np.sqrt(2)

    def __post_init__(self) -> None:
        self.tree = Tree(root_node=self.root_node)

        if self.neural_net_mode:
            self.selection_prior = ...
        else:
            self.selection_prior = lambda node: self._uniform_prior(n=len(node.untried_actions))

    def _uniform_prior(self, n) -> FloatArray:
        return np.ones(n) / n
    
    def _UCB1_score(
        self, 
        node: Node
    ) -> float:
        assert node.n_visits > 0, 'Unvisited node encountered in UCB1 score calculation'
        exploitation_term = node.n_wins / node.n_visits
        exploration_term = self.ucb1_exploration_param*np.sqrt(np.log(node.parent.n_visits) / node.n_visits)
        return exploitation_term + exploration_term

    def _run_iteration(self) -> None:
        # Traverse path through tree in selection step
        path = self._selection()
        last_selected_node = path[-1]

        # If last encoutered node is terminal, omit expansion
        if last_selected_node.game.result is not None:
            result = last_selected_node.game.result

        # if not, perform expansion
        else:
            expanded_node = self._expansion(starting_node=path[-1])

            # Add the new node to the tree and path
            _ = path.append(expanded_node)
            _ = self.tree.nodes.append(expanded_node)

            # Monte Carlo rollout until finished game from new node, +1 for win, 0 for draw, -1 for loss
            result = self._simulation(starting_node = expanded_node)


        # Update statistics of all nodes in path
        _ = self._backpropagation(
            path=path, 
            result=result
        )
    
    def _select(
        self,
        starting_node: Node
    ) -> Node:
        assert not starting_node.untried_actions, 'Non fully-expanded node encountered in _select method'
        ucb1_scores = np.array([self._UCB1_score(node = n) for n in starting_node.children])
        return starting_node.children[np.argmax(ucb1_scores)]

    def _selection(
        self
    ) -> list[Node]:
        path = [self.root_node]
        last_node = self.root_node

        # Selection should stop at any not fully-expanded node
        while last_node.children and not last_node.untried_actions: 
            next_node = self._select(starting_node = last_node)
            path.append(next_node)
            last_node = next_node   
        
        print(f'Selection stopped at depth {len(path)-1}, \
              untried: {len(last_node.untried_actions)}, children: {len(last_node.children)}, terminal: {last_node.game.result is not None}')
        return path

    def _expansion(
        self,
        starting_node: Node
    ) -> Node:
        action_probabilities = self.selection_prior(node = starting_node)
        action_idx = self.rng.choice(len(starting_node.untried_actions), p=action_probabilities)
        action = starting_node.untried_actions[action_idx]

        expanded_node = starting_node.make_child_from_action(
            action = action
        )

        return expanded_node

    def _simulation(
        self,
        starting_node: Node
    ) -> int:
        rollout_game = starting_node.game.copy()

        while rollout_game.result is None:
            legal_moves = rollout_game.board_state.get_legal_moves()
            assert len(legal_moves) > 0, f'No legal moves but result is {rollout_game.result}\nBoard:\n{rollout_game.board_state.state}'
            random_move = self.rng.choice(legal_moves)
            _ = rollout_game.make_move(move = random_move)
        
        return rollout_game.result

    def _backpropagation(
        self, 
        result: int,
        path: list[Node]
    ) -> None:
        for node in path:
            node.n_visits += 1
            if result == node.game.whose_turn:
                node.n_wins += 1

    def run(self, n_iters: int) -> Action:
        for _ in range(n_iters):
            # print(len(self.tree.nodes), self.root_node.n_visits)
            self._run_iteration()
        
        # Select most visited action for robustness, rather than one with highest win-rate
        optimal_action = max(self.root_node.children, key=lambda c: c.n_visits).parent_action

        # # Select most visited action for robustness, rather than one with highest win-rate
        # optimal_action = max(self.root_node.children, key=lambda c: c.n_visits).parent_action

        return optimal_action