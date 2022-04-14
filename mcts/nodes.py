"""Node for MCTS"""

import numpy as np
from constants import (
    MCTS_DISCOUNT,
    MCTS_TOP,
    MCTS_UCT_RATIO,
)

class PushSearchNode:
    """MCTS search node for push prediction."""

    def __init__(self, state, prev_move=None, parent=None):
        self.state = state
        self.prev_move = prev_move
        self.parent = parent
        self.children = []
        self._number_of_visits = 0
        self._results = []
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_actions().copy()
        return self._untried_actions

    @property
    def q(self):
        return self._results

    @property
    def n(self):
        return self._number_of_visits

    @property
    def has_children(self):
        return len(self.children) > 0

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal_node(self):
        return self.state.is_push_over() or (self.is_fully_expanded and not self.has_children)

    def expand(self):
        expanded = False
        child_node = self

        while len(self.untried_actions) > 0:
            action = self.untried_actions.pop()
            result = self.state.move(action)
            if result is None:
                self.state.remove_action(action)
            else:
                next_state, _, _, _ = result
                child_node = PushSearchNode(next_state, action, parent=self)
                self.children.append(child_node)
                expanded = True
                break

        return expanded, child_node

    def rollout(self):
        current_rollout_state = self.state
        discount_accum = 1
        results = [current_rollout_state.push_result]
        # discounts = []
        # cost = 0
        is_consecutive_move = False
        color_image = None
        mask_image = None
        while not current_rollout_state.is_push_over():
            possible_moves = current_rollout_state.get_actions(color_image, mask_image)
            if len(possible_moves) == 0:
                break
            action = self.rollout_policy(possible_moves)
            result = current_rollout_state.move(action, is_consecutive_move=is_consecutive_move)
            if result is None:
                if current_rollout_state == self.state:
                    self.untried_actions.remove(action)
                current_rollout_state.remove_action(action)
                is_consecutive_move = False
            else:
                # cost += MCTS_STEP_COST
                discount_accum *= MCTS_DISCOUNT
                new_rollout_state, color_image, mask_image, in_recorder = result
                current_rollout_state = new_rollout_state
                results.append(current_rollout_state.push_result * discount_accum)
                # results.append(current_rollout_state.push_result - cost)
                # discounts.append(discount_accum)
                if in_recorder:
                    is_consecutive_move = False
                else:
                    is_consecutive_move = True

        # if len(results) > 0:
        #     result_idx = np.argmax(results)
        #     return results[result_idx] * discounts[result_idx], results[result_idx]
        # else:
        #     return (
        #         current_rollout_state.push_result * discount_accum,
        #         current_rollout_state.push_result,
        #     )
        return np.max(results)

    def backpropagate(self, result):
        self._number_of_visits += 1
        # if high_q <= self.state.push_result:
        #     high_q = self.state.push_result
        #     result = high_q
        # if result < self.state.push_result:
        #     result = self.state.push_result
        self._results.append(result)
        if self.parent:
            # discount_factor = MCTS_DISCOUNT
            self.parent.backpropagate(result * MCTS_DISCOUNT)
            # self.parent.backpropagate(result - MCTS_STEP_COST)

    def best_child(self, c_param=MCTS_UCT_RATIO, top=MCTS_TOP):
        choices_weights = [
            (sum(sorted(c.q)[-top:]) / min(c.n, top))
            + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def best_child_top(self):
        # choices_weights = [(sum(sorted(c.q)[-top:]) / min(c.n, top)) for c in self.children]
        choices_weights = [max(c.q) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

