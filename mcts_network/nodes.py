"""Node for MCTS"""

import math
import numpy as np
from constants import (
    MCTS_DISCOUNT,
)


class PushSearchNode:
    """MCTS search node for push prediction."""

    def __init__(self, state=None, prev_move=None, parent=None):
        self.state = state
        self.prev_move = prev_move
        self.parent = parent
        self.children = []
        self._number_of_visits = 1
        self._results = [0]
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_actions().copy()
        return self._untried_actions

    @property
    def nq(self):
        return self.prev_move.q_value

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
        for c in self.children:
            if c.state is None:
                return False
        return True

    def is_terminal_node(self):
        return self.state.is_push_over() or not self.has_children

    def pre_expand(self):
        while len(self.untried_actions) > 0:
            action = self.untried_actions.pop()
            self.children.append(PushSearchNode(None, action, parent=self))

    def expand(self):
        result = self.parent.state.move(self.prev_move)
        if result is None:
            self.parent.state.remove_action(self.prev_move)
            expanded = False
        else:
            next_state, _, _, _ = result
            self.state = next_state
            self.pre_expand()
            expanded = True

        return expanded

    def rollout(self):
        current_rollout_state = self.state
        discount_accum = 1
        results = [current_rollout_state.push_result]
        is_consecutive_move = False
        color_image = None
        mask_image = None
        while not current_rollout_state.is_push_over():
            possible_moves = current_rollout_state.get_actions(color_image, mask_image)
            if len(possible_moves) == 0:
                break
            action = self.rollout_policy(possible_moves)
            # use PPN for rollout
            # result = max(0, min(action.q_value, 1.2))
            # results.append(result * discount_accum)
            # break
            result = current_rollout_state.move(action, is_consecutive_move=is_consecutive_move)
            if result is None:
                if current_rollout_state == self.state:
                    for ci, c in enumerate(self.children):
                        if c.prev_move == action:
                            self.children.pop(ci)
                            break
                current_rollout_state.remove_action(action)
                is_consecutive_move = False
            else:
                discount_accum *= MCTS_DISCOUNT
                new_rollout_state, color_image, mask_image, in_recorder = result
                current_rollout_state = new_rollout_state
                results.append(current_rollout_state.push_result * discount_accum)
                if in_recorder:
                    is_consecutive_move = False
                else:
                    is_consecutive_move = True

        return np.max(results)

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result * MCTS_DISCOUNT)

    def best_child(self, top=3):
        choices_weights = [
            (sum(sorted(c.q)[-top:]) + 1 * max(0, min(c.nq, 1.2))) / c.n
            for c in self.children
        ]
        # if self.prev_move is None:
        #     temp = np.zeros((224, 224, 3))
        #     for idx, c in enumerate(self.children):
        #         node_action = str(c.prev_move).split("_")
        #         cv2.arrowedLine(
        #             temp,
        #             (int(node_action[1]), int(node_action[0])),
        #             (int(node_action[3]), int(node_action[2])),
        #             (255, 0, 255),
        #             1,
        #             tipLength=0.1,
        #         )
        #         cv2.putText(temp, f'{choices_weights[idx]:.2f}', (int(node_action[1]), int(node_action[0]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.5, (125, 125, 0), 1, cv2.LINE_AA)
        #     print(choices_weights)
        #     cv2.imshow('temp', temp)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        child_idx = np.argmax(choices_weights)
        return self.children[child_idx], child_idx

    def best_child_top(self):
        choices_weights = [max(c.q) + 1 * max(0, min(c.nq, 1.2)) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        highest_q = -math.inf
        best_move = None
        for move in possible_moves:
            if move.q_value > highest_q:
                highest_q = move.q_value
                best_move = move

        return best_move

