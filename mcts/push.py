"""Class for MCTS."""

import math

from constants import (
    MCTS_MAX_LEVEL,
    GRASP_Q_PUSH_THRESHOLD,
)


class PushMove:
    """Represent a move from start to end pose"""

    def __init__(self, pos0, pos1):
        self.pos0 = pos0
        self.pos1 = pos1

    def __str__(self):
        return f"{self.pos0[0]}_{self.pos0[1]}_{self.pos1[0]}_{self.pos1[1]}"

    def __repr__(self):
        return f"start: {self.pos0} to: {self.pos1}"

    def __eq__(self, other):
        return self.pos0 == other.pos0 and self.pos1 == other.pos1

    def __hash__(self):
        return hash((self.pos0, self.pos1))


class PushState:
    """Use move_recorder and simulation_recorder from simulation.
    move_recorder, Key is uid: '(key of this)'.
    simulation_recorder, Key is the uid: '(key of this) + (level)' + (move).
    """

    # TODO: how to get a good max_q, which could be used to decide an object is graspable
    def __init__(
        self,
        uid,
        object_states,
        q_value,
        level,
        mcts_helper,
        max_q=GRASP_Q_PUSH_THRESHOLD,
        max_level=MCTS_MAX_LEVEL,
        prev_angle=None,
        prev_move=None,
    ):
        self.uid = uid
        self.object_states = object_states
        self.q_value = q_value
        self.level = level
        self.mcts_helper = mcts_helper
        self.max_q = max_q
        self.max_level = max_level
        self.prev_angle = prev_angle
        self.prev_move = prev_move

    @property
    def push_result(self):
        """Return the grasp q value"""
        result = self.q_value
        result = min(result, 1)
        result = max(result, 0)
        result *= 0.2
        if self.q_value > self.max_q:
            result += 1
        return result

    def is_push_over(self):
        """Should stop the search"""
        # if reaches the last defined level or the object can be grasp
        if self.level == self.max_level or self.q_value > self.max_q:
            return True

        # if no legal actions
        if self.uid in self.mcts_helper.move_recorder:
            if len(self.mcts_helper.move_recorder[self.uid]) == 0:
                return True

        # if not over - no result
        return False

    def _move_result(self, move, is_consecutive_move=False):
        """Return the result after a move"""
        key = self.uid + str(move)

        if key not in self.mcts_helper.simulation_recorder:
            result = self.mcts_helper.simulate(
                move.pos0, move.pos1, self.object_states if not is_consecutive_move else None
            )
            if result is None:
                return None
            color_image, depth_image, mask_image, object_states = result
            new_image_q, _, _ = self.mcts_helper.get_grasp_q(
                color_image, depth_image, post_checking=True
            )
            self.mcts_helper.simulation_recorder[key] = object_states, new_image_q
            in_recorder = False
        else:
            self.mcts_helper.env.restore_objects(self.object_states)
            object_states, new_image_q = self.mcts_helper.simulation_recorder[key]
            color_image, mask_image = None, None
            in_recorder = True

        return object_states, new_image_q, color_image, mask_image, in_recorder

    def move(self, move, is_consecutive_move=False):
        result = self._move_result(move, is_consecutive_move=is_consecutive_move)
        if result is None:
            return None
        object_states, new_image_q, color_image, mask_image, in_recorder = result
        push_angle = math.atan2(move.pos1[1] - move.pos0[1], move.pos1[0] - move.pos0[0])
        move_in_image = ((move.pos0[1], move.pos0[0]), (move.pos1[1], move.pos1[0]))
        return (
            PushState(
                f"{self.uid}.{self.level}-{move}",
                object_states,
                new_image_q,
                self.level + 1,
                self.mcts_helper,
                max_q=self.max_q,
                max_level=self.max_level,
                prev_angle=push_angle,
                prev_move=move_in_image,
            ),
            color_image,
            mask_image,
            in_recorder,
        )

    def get_actions(self, color_image=None, mask_image=None):
        key = self.uid
        if key not in self.mcts_helper.move_recorder:
            actions = self.mcts_helper.sample_actions(
                self.object_states, color_image, mask_image, plot=False
            )
            moves = []
            for action in actions:
                moves.append(PushMove(action[0], action[1]))
            self.mcts_helper.move_recorder[key] = moves
        else:
            moves = self.mcts_helper.move_recorder[key]

        return moves

    def remove_action(self, move):
        key = self.uid
        if key in self.mcts_helper.move_recorder:
            moves = self.mcts_helper.move_recorder[key]
            if move in moves:
                moves.remove(move)
