from tqdm import tqdm


class MonteCarloTreeSearch(object):
    def __init__(self, node):
        self.root = node

    def best_action(self, simulations_number, early_stop_number, eval=False):
        early_stop_sign = False
        stop_level = 1
        for itr in tqdm(range(simulations_number)):
            child_node = self._tree_policy()
            reward = child_node.rollout()
            child_node.backpropagate(reward)
            if eval:
                # stop early if a solutin within one step has been found
                if child_node.state.level == stop_level and child_node.state.push_result > child_node.state.max_q:
                    early_stop_sign = True
                if itr > early_stop_number and early_stop_sign:
                    break
                if self.root.is_fully_expanded:
                    stop_level = 2  # TODO: good for now                    
        # to select best child go for exploitation only
        # return self.root.best_child(c_param=0.0)
        return self.root.best_child_top()

    def _tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded:
                expanded, node = current_node.expand()
                if expanded:
                    return node
            if current_node.has_children:
                current_node = current_node.best_child()
        return current_node
