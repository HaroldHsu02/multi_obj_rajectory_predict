from .tree import T_tree


def t_tree_dfs_prediction_recursive(
    current_path_nodes, current_treenode, target_sequence_keys, all_predicted_paths_list
):
    """
    Recursive DFS function to find predicted next RSUs.
    Args:
        current_path_nodes (list): List of T_tree nodes matching target_sequence_keys so far.
        current_treenode (T_tree): The current T_tree node in the traversal.
        target_sequence_keys (list): The history of RSU keys (strings) we are trying to match.
        all_predicted_paths_list (list): List to append full predicted paths to.
    """
    # Base case 1: Successfully matched the entire target_sequence_keys
    if len(current_path_nodes) == len(target_sequence_keys):
        if current_treenode.child:  # If there are children, these are potential predictions
            for next_potential_node in current_treenode.child:
                # Copy current matched path
                path_with_prediction = list(current_path_nodes)
                path_with_prediction.append(
                    next_potential_node)  # Add the predicted node
                all_predicted_paths_list.append(path_with_prediction)
        return

    # Base case 2: Mismatch or dead end before matching full target_sequence_keys
    if not current_treenode.child and len(current_path_nodes) < len(target_sequence_keys):
        return

    # Recursive step: Try to match the next key in target_sequence_keys
    if len(current_path_nodes) < len(target_sequence_keys):
        key_to_match = target_sequence_keys[len(current_path_nodes)]
        found_match_in_children = False
        for child_node in current_treenode.child:
            if child_node.key == key_to_match:
                found_match_in_children = True
                current_path_nodes.append(child_node)
                t_tree_dfs_prediction_recursive(
                    current_path_nodes,
                    child_node,
                    target_sequence_keys,
                    all_predicted_paths_list,
                )
                current_path_nodes.pop()  # Backtrack: remove node for other sibling paths


def predict_next_rsu(tree_root, history_rsu_sequence):
    """
    Predicts the next RSU based on the history.
    Args:
        tree_root (T_tree): Root of the prediction tree
        history_rsu_sequence (list): History of RSU IDs
    Returns:
        str: Predicted next RSU ID, or None if no prediction can be made
    """
    if not history_rsu_sequence:
        return None

    all_predicted_paths = []
    t_tree_dfs_prediction_recursive(
        [], tree_root, history_rsu_sequence, all_predicted_paths)

    if not all_predicted_paths:
        return None

    candidate_predictions = {}  # RSU_key -> total_support
    for path in all_predicted_paths:
        if path:
            predicted_node = path[-1]
            support = predicted_node.support
            key = predicted_node.key
            candidate_predictions[key] = candidate_predictions.get(
                key, 0) + support

    if not candidate_predictions:
        return None

    best_rsu_key = None
    highest_score = -1
    for rsu_key, total_support in candidate_predictions.items():
        if total_support > highest_score:
            highest_score = total_support
            best_rsu_key = rsu_key

    return best_rsu_key
