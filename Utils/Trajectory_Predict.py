# coding=utf8
import numpy as np
import time
import os
from collections import namedtuple

# Define Point for clarity if needed, though not strictly used by RSU mapping here
Point = namedtuple("Point", ["x", "y"])

# --- Configuration for User's Experimental Setup ---
# RSU_LOCATIONS are defined in the original 1500m x 1500m space
# RSU IDs and their original coordinates (centers of 1km x 1km coverage)
# "田" layout in a 1500x1500 area:
# RSU0 (top-left), RSU1 (top-right), RSU2 (bottom-left), RSU3 (bottom-right)
RSU_COORDS = {
    "RSU_0": np.array([375.0, 375.0]),
    "RSU_1": np.array([1125.0, 375.0]),
    "RSU_2": np.array([375.0, 1125.0]),
    "RSU_3": np.array([1125.0, 1125.0]),
}

# Path to the processed .npy data file
# Assuming the .npy file is in a subdirectory 'data' relative to this script
# Replace with the actual path to your 'rome_trajectory.npy' file
DATA_FILE_PATH = "/Datasets/Datasets/rome_trajectory_100_400.npy"
# Example: DATA_FILE_PATH = "/path/to/your/Datasets/Datasets/rome_trajectory.npy"

# --- Helper Functions for RSU Mapping and Data Processing ---


def calculate_distance(point1, point2):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def get_closest_rsu(vehicle_coord, rsu_map_scaled):
    """
    Finds the closest RSU to the vehicle's current coordinates.
    Args:
        vehicle_coord (np.array): The vehicle's [x, y] coordinates (scaled).
        rsu_map_scaled (dict): A dictionary mapping RSU IDs to their scaled [x, y] coordinates.
    Returns:
        str: The ID of the closest RSU.
    """
    min_dist = float("inf")
    closest_rsu_id = None
    for rsu_id, rsu_coord in rsu_map_scaled.items():
        dist = calculate_distance(vehicle_coord, rsu_coord)
        if dist < min_dist:
            min_dist = dist
            closest_rsu_id = rsu_id
    return closest_rsu_id


def load_and_transform_data(npy_filepath, rsu_map):
    """
    Loads vehicle trajectories from the .npy file and transforms coordinate
    sequences into RSU ID sequences.
    Args:
        npy_filepath (str): Path to the .npy file.
                            Expected shape: (num_timesteps, num_vehicles, 2)
        rsu_map_scaled (dict): Dictionary of RSU IDs to coordinates.
    Returns:
        list: A list of lists, where each inner list is a vehicle's trajectory
              represented as a sequence of RSU IDs.
              Example: [['RSU_0', 'RSU_1', 'RSU_0'], ['RSU_2', 'RSU_3']]
    """
    if not os.path.exists(npy_filepath):
        print(f"Error: Data file not found at {npy_filepath}")
        print("Please ensure 'rome_trajectory.npy' exists or update DATA_FILE_PATH.")
        # Create a dummy file for demonstration if it doesn't exist
        print("Creating a dummy 'rome_trajectory.npy' for demonstration purposes.")
        dummy_data_dir = os.path.dirname(npy_filepath)
        if dummy_data_dir and not os.path.exists(dummy_data_dir):
            os.makedirs(dummy_data_dir)
        # Dummy data: 5 timesteps, 2 vehicles, 2D coords (already scaled)
        # Vehicle 1: moves near RSU_0 then RSU_1
        # Vehicle 2: moves near RSU_2 then RSU_3
        dummy_vehicle1_coords = np.array(
            [
                [180.0, 180.0],
                [190.0, 190.0],
                [500.0, 180.0],
                [550.0, 190.0],
                [560.0, 190.0],
            ]
        )
        dummy_vehicle2_coords = np.array(
            [
                [180.0, 550.0],
                [190.0, 560.0],
                [500.0, 550.0],
                [550.0, 560.0],
                [560.0, 560.0],
            ]
        )
        # 重塑数据为(num_timesteps, num_vehicles, 2)格式
        dummy_trajectories = np.stack(
            [dummy_vehicle1_coords, dummy_vehicle2_coords], axis=1)
        np.save(npy_filepath, dummy_trajectories)
        print(f"Dummy data saved to {npy_filepath}")

    vehicle_coord_trajectories = np.load(npy_filepath, allow_pickle=True)

    # 转置数据以获取每个车辆的轨迹
    # 从(num_timesteps, num_vehicles, 2)转换为(num_vehicles, num_timesteps, 2)
    vehicle_coord_trajectories = np.transpose(
        vehicle_coord_trajectories, (1, 0, 2))

    all_rsu_sequences = []
    for vehicle_coords in vehicle_coord_trajectories:
        rsu_sequence = []
        for coord_pair in vehicle_coords:
            # Ensure coord_pair is a numpy array for get_closest_rsu
            if not isinstance(coord_pair, np.ndarray):
                coord_pair = np.array(coord_pair)

            # Skip if coordinates are NaN or invalid
            if np.isnan(coord_pair).any():
                continue

            closest_rsu = get_closest_rsu(coord_pair, rsu_map)
            if closest_rsu:
                # Add RSU to sequence only if it's different from the last one,
                # or if it's the first one
                if not rsu_sequence or rsu_sequence[-1] != closest_rsu:
                    rsu_sequence.append(closest_rsu)

        if len(rsu_sequence) > 1:  # Only consider sequences with at least 2 RSU transitions
            all_rsu_sequences.append(rsu_sequence)

    print(f"Loaded and transformed {len(all_rsu_sequences)} RSU sequences.")
    if all_rsu_sequences:
        print(f"Example RSU sequence: {all_rsu_sequences[0][:10]}")
    return all_rsu_sequences


def split_data(all_trajectories, train_ratio=0.8):
    """Splits trajectories into training and testing sets."""
    np.random.shuffle(all_trajectories)  # Shuffle for random split
    split_idx = int(len(all_trajectories) * train_ratio)
    train_set = all_trajectories[:split_idx]
    test_set = all_trajectories[split_idx:]
    print(
        f"Data split: {len(train_set)} training sequences, {len(test_set)} testing sequences."
    )
    return train_set, test_set


# --- T_tree Class and Prediction Logic (from bayonet_prediction_ex1.py) ---
# Minor modifications:
# - Made `result_list` a parameter/return value for `t_tree_dfs_prediction` and `predict`
#   to avoid global variables.


class T_tree:
    def __init__(self, name):
        # key是名字 child是子节点，support是支持度
        self.key = name
        self.child = []
        self.support = 1

    def get(self, getname):
        # 遍历每一个子节点
        for each_child in self.child:
            # 如果找到名字一样的
            if each_child.key == getname:
                # 那就准备返回这个节点
                return each_child  # Modified to return directly
        return None  # Return None if not found

    def insert(self, newname):
        # 这里和get是一样的，只不过就是support+1
        if self.child:
            for each_child in self.child:
                if each_child.key == newname:
                    each_child.support += 1
                    return 0  # Indicates existing node support incremented
        # 如果没找到，那就新建一个对象，然后把对象放在子节点列表中
        node = T_tree(newname)
        self.child.append(node)
        return 1  # Indicates new node added


class New_Node_Support:  # Renamed from new_node to avoid confusion, used for aggregated support
    def __init__(self):
        self.key = ""
        self.support = 1


def build_t_tree(train_rsu_sequences):
    """Builds the T_tree from training RSU sequences."""
    tree = T_tree("root")
    for each_individual_sequence in train_rsu_sequences:
        current_node = tree
        for point_idx, point_key in enumerate(each_individual_sequence):
            # Check if child exists
            existing_child = current_node.get(point_key)
            if existing_child:
                existing_child.support += 1  # Increment support if path segment exists
                current_node = existing_child
            else:
                # If not, insert new node and move to it
                new_child_node = T_tree(point_key)  # Support is 1 by default
                current_node.child.append(new_child_node)
                current_node = new_child_node
    print("T_tree built.")
    return tree


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
                                         Each path is [hist_node1, ..., hist_nodeN, predicted_next_node].
    """
    # Base case 1: Successfully matched the entire target_sequence_keys
    if len(current_path_nodes) == len(target_sequence_keys):
        if (
            current_treenode.child
        ):  # If there are children, these are potential predictions
            for next_potential_node in current_treenode.child:
                path_with_prediction = list(
                    current_path_nodes
                )  # Copy current matched path
                path_with_prediction.append(
                    next_potential_node
                )  # Add the predicted node
                all_predicted_paths_list.append(path_with_prediction)
        return  # Stop recursion for this branch after finding predictions or no children

    # Base case 2: Mismatch or dead end before matching full target_sequence_keys
    if not current_treenode.child and len(current_path_nodes) < len(
        target_sequence_keys
    ):
        return  # Cannot continue matching

    # Recursive step: Try to match the next key in target_sequence_keys
    if len(current_path_nodes) < len(target_sequence_keys):
        key_to_match = target_sequence_keys[len(current_path_nodes)]
        found_match_in_children = False
        for child_node in current_treenode.child:
            if child_node.key == key_to_match:
                found_match_in_children = True
                # Add matched node to path
                current_path_nodes.append(child_node)
                t_tree_dfs_prediction_recursive(
                    current_path_nodes,
                    child_node,
                    target_sequence_keys,
                    all_predicted_paths_list,
                )
                current_path_nodes.pop()  # Backtrack: remove node for other sibling paths

        # If the current key_to_match was not found in children, this path is a dead end for the target_sequence_keys
        # No explicit action needed here, the recursion will simply not proceed further down this path for this target.


def predict_next_rsu(tree_root, history_rsu_sequence):
    """
    Predicts the next RSU based on the history.
    Adapts the logic from the original bayonet_prediction_ex1.py's predict function.
    """
    if not history_rsu_sequence:
        return None  # Cannot predict with no history

    all_predicted_paths = []  # This will store lists of T_tree nodes

    # We need to find the starting node in the tree that matches the first element of history_rsu_sequence
    # The DFS should explore paths from the root that match the history_rsu_sequence.

    # Initial call to recursive DFS:
    # Start with an empty path of nodes, from the tree_root, targeting the history_rsu_sequence.
    t_tree_dfs_prediction_recursive(
        [], tree_root, history_rsu_sequence, all_predicted_paths
    )

    if not all_predicted_paths:
        return None  # No prediction found

    # --- Process all_predicted_paths to find the best prediction ---
    # This part mimics the original script's logic for aggregating support and finding the highest score.

    # 1. Deduplicate paths (original script used `if i not in lst1:`)
    #    Here, paths are lists of T_tree objects. True deduplication would be by sequence of keys.
    #    For simplicity, let's assume distinct paths from DFS are what we need.
    #    The original script's deduplication might have been to handle multiple ways DFS found same sequence.

    # 2. Aggregate support for identical *predicted RSU keys*.
    #    The original script's `get_list` and `New_Node_Support` aimed to do this.
    #    A simpler approach: collect all predicted next_node.key and their supports.

    candidate_predictions = {}  # RSU_key -> total_support
    for path in all_predicted_paths:
        # The last node in each path is the predicted_next_node
        if path:  # Path should not be empty if populated correctly
            predicted_node = path[-1]
            # The support we care about is that of the predicted_node itself
            # as it represents frequency of that RSU appearing after the history.
            support = predicted_node.support
            key = predicted_node.key
            candidate_predictions[key] = candidate_predictions.get(
                key, 0) + support

    if not candidate_predictions:
        return None

    # Find the RSU key with the highest aggregated support
    best_rsu_key = None
    highest_score = -1
    for rsu_key, total_support in candidate_predictions.items():
        if total_support > highest_score:
            highest_score = total_support
            best_rsu_key = rsu_key

    return best_rsu_key


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting RSU-based trajectory prediction adaptation...")
    print(f"RSU coordinates: {RSU_COORDS}")

    # 1. Load and transform data
    # This function now also creates dummy data if DATA_FILE_PATH is not found.
    all_rsu_sequences = load_and_transform_data(DATA_FILE_PATH, RSU_COORDS)

    if not all_rsu_sequences:
        print("No RSU sequences loaded or generated. Exiting.")
    else:
        # 2. Split data
        train_sequences, test_sequences = split_data(
            all_rsu_sequences, train_ratio=0.8)

        if not train_sequences:
            print("No training sequences available after split. Exiting.")
        else:
            # 3. Build T_tree
            # The original main() in bayonet_prediction_ex1.py built the tree.
            # We adapt that here.

            # --- Building the T-Tree (adapted from original main()) ---
            # Root of the prediction tree
            prediction_tree_root = T_tree("root")

            for rsu_seq in train_sequences:
                current_t_node = prediction_tree_root
                for rsu_id_idx, rsu_id_key in enumerate(rsu_seq):
                    # Try to find the RSU ID in the current node's children
                    child_found = False
                    for child in current_t_node.child:
                        if child.key == rsu_id_key:
                            child.support += 1
                            current_t_node = child
                            child_found = True
                            break
                    if not child_found:
                        # If RSU ID not found, insert it as a new child
                        # Support is 1 initially
                        new_child = T_tree(rsu_id_key)
                        current_t_node.child.append(new_child)
                        current_t_node = new_child
            print("T_tree built successfully from training data.")

            # 4. Evaluate predictions
            correct_predictions = 0
            total_predictions = 0

            if not test_sequences:
                print("No test sequences to evaluate.")
            else:
                print(
                    f"\nEvaluating on {len(test_sequences)} test sequences...")
                for test_idx, full_rsu_sequence in enumerate(test_sequences):
                    if len(full_rsu_sequence) < 2:
                        continue  # Need at least one history point and one actual next point

                    # Ensure it's a list copy
                    history = list(full_rsu_sequence[:-1])
                    actual_next_rsu = full_rsu_sequence[-1]

                    # The prediction attempt loop from original script
                    predicted_rsu = None
                    # Work with a copy for shortening
                    temp_history = list(history)

                    while not predicted_rsu and temp_history:
                        predicted_rsu = predict_next_rsu(
                            prediction_tree_root, temp_history
                        )
                        if not predicted_rsu:
                            temp_history = temp_history[
                                1:
                            ]  # Shorten history from the front

                    if predicted_rsu:
                        total_predictions += 1
                        if predicted_rsu == actual_next_rsu:
                            correct_predictions += 1
                        # if test_idx < 20 : # Print some sample predictions
                        #     print(f"  Test {test_idx}: History: {history}, Actual: {actual_next_rsu}, Predicted: {predicted_rsu} {'CORRECT' if predicted_rsu == actual_next_rsu else 'WRONG'}")
                    # else:
                    # if test_idx < 20 :
                    #    print(f"  Test {test_idx}: History: {history}, Actual: {actual_next_rsu}, Predicted: None (could not predict)")

                if total_predictions > 0:
                    accuracy = (correct_predictions / total_predictions) * 100
                    print(
                        f"\nPrediction Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})"
                    )
                else:
                    print("\nNo predictions could be made on the test set.")
