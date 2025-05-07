import numpy as np
import os
from .utils import get_closest_rsu


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
            if not isinstance(coord_pair, np.ndarray):
                coord_pair = np.array(coord_pair)

            if np.isnan(coord_pair).any():
                continue

            closest_rsu = get_closest_rsu(coord_pair, rsu_map)
            if closest_rsu:
                if not rsu_sequence or rsu_sequence[-1] != closest_rsu:
                    rsu_sequence.append(closest_rsu)

        if len(rsu_sequence) > 1:
            all_rsu_sequences.append(rsu_sequence)

    print(f"Loaded and transformed {len(all_rsu_sequences)} RSU sequences.")
    if all_rsu_sequences:
        print(f"Example RSU sequence: {all_rsu_sequences[0][:10]}")
    return all_rsu_sequences
