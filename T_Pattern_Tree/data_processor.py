import numpy as np
import os
import sys

# 添加项目根目录到Python路径（当直接运行此文件时）
if __name__ == "__main__":
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))

from T_Pattern_Tree.utils import get_closest_rsu


def load_and_transform_data(npy_filepath, rsu_map, transformed_data_path=None, force_transform=False):
    """
    Loads vehicle trajectories from the .npy file and transforms coordinate
    sequences into RSU ID sequences.
    Args:
        npy_filepath (str): Path to the .npy file.
                            Expected shape: (num_timesteps, num_vehicles, 2)
        rsu_map_scaled (dict): Dictionary of RSU IDs to coordinates.
        transformed_data_path (str, optional): Path to save/load transformed RSU sequences.
        force_transform (bool): Force re-transformation even if transformed data exists.
    Returns:
        np.ndarray: A numpy array of RSU sequences.
    """
    # 尝试加载已转换的数据
    if transformed_data_path and os.path.exists(transformed_data_path) and not force_transform:
        print(
            f"Loading pre-transformed RSU sequences from {transformed_data_path}")
        try:
            all_rsu_sequences = np.load(
                transformed_data_path, allow_pickle=True)
            print(
                f"Loaded {len(all_rsu_sequences)} pre-transformed RSU sequences.")
            print(f"RSU sequences shape: {all_rsu_sequences.shape}")
            if len(all_rsu_sequences) > 0:
                sequence_lengths = [len(seq) for seq in all_rsu_sequences]
                print(
                    f"RSU sequences shape: {len(all_rsu_sequences)} sequences with lengths ranging from {min(sequence_lengths)} to {max(sequence_lengths)}")
                print(
                    f"Average sequence length: {sum(sequence_lengths)/len(sequence_lengths):.2f}")
                print(f"Example RSU sequence: {all_rsu_sequences[0][:10]}")
            return all_rsu_sequences
        except Exception as e:
            print(
                f"Error loading transformed data: {e}. Will perform transformation.")

    # 确保目录存在
    os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)

    if not os.path.exists(npy_filepath):
        print(f"Error: Data file not found at {npy_filepath}")
        print("Creating dummy data file...")

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

    print(f"Transforming coordinate data from {npy_filepath}")
    try:
        vehicle_coord_trajectories = np.load(npy_filepath, allow_pickle=True)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return np.array([], dtype=object)

    # 转置数据以获取每个车辆的轨迹
    # 从(num_timesteps, num_vehicles, 2)转换为(num_vehicles, num_timesteps, 2)
    vehicle_coord_trajectories = np.transpose(
        vehicle_coord_trajectories, (1, 0, 2))
    # 打印转置后数组的形状
    print(f"Transposed array shape: {vehicle_coord_trajectories.shape}")

    print("Processing RSU sequences...")
    all_rsu_sequences = []
    for vehicle_coords in vehicle_coord_trajectories:
        # vehicle_coords的形状为(1202, 2)
        rsu_sequence = []
        for coord_pair in vehicle_coords:
            if not isinstance(coord_pair, np.ndarray):
                coord_pair = np.array(coord_pair)
            if np.isnan(coord_pair).any():
                continue

            closest_rsu = get_closest_rsu(coord_pair, rsu_map)
            if closest_rsu:
                if not rsu_sequence or rsu_sequence[-1] != closest_rsu:
                    # 如果rsu_sequence为空，或者rsu_sequence的最后一个元素不等于closest_rsu，则将closest_rsu添加到rsu_sequence中
                    rsu_sequence.append(closest_rsu)
        # 如果rsu_sequence的长度大于1，则将rsu_sequence添加到all_rsu_sequences中
        if len(rsu_sequence) > 1:
            all_rsu_sequences.append(rsu_sequence)

    # 将all_rsu_sequences转换为numpy数组
    all_rsu_sequences = np.array(all_rsu_sequences, dtype=object)

    # 保存转换后的数据
    if transformed_data_path:
        print(f"Saving transformed RSU sequences to {transformed_data_path}")
        np.save(transformed_data_path, all_rsu_sequences)

    print(f"Loaded and transformed {len(all_rsu_sequences)} RSU sequences.")
    # 打印all_rsu_sequences的形状
    print(f"RSU sequences shape: {all_rsu_sequences.shape}")
    sequence_lengths = [len(seq) for seq in all_rsu_sequences]
    print(f"RSU sequences shape: {len(all_rsu_sequences)} sequences with lengths ranging from {min(sequence_lengths) if sequence_lengths else 0} to {max(sequence_lengths) if sequence_lengths else 0}")
    print(
        f"Average sequence length: {sum(sequence_lengths)/len(sequence_lengths) if sequence_lengths else 0:.2f}")

    if all_rsu_sequences.size > 0:
        print(f"Example RSU sequence: {all_rsu_sequences[0][:10]}")
    return all_rsu_sequences
