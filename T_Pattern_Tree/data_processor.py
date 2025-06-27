import numpy as np
import os
import sys

# 添加项目根目录到Python路径（当直接运行此文件时）
if __name__ == "__main__":
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))

from T_Pattern_Tree.utils import get_closest_rsu
from T_Pattern_Tree.config import USE_TIMESLOT_MODE, NO_RSU_MARKER, RSU_COORDS


def load_and_transform_data(npy_filepath, rsu_map, transformed_data_path=None, force_transform=False):
    """
    从.npy文件加载车辆轨迹数据,并将坐标序列转换为RSU ID序列。

    参数:
        npy_filepath (str): .npy文件路径。
                           预期数据形状: (时间步数, 车辆数, 2)
        rsu_map_scaled (dict): RSU ID到坐标的映射字典。
        transformed_data_path (str, optional): 保存/加载转换后RSU序列的路径。
        force_transform (bool): 即使存在转换后的数据也强制重新转换。

    返回:
        np.ndarray: RSU序列的numpy数组。
    """
    # 如果存在已转换的数据且不强制转换,则直接加载
    if transformed_data_path and os.path.exists(transformed_data_path) and not force_transform:
        print(f"从{transformed_data_path}加载预转换的RSU序列")
        try:
            all_rsu_sequences = np.load(
                transformed_data_path, allow_pickle=True)
            print(f"已加载{len(all_rsu_sequences)}个预转换的RSU序列。")
            print(f"RSU序列形状: {all_rsu_sequences.shape}")
            if len(all_rsu_sequences) > 0:
                sequence_lengths = [len(seq) for seq in all_rsu_sequences]
                print(
                    f"RSU序列形状: {len(all_rsu_sequences)}个序列,长度范围从{min(sequence_lengths)}到{max(sequence_lengths)}")
                print(
                    f"平均序列长度: {sum(sequence_lengths)/len(sequence_lengths):.2f}")
                print(f"RSU序列示例: {all_rsu_sequences[0][:10]}")
            return all_rsu_sequences
        except Exception as e:
            print(f"加载转换数据时出错: {e}。将执行转换。")

    # 确保目录存在
    os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)

    # 如果数据文件不存在,创建虚拟数据
    if not os.path.exists(npy_filepath):
        print(f"错误:在{npy_filepath}未找到数据文件")
        print("创建虚拟数据文件...")

        # 虚拟数据:5个时间步,2辆车,2D坐标(已缩放)
        dummy_vehicle1_coords = np.array([
            [180.0, 180.0],
            [190.0, 190.0],
            [500.0, 180.0],
            [550.0, 190.0],
            [560.0, 190.0],
        ])
        dummy_vehicle2_coords = np.array([
            [180.0, 550.0],
            [190.0, 560.0],
            [500.0, 550.0],
            [550.0, 560.0],
            [560.0, 560.0],
        ])
        # 重塑数据为(时间步数,车辆数,2)格式
        dummy_trajectories = np.stack(
            [dummy_vehicle1_coords, dummy_vehicle2_coords], axis=1)
        np.save(npy_filepath, dummy_trajectories)
        print(f"虚拟数据已保存到{npy_filepath}")

    print(f"从{npy_filepath}转换坐标数据")
    try:
        vehicle_coord_trajectories = np.load(npy_filepath, allow_pickle=True)
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return np.array([], dtype=object)

    # 转置数据以获取每个车辆的轨迹
    # 从(时间步数,车辆数,2)转换为(车辆数,时间步数,2)
    vehicle_coord_trajectories = np.transpose(
        vehicle_coord_trajectories, (1, 0, 2))
    print(f"转置后数组形状: {vehicle_coord_trajectories.shape}")

    print("处理RSU序列...")
    all_rsu_sequences = []

    if USE_TIMESLOT_MODE:
        # 时序模式:记录每个时隙连接的RSU
        for vehicle_coords in vehicle_coord_trajectories:
            # vehicle_coords形状为(时间步数,2)
            rsu_sequence = []
            for coord_pair in vehicle_coords:
                if not isinstance(coord_pair, np.ndarray):
                    coord_pair = np.array(coord_pair)
                if np.isnan(coord_pair).any():
                    # 处理缺失值,使用无连接标记
                    rsu_sequence.append(NO_RSU_MARKER)
                    continue

                closest_rsu = get_closest_rsu(coord_pair, rsu_map)
                if closest_rsu:
                    rsu_sequence.append(closest_rsu)
                else:
                    # 如果没有找到最近的RSU,使用无连接标记
                    rsu_sequence.append(NO_RSU_MARKER)

            # 如果rsu_sequence长度大于1,添加到all_rsu_sequences
            if len(rsu_sequence) > 1:
                all_rsu_sequences.append(rsu_sequence)
    else:
        # 原有模式:去重RSU序列
        for vehicle_coords in vehicle_coord_trajectories:
            # vehicle_coords形状为(时间步数,2)
            rsu_sequence = []
            for coord_pair in vehicle_coords:
                if not isinstance(coord_pair, np.ndarray):
                    coord_pair = np.array(coord_pair)
                if np.isnan(coord_pair).any():
                    continue

                closest_rsu = get_closest_rsu(coord_pair, rsu_map)
                if closest_rsu:
                    if not rsu_sequence or rsu_sequence[-1] != closest_rsu:
                        # 如果rsu_sequence为空或最后一个元素不等于closest_rsu,
                        # 则添加closest_rsu
                        rsu_sequence.append(closest_rsu)
            # 如果rsu_sequence长度大于1,添加到all_rsu_sequences
            if len(rsu_sequence) > 1:
                all_rsu_sequences.append(rsu_sequence)

    # 将all_rsu_sequences转换为numpy数组
    all_rsu_sequences = np.array(all_rsu_sequences, dtype=object)

    # 保存转换后的数据
    if transformed_data_path:
        print(f"保存转换后的RSU序列到{transformed_data_path}")
        np.save(transformed_data_path, all_rsu_sequences)

    print(f"已加载并转换{len(all_rsu_sequences)}个RSU序列。")
    print(f"RSU序列形状: {all_rsu_sequences.shape}")
    sequence_lengths = [len(seq) for seq in all_rsu_sequences]
    print(f"RSU序列形状: {len(all_rsu_sequences)}个序列,长度范围从{min(sequence_lengths) if sequence_lengths else 0}到{max(sequence_lengths) if sequence_lengths else 0}")
    print(
        f"平均序列长度: {sum(sequence_lengths)/len(sequence_lengths) if sequence_lengths else 0:.2f}")

    if all_rsu_sequences.size > 0:
        print(f"RSU序列示例: {all_rsu_sequences[0][:10]}")
    return all_rsu_sequences


def validate_timeslot_sequence(rsu_sequence):
    """
    验证时序RSU序列的有效性
    Args:
        rsu_sequence (list or np.ndarray): RSU序列
    Returns:
        bool: 序列是否有效
    """
    # 处理numpy数组
    if isinstance(rsu_sequence, np.ndarray):
        rsu_sequence = rsu_sequence.tolist()

    if not rsu_sequence or len(rsu_sequence) < 2:
        return False

    # 检查是否包含有效的RSU ID或NO_RSU_MARKER
    valid_markers = list(RSU_COORDS.keys()) + [NO_RSU_MARKER]
    for rsu_id in rsu_sequence:
        if rsu_id not in valid_markers:
            return False

    return True


def get_sequence_statistics(all_sequences):
    """
    获取时序序列的统计信息
    Args:
        all_sequences (list): 所有RSU序列
    Returns:
        dict: 统计信息
    """
    if not all_sequences:
        return {}

    stats = {
        'total_sequences': len(all_sequences),
        'min_length': min(len(seq) for seq in all_sequences),
        'max_length': max(len(seq) for seq in all_sequences),
        'avg_length': sum(len(seq) for seq in all_sequences) / len(all_sequences),
        'no_rsu_count': 0,
        'rsu_frequency': {}
    }

    # 统计NO_RSU出现次数和各RSU频率
    for sequence in all_sequences:
        for rsu_id in sequence:
            if rsu_id == NO_RSU_MARKER:
                stats['no_rsu_count'] += 1
            else:
                stats['rsu_frequency'][rsu_id] = stats['rsu_frequency'].get(
                    rsu_id, 0) + 1

    return stats


def save_timeslot_data(all_sequences, filepath):
    """
    保存时序RSU数据到指定路径
    Args:
        all_sequences (list): 所有时序RSU序列
        filepath (str): 保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存数据
        np.save(filepath, np.array(all_sequences, dtype=object))
        print(f"Timeslot data saved to {filepath}")

        # 保存元数据
        metadata = {
            'total_sequences': len(all_sequences),
            'use_timeslot_mode': True,
            'no_rsu_marker': NO_RSU_MARKER,
            'rsu_coords_keys': list(RSU_COORDS.keys())
        }

        metadata_path = filepath.replace('.npy', '_metadata.npy')
        np.save(metadata_path, metadata)
        print(f"Metadata saved to {metadata_path}")

    except Exception as e:
        print(f"Error saving timeslot data: {e}")


def load_timeslot_data(filepath):
    """
    从指定路径加载时序RSU数据
    Args:
        filepath (str): 数据文件路径
    Returns:
        tuple: (sequences, metadata) 或 (None, None) 如果加载失败
    """
    try:
        # 加载数据
        sequences = np.load(filepath, allow_pickle=True)

        # 尝试加载元数据
        metadata_path = filepath.replace('.npy', '_metadata.npy')
        metadata = None
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()

        print(f"Timeslot data loaded from {filepath}")
        print(f"Loaded {len(sequences)} sequences")

        return sequences, metadata

    except Exception as e:
        print(f"Error loading timeslot data: {e}")
        return None, None
