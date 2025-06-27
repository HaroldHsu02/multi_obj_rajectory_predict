import numpy as np
import os
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])


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


def split_data(all_trajectories, train_ratio=0.8):
    """Splits trajectories into training and testing sets."""
    np.random.seed(1)
    np.random.shuffle(all_trajectories)  # Shuffle for random split
    split_idx = int(len(all_trajectories) * train_ratio)
    train_set = all_trajectories[:split_idx]
    test_set = all_trajectories[split_idx:]
    print(
        f"Data split: {len(train_set)} training sequences, {len(test_set)} testing sequences.")
    return train_set, test_set


def create_sliding_windows(sequence, window_size, step_size=1):
    """
    从序列中创建滑动窗口
    Args:
        sequence (list): 输入序列
        window_size (int): 窗口大小
        step_size (int): 步长
    Returns:
        list: 滑动窗口列表
    """
    if len(sequence) < window_size:
        return [sequence] if sequence else []
    
    windows = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        windows.append(sequence[i:i + window_size])
    
    return windows


def validate_sliding_window(window, min_size=1):
    """
    验证滑动窗口的有效性
    Args:
        window (list): 滑动窗口
        min_size (int): 最小窗口大小
    Returns:
        bool: 窗口是否有效
    """
    if not window or len(window) < min_size:
        return False
    
    # 检查窗口中的元素是否都是有效的RSU ID或NO_RSU_MARKER
    from T_Pattern_Tree.config import RSU_COORDS, NO_RSU_MARKER
    valid_markers = list(RSU_COORDS.keys()) + [NO_RSU_MARKER]
    
    for item in window:
        if item not in valid_markers:
            return False
    
    return True


def calculate_prediction_confidence(predicted_rsu, candidate_predictions):
    """
    计算预测结果的置信度
    Args:
        predicted_rsu (str): 预测的RSU ID
        candidate_predictions (dict): 候选预测及其支持度
    Returns:
        float: 置信度分数 (0-1)
    """
    if not candidate_predictions or predicted_rsu not in candidate_predictions:
        return 0.0
    
    total_support = sum(candidate_predictions.values())
    predicted_support = candidate_predictions[predicted_rsu]
    
    return predicted_support / total_support if total_support > 0 else 0.0
