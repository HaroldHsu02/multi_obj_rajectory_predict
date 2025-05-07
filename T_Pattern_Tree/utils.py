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
    np.random.shuffle(all_trajectories)  # Shuffle for random split
    split_idx = int(len(all_trajectories) * train_ratio)
    train_set = all_trajectories[:split_idx]
    test_set = all_trajectories[split_idx:]
    print(
        f"Data split: {len(train_set)} training sequences, {len(test_set)} testing sequences.")
    return train_set, test_set
