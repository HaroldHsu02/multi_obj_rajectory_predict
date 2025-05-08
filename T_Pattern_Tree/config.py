import numpy as np
import os

# 获取项目根目录的绝对路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# RSU坐标配置
RSU_COORDS = {
    "RSU_0": np.array([375.0, 375.0]),
    "RSU_1": np.array([1125.0, 375.0]),
    "RSU_2": np.array([375.0, 1125.0]),
    "RSU_3": np.array([1125.0, 1125.0]),
}

# 数据文件路径
DATA_FILE_PATH = os.path.join(
    BASE_DIR, "Utils", "Datasets", "Datasets", "rome_trajectory.npy")

# 训练集比例
TRAIN_RATIO = 0.8
