import numpy as np

# RSU坐标配置
RSU_COORDS = {
    "RSU_0": np.array([375.0, 375.0]),
    "RSU_1": np.array([1125.0, 375.0]),
    "RSU_2": np.array([375.0, 1125.0]),
    "RSU_3": np.array([1125.0, 1125.0]),
}

# 数据文件路径
DATA_FILE_PATH = "/Datasets/Datasets/rome_trajectory_100_400.npy"

# 训练集比例
TRAIN_RATIO = 0.8
