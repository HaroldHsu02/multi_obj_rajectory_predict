from Environments.config import CELLULAR_LOCATIONS
import os
import sys
import numpy as np

# 添加项目根目录到Python路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# 从Environments导入配置

# RSU坐标配置
RSU_COORDS = {
    f"RSU_{i}": np.array(loc) for i, loc in enumerate(CELLULAR_LOCATIONS)
}

# 数据文件路径
DATA_FILE_PATH = os.path.join(
    BASE_DIR,  "Datasets", "Datasets", "rome_trajectory.npy")

# 转换后的RSU序列文件路径
TRANSFORMED_DATA_PATH = os.path.join(
    BASE_DIR, "Utils", "Datasets", "Datasets", "rsu_sequences.npy")

# 训练集比例
TRAIN_RATIO = 0.8
