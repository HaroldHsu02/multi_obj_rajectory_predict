import numpy as np
from Utils.File_operation import FileOperation
result = np.load('rome_trajectory.npy')[:, 20:400]
filepath = FileOperation.get_BASE_DIR() + "/Datasets/Datasets" + "/" + "rome_trajectory_20_400"  # 获取存储路径
print(filepath)
np.save(filepath,result)
print(result)