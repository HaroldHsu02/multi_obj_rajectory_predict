'''
Author: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
Date: 2025-05-12 19:47:30
LastEditors: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
LastEditTime: 2025-06-27 17:32:57
FilePath: \multi_obj\Environments\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os

# 获取项目根目录的绝对路径
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# ============== 环境参数 ==============
# 场地参数
GROUND_LENGTH = 2000  # 场地长度（米）
GROUND_WIDTH = 2000   # 场地宽度（米）

# 模拟参数
MAX_TIME_SLOT = 240   # 最大时隙数
RANDOM_SEED = 1       # 随机种子

# ============== 蜂窝基站参数 ==============
CELLULAR_NUMBER = 4  # 蜂窝基站数目
CELLULAR_RADIUS = 1000  # 蜂窝网络半径（米）
CELLULAR_CAPABILITY = 60e9  # 计算能力（Hz）
CELLULAR_BANDWIDTH = 10e6  # 带宽（Hz）
CELLULAR_BACKHAUL_NETWORK = 500e6  # 回程链路带宽（bps）
CELLULAR_CHANNEL_GAIN = 1e-5  # 距离1m时的参考信道增益
CELLULAR_GAUSSIAN_NOISE = 1e-13  # 信道中的高斯白噪声

# 蜂窝基站位置坐标（形成2*2网格布局）
CELLULAR_LOCATIONS = np.array([
    [500, 500], [1500, 500],
    [500, 1500], [1500, 1500]
])

# ============== 车辆参数 ==============
VEHICLE_NUMBER = 60  # 车辆数目
VEHICLE_POWER_RANGE = [0.4, 0.6]  # 车辆发射功率范围（W）
VEHICLE_CPU_POWER_RANGE = [2.0, 6.0]  # 车辆CPU能力范围（GHz）
VEHICLE_ETA = 1e-26  # 芯片能耗常数（用于能耗计算）

# 车辆轨迹数据路径
VEHICLE_TRAJECTORY_PATH = os.path.join(
    BASE_DIR, 'Utils', 'Datasets', 'Datasets', 'rome_trajectory_150_400.npy')

# ============== 应用参数 ==============
# 任务参数
TASK_SIZE_RANGE = [4194304, 12582912]  # 任务数据量范围 [4MB, 12MB]
TASK_DENSITY_RANGE = [200, 1000]  # 计算密度范围 [200, 1000]
# TASK_OUTPUT_SIZE_RANGE = [500 * 1024, 800 * 1024]  # 输出数据量范围 [500KB, 800KB]

# 任务生成概率
TASK_GENERATION_PROBABILITY = 0.7  # 每个时隙生成任务的概率

# 服务实例参数
INSTANCE_SIZE_MULTIPLIER = 20  # 服务实例大小相对于任务大小的倍数
INSTANCE_CHANGE_RANGE = [-512000, 512000]  # 服务实例大小变化范围

# ============== 时间参数 ==============
BACKHAUL_ONE_HOP = 0.3  # 任务传输一跳的时间

# ============== 状态空间参数 ==============
LOC_PAGE = 3  # 每个时隙采样的轨迹点间隔,用于控制轨迹数据采样频率
LOC_INIT = 10  # 轨迹起始位置的偏移量,用于跳过轨迹前面的一些数据点
