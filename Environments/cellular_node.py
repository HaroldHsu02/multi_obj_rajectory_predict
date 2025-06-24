'''
Author: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
Date: 2025-04-11 15:45:20
LastEditors: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
LastEditTime: 2025-06-24 11:25:15
FilePath: \multi_obj\Environments\cellular_node.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from config import *


class cellular_node:
    """边缘节点设置"""

    def __init__(self, cellular_index):
        self.cellular_index = cellular_index  # 基站索引
        self.radius = CELLULAR_RADIUS  # 蜂窝网络半径
        self.capability = CELLULAR_CAPABILITY  # 计算能力
        self.bandwidth = CELLULAR_BANDWIDTH  # 带宽
        self.backhaul_network = CELLULAR_BACKHAUL_NETWORK  # 回程链路带宽
        self.channel_gain = CELLULAR_CHANNEL_GAIN  # 距离1m时的参考信道增益
        self.gaussian_noise = CELLULAR_GAUSSIAN_NOISE  # 信道中的高斯白噪声
        self.cellular_loc_array = CELLULAR_LOCATIONS  # 蜂窝基站位置坐标数组
        self.cellular_loc = self.get_cellular_loc()  # 根据基站索引获取物理坐标
        self.server_app = 0  # 初始化服务应用数量为 0

    def get_cellular_loc(self):
        """
        根据蜂窝节点索引返回对应的物理坐标。
        """
        return self.cellular_loc_array[self.cellular_index]

    def compute_channel_rate(self, vehicle_transmit_power, vehicle_loc):
        """
        根据车辆（TU）的发射功率和位置，计算车辆与蜂窝节点之间的无线信道传输速率。

        参数：
            vehicle_transmit_power: 车辆发射功率（单位：W）
            vehicle_loc: 车辆当前位置，[x, y] 坐标（单位：m）

        返回：
            信道传输速率（单位：bps）

        公式：
            R = bandwidth * log2(1 + (P_vehicle * G_eff) / gaussian_noise)
        其中：
            G_eff = channel_gain / d², d 为车辆与基站之间的距离。
        """
        # 计算车辆与基站之间的欧氏距离
        distance = np.linalg.norm(
            np.array(vehicle_loc) - np.array(self.cellular_loc))
        # 为避免距离过小导致异常，将最小距离限制为 1 m
        if distance < 1.0:
            distance = 1.0
        # 计算有效信道增益（自由空间路径衰减模型，路径损耗指数 n=2）
        G_eff = self.channel_gain / (distance ** 2)
        # 计算信噪比（SNR）
        snr = (vehicle_transmit_power * G_eff) / self.gaussian_noise
        # 计算传输速率（bps）
        rate = self.bandwidth * np.log2(1 + snr)
        return rate
