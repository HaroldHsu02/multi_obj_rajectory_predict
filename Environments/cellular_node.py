import numpy as np


class cellular_node:
    """边缘节点设置"""
    radius = 500  # 蜂窝网络的半径，单位m，整个蜂窝覆盖1km2的面积
    capability = 60e9  # 计算能力，单位GHz=20e9Hz
    bandwidth = 20e6  # 带宽，单位MHz =10e6Hz
    backhaul_network = 500e6  # 回程链路带宽500Mbps=500e6 bps,这里的M指的是6次方
    channel_gain = 1e-5  # 距离1m时的参考信道增益，10的-5次方
    gaussian_noise = 1e-13  # 信道中的高斯白噪声，设置参考现有工作，10的-13次方
    # 16个蜂窝基站的二维坐标数组，形成4×4网格布局
    cellular_loc_array = np.array([[500, 500], [1500, 500], [2500, 500], [3500, 500],
                                   [500, 1500], [1500, 1500], [2500, 1500], [3500, 1500],
                                   [500, 2500], [1500, 2500], [2500, 2500], [3500, 2500],
                                   [500, 3500], [1500, 3500], [2500, 3500], [3500, 3500]])

    # cellular_loc_path = 'C:\\Users\\16923\\Desktop\\new_migration\\Migration\\data\\cellular.npy'

    def __init__(self, cellular_index):
        self.cellular_index = cellular_index  # 基站索引
        self.loc = self.get_cellular_loc()  # 根据基站索引获取物理坐标
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
        distance = np.linalg.norm(np.array(vehicle_loc) - np.array(self.loc))
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
