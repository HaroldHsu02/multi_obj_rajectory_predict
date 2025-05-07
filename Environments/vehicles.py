import numpy as np
from Environments.application import application


class vehicle:
    def __init__(self, vehicle_index):
        self.P = round(np.random.uniform(0.4, 0.6), 4)  # 车辆的发射功率
        self.vehicle_index = vehicle_index  # 车辆的索引
        self.application = application()# 初始化 TU 对应的应用
        self.loc_data_path = './Utils/Datasets/Datasets/rome_trajectory_150_400.npy'  # 所有用户的轨迹
        self.label_data = self.get_trajectory()
        # 为 TU 设置计算资源（单位 GHz），按照论文建议的范围 [2, 6]
        self.cpu_power = np.random.uniform(2.0, 6.0)
        # 芯片相关常数（用于能耗计算），例如论文中 η 可取 1e-26
        self.eta = 1e-26

        # 可选：存储 TU 当前任务执行状态或累计统计数据
        self.total_delay_local = 0.0
        self.total_energy_local = 0.0
    def get_trajectory(self):
        return np.load(self.loc_data_path)[:, self.vehicle_index]

    def get_loc(self, time_slot):
        return self.label_data[time_slot]

    def compute_local_task(self, task):
        """
        模拟 TU 本地执行任务。
        参数：
            task: 字典类型，包含以下键：
                'i': 输入数据量（字节）
                'density': 计算密度（单位：CPU周期/字节）
                'u': 任务需要的 CPU 周期数 (通常 u = i * density)
                'o': 输出数据量（字节）
        返回：
            local_delay: 本地执行延时，根据公式 delay = u / cpu_power  （公式 (7)）
            local_energy: 本地执行能耗，根据公式 energy = η * u * (cpu_power)^2  （公式 (8)）
        """
        u = task['numberOfCycles']
        local_delay = u / self.cpu_power
        local_energy = self.eta * u * (self.cpu_power ** 2)
        # 更新累计统计信息
        self.total_delay_local += local_delay
        self.total_energy_local += local_energy
        return local_delay, local_energy

    def offload_task(self, task, channel_rate):
        """
        模拟任务卸载到 UAV 端执行（简化模型），计算任务卸载过程的延时和能耗。
        卸载过程包括三个部分：
            1. 数据上传延时与能耗：上行传输时间 = 输入数据量 / channel_rate，能耗 = 车辆发射功率 * 上传时间
            2. UAV 边缘服务器处理延时：处理延时 = u / edge_cpu_power，
               其中 edge_cpu_power 假定足够大，本例中可以采用一个常数（例如 10 GHz）；
            3. 下行传输延时：下载时间 = 输出数据量 / channel_rate
        参数：
            task: 同 compute_local_task 中的任务参数字典。
            channel_rate: 信道上行/下行传输速率（单位：字节/秒），实际应根据无线信道模型计算。

        返回：
            offload_delay: 总延时 = 上传时间 + UAV 执行时间 + 下行时间
            offload_energy: 总能耗 = 上传能耗（本例中仅考虑 TU 上传能耗）
        """
        # 上行传输：上传时间
        upload_time = task['i'] / channel_rate
        upload_energy = self.P * upload_time

        # UAV 端执行延时（假设 UAV 边缘服务器的计算能力较高，这里采用常数模拟，例如 10 GHz）
        edge_cpu_power = 10.0  # 假设值，单位 GHz
        execution_time = task['u'] / edge_cpu_power

        # 下行传输：下载时间（采用与上行相同的速率）
        download_time = task['o'] / channel_rate

        offload_delay = upload_time + execution_time + download_time
        offload_energy = upload_energy  # TU 端仅计算上传消耗能耗

        return offload_delay, offload_energy

