'''
Author: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
Date: 2025-04-11 15:47:39
LastEditors: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
LastEditTime: 2025-06-27 19:32:49
FilePath: \multi_obj\Environments\vehicles.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from Environments.application import application
from Environments.config import *


class vehicle:
    def __init__(self, vehicle_index):
        """初始化车辆对象

        根据配置的车辆参数初始化车辆对象,包括:
        - 发射功率(P): 随机生成在VEHICLE_POWER_RANGE范围内
        - 车辆索引(vehicle_index): 传入的车辆编号 
        - 应用(application): 初始化一个application对象
        - CPU能力: 随机生成在VEHICLE_CPU_POWER_RANGE范围内,单位GHz
        - 芯片能耗常数: 用于计算能耗,取值为VEHICLE_ETA
        """
        # 初始化发射功率，在VEHICLE_POWER_RANGE范围内随机生成4个
        self.P = round(np.random.uniform(
            VEHICLE_POWER_RANGE[0], VEHICLE_POWER_RANGE[1]), 4)

        # 初始化基本属性
        self.vehicle_index = vehicle_index
        self.application = application()
        # 轨迹数据集路径，读取VEHICLE_TRAJECTORY_PATH
        self.loc_data_path = VEHICLE_TRAJECTORY_PATH
        # 读取轨迹数据label_data格式为
        self.label_data = self.get_trajectory()

        # 初始化计算资源
        # 车辆CPU能力
        self.cpu_power = np.random.uniform(
            VEHICLE_CPU_POWER_RANGE[0], VEHICLE_CPU_POWER_RANGE[1])
        # # 芯片能耗常数
        self.eta = VEHICLE_ETA

        # 初始化统计数据，累计延时和累计能耗
        self.total_delay_local = 0.0
        self.total_energy_local = 0.0

    def get_trajectory(self):
        """获取车辆轨迹数据

        从轨迹数据集文件中加载当前车辆的轨迹数据。
        轨迹数据集文件包含所有车辆的轨迹,每一列对应一辆车的轨迹。
        使用self.vehicle_index作为列索引获取当前车辆的轨迹。

        语法解释:
        - np.load(): NumPy提供的加载.npy文件的函数
        - self.loc_data_path: 轨迹数据文件路径
        - [:, self.vehicle_index]: 切片语法,取所有行(:)和第vehicle_index列

        Returns:
            ndarray: 当前车辆的轨迹数据,一维数组,包含车辆在每个时隙的位置
        """
        return np.load(self.loc_data_path)[:, self.vehicle_index]

    def get_loc(self, time_slot):
        """获取车辆在指定时隙的位置

        根据传入的时隙索引,从车辆轨迹数据中获取该时隙的车辆位置坐标。

        Args:
            time_slot: 时隙索引,整数类型
        """
        return self.label_data[time_slot]

    def compute_local_task(self):
        """
        模拟 TU 本地执行任务。
        使用application中的当前任务进行计算。
        返回：
            local_delay: 本地执行延时，根据公式 delay = u / cpu_power  （公式 (7)）
            local_energy: 本地执行能耗，根据公式 energy = η * u * (cpu_power)^2  （公式 (8)）
        """
        # 检查是否有任务
        if not self.application.has_task():
            return 0.0, 0.0
        
        # 获取任务所需CPU周期数
        u = self.application.get_task_cycles()
        local_delay = u / self.cpu_power
        local_energy = self.eta * u * (self.cpu_power ** 2)
        # 更新累计统计信息
        self.total_delay_local += local_delay
        self.total_energy_local += local_energy
        return local_delay, local_energy

    def offload_task(self, channel_rate):
        """
        模拟任务卸载到 RSU 端执行，计算任务卸载过程的延时和能耗。
        卸载过程包括三个部分：
            1. 数据上传延时与能耗：上行传输时间 = 输入数据量 / channel_rate，能耗 = 车辆发射功率 * 上传时间
            2. RSU 边缘服务器处理延时：处理延时 = u / edge_cpu_power，
               其中 edge_cpu_power 假定足够大，本例中可以采用一个常数（例如 10 GHz）；
            3. 下行传输延时：下载时间 = 输出数据量 / channel_rate
        参数：
            channel_rate: 信道上行/下行传输速率（单位：字节/秒），实际应根据无线信道模型计算。

        返回：
            offload_delay: 总延时 = 上传时间 + RSU 执行时间 + 下行时间
            offload_energy: 总能耗 = 上传能耗（本例中仅考虑 TU 上传能耗）
        """
        # 检查是否有任务
        if not self.application.has_task():
            return 0.0, 0.0
        
        # 获取任务参数
        input_size = self.application.get_task_size()
        cpu_cycles = self.application.get_task_cycles()
        
        # 假设输出数据量为输入数据量的10%
        output_size = int(input_size * 0.1)
        
        # 上行传输：上传时间
        upload_time = input_size / channel_rate
        upload_energy = self.P * upload_time

        # RSU 端执行延时（假设 RSU 边缘服务器的计算能力较高，这里采用常数模拟，例如 10 GHz）
        edge_cpu_power = 10.0  # 假设值，单位 GHz
        execution_time = cpu_cycles / edge_cpu_power

        # 下行传输：下载时间（采用与上行相同的速率）
        download_time = output_size / channel_rate

        offload_delay = upload_time + execution_time + download_time
        offload_energy = upload_energy  # TU 端仅计算上传消耗能耗

        return offload_delay, offload_energy
