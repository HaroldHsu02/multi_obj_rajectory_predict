from Environments.config import *
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)


class application:
    """应用参数的设置"""

    def __init__(self, task_probability=None):
        """
        task_probability: 任务生成概率，即每个时隙生成任务的概率
        """
        # 初始化应用程序的任务和服务实例
        if task_probability is None:
            task_probability = TASK_GENERATION_PROBABILITY
        self.task_probability = task_probability  # 任务生成概率
        self.task = self.generate_task()  # 任务数据量和计算密度
        self.instance = self.generate_instance()  # 服务实例数据大小

    def generate_task(self):
        """
        生成任务的输入数据大小和计算密度
        根据task_probability决定是否生成任务
        """
        # 根据概率决定是否生成任务
        if np.random.random() < self.task_probability:
            # 生成任务
            return [np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1]),
                    np.random.randint(TASK_DENSITY_RANGE[0], TASK_DENSITY_RANGE[1])]
        else:
            # 不生成任务，返回None或空值
            return None

    def generate_instance(self):
        """生成服务实例的大小"""
        return np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1] * INSTANCE_SIZE_MULTIPLIER)

    def instance_change(self):
        """服务实例大小变化"""
        instance_change_value = np.random.randint(
            INSTANCE_CHANGE_RANGE[0], INSTANCE_CHANGE_RANGE[1])
        self.instance = max(
            self.instance + instance_change_value, TASK_SIZE_RANGE[0])

    def has_task(self):
        """
        检查当前应用是否有任务
        Returns:
            bool: 是否有任务
        """
        return self.task is not None

    def get_task_size(self):
        """
        获取任务数据量
        Returns:
            int: 任务数据量，如果没有任务则返回0
        """
        if self.has_task():
            return self.task[0]
        return 0

    def get_task_density(self):
        """
        获取任务计算密度
        Returns:
            int: 任务计算密度，如果没有任务则返回0
        """
        if self.has_task():
            return self.task[1]
        return 0

    def get_task_cycles(self):
        """
        获取任务所需CPU周期数
        Returns:
            int: CPU周期数，如果没有任务则返回0
        """
        if self.has_task():
            return self.task[0] * self.task[1]
        return 0
