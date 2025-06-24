import numpy as np
from config import *


class application:
    """应用参数的设置"""

    task_size = [4194304, 12582912]  # 定义任务数据量范围 [4MB, 12MB]
    task_density = [200, 1000]  # 定义计算密度范围 [200, 1000]
    output_size_range = [500 * 1024, 800 * 1024]  # 定义输出数据量范围 [500KB, 800KB]

    def __init__(self, n_tasks=5):
        """
        n_tasks: 任务数量，即应用中包含的任务数
        """
        # self.task_belong_cellular = -1  # 初始化蜂窝网络编号
        # 生成任务列表，每个任务为一个字典，包含任务参数
        self.tasks = self.generate_tasks(n_tasks)
        # 生成基于任务列表的任务依赖DAG，返回一个字典：
        #   {task_id: {'predecessors': [...], 'successors': [...]} }
        self.dag = self.generate_dag(self.tasks)

        # 初始化应用程序的任务和服务实例
        self.task = self.generate_task()  # 任务数据量和计算密度
        self.instance = self.generate_instance()  # 服务实例数据大小
        self.instance_belong_cellular = -1  # 初始化服务实例所在的蜂窝网络编号

    def generate_task(self):
        """生成任务的输入数据大小和计算密度"""
        return [np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1]),
                np.random.randint(TASK_DENSITY_RANGE[0], TASK_DENSITY_RANGE[1])]

    def generate_instance(self):
        """生成服务实例的大小"""
        return np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1] * 20)

    def instance_change(self):
        """服务实例大小变化"""
        instance_change_value = np.random.randint(-512000, 512000)
        self.instance = max(
            self.instance + instance_change_value, TASK_SIZE_RANGE[0])

    def generate_tasks(self, n_tasks):
        """
        生成 n_tasks 个任务，每个任务用字典表示，包含：
            - id: 任务编号
            - i: 输入数据量（字节）
            - density: 计算密度（CPU周期/字节）
            - numberOfCycles: 任务所需 CPU 周期数（i * density）
            - o: 输出数据量（字节）
            - location: 任务执行位置（服务器编号），初始为 -1，表示未分配或本地执行
            - finished: 任务是否完成的标志，初始为 False
        返回任务列表。
        """
        tasks = []
        for idx in range(n_tasks):
            # 随机生成输入数据量
            input_size = np.int64(np.random.randint(
                TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1]))
            # 随机生成计算密度
            density = np.int64(np.random.randint(
                TASK_DENSITY_RANGE[0], TASK_DENSITY_RANGE[1]))
            # 计算 CPU 周期数
            cpuCycles = input_size * density
            # 随机生成输出数据量，范围较小
            output_size = np.int64(np.random.randint(
                TASK_OUTPUT_SIZE_RANGE[0], TASK_OUTPUT_SIZE_RANGE[1]))
            task = {
                'id': idx,
                'i': input_size,            # 输入数据量
                'density': density,         # 计算密度
                'numberOfCycles': cpuCycles,  # 需要的CPU周期数
                'o': output_size,           # 输出数据量
                'location': -1,             # 任务执行位置（初始为 -1，后续可设置为具体服务器编号）
                'finished': False           # 任务是否完成的指示变量
            }
            tasks.append(task)
        return tasks

    @staticmethod
    def generate_dag(tasks):
        """
        根据任务列表生成任务间的依赖关系（DAG）。
        假设任务的自然编号反映了执行顺序（编号较小的任务先执行），
        对于每个任务（除第一个外），至少随机选取一个前置任务作为依赖，
        然后以一定概率（例如 0.3）额外添加其它前置任务（保证无环依赖）。
        返回字典，格式为：
            { task_id: {'predecessors': [id,...], 'successors': [id,...] } }
        """
        n_tasks = len(tasks)
        dag = {task['id']: {'predecessors': [], 'successors': []}
               for task in tasks}
        # 对于除第一个任务之外的每个任务
        for j in range(1, n_tasks):
            # 保证至少有一个前置任务：从 0 到 j-1 随机选一个
            pred = np.random.randint(0, j)
            dag[j]['predecessors'].append(pred)
            dag[pred]['successors'].append(j)
            # 对于前面的其他任务也以一定概率添加为前置任务
            for i in range(0, j):
                if i == pred:
                    continue
                # 以0.3的概率添加任务 i 为任务 j 的前置任务（根据实际需求可调整概率）
                if np.random.rand() < 0.3:
                    dag[j]['predecessors'].append(i)
                    dag[i]['successors'].append(j)
        return dag
