import math
import os
import random
import torch
import numpy as np
from scipy.spatial import KDTree
from cellular_node import cellular_node
from vehicles import vehicle
from config import *


def seed_everything(seed):
    """
    设置随机种子以确保实验结果可复现

    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Env:
    """
    车联网环境类，模拟车辆和基站之间的交互

    属性:
        Ground_Length (int): 场地长度(米)
        Ground_Width (int): 场地宽度(米) 
        cellular_number (int): 蜂窝基站数量
        vehicle_number (int): 车辆数量
        max_time_slot (int): 最大时隙数
        migration_one_hop (float): 服务实例迁移一跳的时间
        backhaul_one_hop (float): 任务传输一跳的时间
        migration_prepare_time (float): 在目标服务器准备环境的时间
        cellular_list (list): 蜂窝基站对象列表
        vehicle_list (list): 车辆对象列表
        cellular_locations (ndarray): 所有基站的位置坐标
        kdtree (KDTree): 用于快速查找最近基站的KD树
        loc_page (int): 位置偏移量，用于轨迹数据采样
        loc_init (int): 初始位置偏移
        state (ndarray): 当前环境状态
        time_slot (int): 当前时隙
        cellular_computation_load (list): 各基站当前时隙的计算负载
        cellular_communication_load (list): 各基站当前时隙的通信负载
        vehicle_this_belong (list): 记录每个车辆当前所属的基站编号
        vehicle_distance_this_belong (list): 记录每个车辆到其所属基站的距离
    """

    def __init__(self):
        # 设置随机种子，保证实验可复现
        seed_everything(RANDOM_SEED)

        # 初始化环境基本参数
        self.Ground_Length = GROUND_LENGTH
        self.Ground_Width = GROUND_WIDTH
        self.cellular_number = CELLULAR_NUMBER
        self.vehicle_number = VEHICLE_NUMBER
        self.max_time_slot = MAX_TIME_SLOT

        # 初始化迁移相关参数
        self.migration_one_hop = MIGRATION_ONE_HOP
        self.backhaul_one_hop = BACKHAUL_ONE_HOP
        self.migration_prepare_time = MIGRATION_PREPARE_TIME

        # 初始化基站和车辆
        self.cellular_list = self.get_cellulars()
        self.vehicle_list = self.get_vehicles()

        # 构建KD树用于快速查找最近基站
        # 将基站位置列表转换为numpy数组，用于构建KD树
        # 使用列表推导式从cellular_list中提取每个基站的位置坐标
        self.cellular_locations = np.array(
            [cel.cellular_loc for cel in self.cellular_list])

        # 使用基站位置坐标构建KD树，用于快速查找最近邻基站
        # KDTree是一种空间数据结构，可以高效地进行最近邻搜索
        self.kdtree = KDTree(self.cellular_locations)

        # 初始化位置相关参数
        self.loc_page = LOC_PAGE
        self.loc_init = LOC_INIT

        # 初始化状态和时隙
        self.state = None
        self.time_slot = 0

        # 初始化负载统计数组
        # 计算负载
        self.cellular_computation_load = [0] * self.cellular_number
        # 通信负载
        self.cellular_communication_load = [0] * self.cellular_number
        # 车辆所属基站
        self.vehicle_this_belong = [-1] * self.vehicle_number
        # 车辆到基站距离
        self.vehicle_distance_this_belong = [0] * self.vehicle_number

    def get_cellulars(self):
        """
        获取蜂窝基站列表

        Returns:
            np.array: 包含所有蜂窝基站对象的numpy数组
        """
        return np.array([cellular_node(cel) for cel in range(self.cellular_number)])

    def get_vehicles(self):
        """
        获取车辆列表

        根据配置的车辆数量(self.vehicle_number)生成对应数量的车辆对象。
        每个车辆对象通过vehicle类初始化，传入车辆编号作为标识。

        Returns:
            np.array: 包含所有车辆对象的numpy数组
        """
        return np.array([vehicle(vec) for vec in range(self.vehicle_number)])

    def get_state(self):
        """
        获取环境当前状态

        对每个车辆收集以下状态信息:
        1. 车辆当前位置坐标(x,y) - 从车辆轨迹数据中获取
        2. 任务相关参数:
           - 任务数据量(字节)
           - 任务计算密度(每字节所需CPU周期数) 
           - 服务实例大小(字节)
           - 服务实例所在基站编号

        Returns:
            np.array: 包含所有车辆状态信息的状态向量
        """
        state = []
        for vec in self.vehicle_list:
            # 获取车辆当前位置坐标
            # 根据初始位置(loc_init)、当前时隙(time_slot)和每页位置数(loc_page)计算当前位置索引
            state.append(
                vec.label_data[self.loc_init + self.time_slot * self.loc_page][0])  # 车辆坐标 X轴
            state.append(
                vec.label_data[self.loc_init + self.time_slot * self.loc_page][1])  # 车辆坐标 Y轴

            # 获取任务相关参数
            state.append(vec.application.task[0])  # 任务数据量
            state.append(vec.application.task[1])  # 任务计算密度
            state.append(vec.application.instance)  # 服务实例数据大小
            # 服务实例所在基站编号
            state.append(vec.application.instance_belong_cellular)

        # 将状态列表转换为numpy数组并保存
        self.state = np.array(state)
        return self.state

    def get_state_normalize(self):
        """
        获取标准化后的环境状态

        对每个车辆的状态信息进行归一化处理，包括:
        1. 车辆位置坐标(x,y) - 除以地图宽度进行归一化到[0,1]区间
        2. 任务数据量 - 线性归一化到[0,1]区间，原始范围[4194304, 12582912]字节
        3. 任务计算密度 - 线性归一化到[0,1]区间，原始范围[200, 1000]每字节所需CPU周期
        4. 服务实例大小 - 线性归一化到[0,1]区间，原始范围[4194304, 419430400]字节
        5. 服务实例所属基站 - 除以基站总数(15)归一化到[0,1]区间

        Returns:
            np.array: 归一化后的状态向量
        """
        state = []
        for vec in self.vehicle_list:
            state.append(vec.label_data[self.loc_init + self.time_slot * self.loc_page][
                0] / self.Ground_Width)  # 归一化X坐标
            state.append(vec.label_data[self.loc_init + self.time_slot * self.loc_page][
                1] / self.Ground_Width)  # 归一化Y坐标
            state.append(
                (vec.application.task[0] - 4194304) / (12582912 - 4194304))  # 归一化任务数据量
            # 归一化任务计算密度
            state.append((vec.application.task[1] - 200) / (1000 - 200))
            state.append((vec.application.instance - 4194304) /
                         (419430400 - 4194304))  # 归一化服务实例大小
            # 归一化所属基站编号
            state.append(vec.application.instance_belong_cellular / 15)
        self.state = np.array(state)
        return self.state

    def reset_environments(self):
        """
        重置环境函数

        Returns:
            np.array: 重置后的环境状态
        """
        self.__init__()  # 重新设置随机数
        return self.get_state()

    def reset_environment_normalize(self):
        """
        重置环境函数（返回标准化状态）

        Returns:
            np.array: 重置后的标准化环境状态
        """
        self.__init__()
        return self.get_state_normalize()

    def belong_cellular(self, location):
        """
        确定车辆属于哪个蜂窝网络

        Args:
            location (tuple): 车辆位置坐标(x, y)

        Returns:
            tuple: (最近基站编号, 到基站的距离)
        """
        distance, nearest = self.kdtree.query(location)
        return nearest, distance  # 返回的是节点编号，以及距离节点的距离

    def manhattan_distance(self, i, j):
        """
        计算两个节点之间的曼哈顿距离

        Args:
            i (int): 旧服务器编号
            j (int): 新服务器编号

        Returns:
            float: 两个服务器之间的跳数距离
        """
        old_location_mec = self.cellular_list[i].cellular_loc
        new_location_mec = self.cellular_list[j].cellular_loc
        return (abs(old_location_mec[0] - new_location_mec[0]) + abs(
            old_location_mec[1] - new_location_mec[1])) / 1000

    def get_this_load(self, time_slot, action):
        """
        根据action获取该时隙所有蜂窝节点的负载：计算负载以及通信负载

        Args:
            time_slot (int): 当前时隙
            action (list): 动作列表，指定每个车辆的目标基站
        """
        self.cellular_computation_load = [0] * self.cellular_number
        self.cellular_communication_load = [0] * self.cellular_number
        self.vehicle_this_belong = [-1] * self.vehicle_number
        self.vehicle_distance_this_belong = [0] * self.vehicle_number

        for vec in self.vehicle_list:
            cellular_index, distance = self.belong_cellular(
                vec.label_data[self.loc_init + time_slot * self.loc_page])
            # 车辆属于哪个基站
            self.vehicle_this_belong[vec.vehicle_index] = cellular_index
            # 车辆与基站之间的距离
            self.vehicle_distance_this_belong[vec.vehicle_index] = distance
            # 基站通信负载+1，按人数均分带宽
            self.cellular_communication_load[cellular_index] += 1

            if action is None:  # 如果action为None，则代表是第一个时隙，此时应该根据车辆的位置统计负载
                self.cellular_computation_load[cellular_index] += np.sqrt(np.int64(
                    np.int64(vec.application.task[0]) * np.int64(
                        vec.application.task[1])))  # 统计负载，到时候先按加权分配
            else:  # action不为空，根据action统计负载
                self.cellular_computation_load[action[vec.vehicle_index]] += np.sqrt(np.int64(
                    np.int64(vec.application.task[0]) * np.int64(
                        vec.application.task[1])))  # 统计负载，到时候先按加权分配

    def communication_time(self, vec, action):
        """
        计算通信时间

        Args:
            vec (vehicle): 车辆对象
            action (list): 动作列表

        Returns:
            tuple: (本地通信时间, 回程传输时间1, 回程传输时间2, 本地基站编号, 跳数)
        """
        # Step 1: 得出本地基站的位置以及二者之间的距离
        local_cellular = self.vehicle_this_belong[vec.vehicle_index]  # 本地基站的编号
        # 直接计算距离，不用开根号（方便后面的计算）
        length = self.vehicle_distance_this_belong[vec.vehicle_index] ** 2

        # Step 2: 算出用户与本地基站之间的传输速率，当前基站给用户分配了多少通信资源
        bandwidth = (1 / self.cellular_communication_load[local_cellular]) * self.cellular_list[
            local_cellular].bandwidth
        channel_gain = abs(
            self.cellular_list[local_cellular].channel_gain / length)
        trans_rate = bandwidth * math.log2(
            1 + vec.P * channel_gain / self.cellular_list[local_cellular].gaussian_noise)

        # Step 3: 用户将任务传输到本地基站
        communication_time1 = vec.application.task[0] / trans_rate

        # Step 4: 任务从本地基站通过回程链路传输到服务基站（前提条件：本地基站与服务基站不是一个基站）
        hop = 0  # 到目标服务器的跳数
        # 代表是其他时隙，需要判断一下，服务实例位于哪个位置，然后转发
        if local_cellular != action[vec.vehicle_index]:
            hop = self.manhattan_distance(
                local_cellular, action[vec.vehicle_index])
            communication_time2_1 = vec.application.task[0] / self.cellular_list[
                local_cellular].backhaul_network
            communication_time2_2 = self.backhaul_one_hop * hop
        else:  # 服务实例就位于本地基站位置，不需要转发
            communication_time2_1 = 0
            communication_time2_2 = 0
            hop = 0

        return round(communication_time1, 8), round(communication_time2_1, 8), round(
            communication_time2_2, 8), local_cellular, hop

    def computation_time(self, vec, action):
        """
        计算计算时间

        Args:
            vec (vehicle): 车辆对象
            action (list): 动作列表

        Returns:
            tuple: (计算时间, 分配的计算能力)
        """
        # Step1: 目标action基站给用户分配了多少计算资源
        compute_vol = np.int64(
            vec.application.task[0]) * np.int64(vec.application.task[1])  # 计算量
        ac = action[vec.vehicle_index]  # 在哪里计算

        capability = (np.sqrt(compute_vol) / self.cellular_computation_load[ac]) * \
            self.cellular_list[ac].capability
        computation_time = compute_vol / capability
        return round(computation_time, 8), capability

    def migration_time(self, vec, action):
        """
        计算迁移时间

        Args:
            vec (vehicle): 车辆对象
            action (list): 动作列表

        Returns:
            tuple: (服务数据从本地基站发出的时间, 服务数据从本地基站经过回程链路传输的时间, 跳数)
        """
        ins_belong = vec.application.instance_belong_cellular  # 迁移之前服务实例所在的MEC服务器编号
        ac = action[vec.vehicle_index]

        hop = self.manhattan_distance(ins_belong, ac)
        migration_time1 = vec.application.instance / \
            self.cellular_list[ins_belong].backhaul_network
        migration_time2 = hop * self.migration_one_hop

        return round(migration_time1, 8), round(migration_time2, 8), hop

    def vehicle_all_consume(self, action):
        """
        计算单辆车完成任务的消耗

        Args:
            action (list): 动作列表，指定每个车辆的目标基站

        Returns:
            tuple: (总奖励, 每辆车的详细结果)
        """
        vec_result = []  # 存储每辆车的结果
        reward = 0  # 奖励

        for vec in self.vehicle_list:  # 以下三个分支，具体只会进入一个
            create_time = 0  # 创建虚拟机的时间，其实可以等同于迁移虚拟机准备的时间（环境准备）
            comm_time = 0  # 通信时间
            comp_time = 0  # 任务在目的基站上计算的时间
            mig_time = 0  # 服务迁移时间
            mig_prepare_time = 0  # 如果迁移目的地没有对应应用程序，需要提前准备环境
            ac = action[vec.vehicle_index]

            if vec.application.instance_belong_cellular == -1:
                # 代表此时是第0个时隙，应该创建服务实例
                mig_time = self.migration_prepare_time

                comm_time1, comm_time2_1, comm_time2_2, local_cellular, comm_hop = (
                    self.communication_time(
                        vec, action))  # 进入通信函数
                comm_time = comm_time1 + comm_time2_1 + comm_time2_2  # 全部通信时间

                comp_time, capability = self.computation_time(
                    vec, action)  # 进入计算函数计算

            elif vec.application.instance_belong_cellular == ac:
                # 服务实例上一时隙所在的服务器与新的动作指向的是同一个服务器，不进行迁移
                comm_time1, comm_time2_1, comm_time2_2, local_cellular, comm_hop = (
                    self.communication_time(
                        vec, action))  # 进入通信函数
                comm_time = comm_time1 + comm_time2_1 + comm_time2_2  # 全部通信时间

                comp_time, capability = self.computation_time(
                    vec, action)  # 进入计算函数计算

            elif vec.application.instance_belong_cellular != ac:
                # 服务实例上一时隙所在的服务器与新的动作指向的不是同一个服务器，应当进行迁移，迁移到action处
                if self.cellular_list[ac].server_app == 0:
                    mig_prepare_time = self.migration_prepare_time
                mig_time1, mig_time2, mig_hop = self.migration_time(
                    vec, action)  # 进入迁移函数
                # 全部迁移时间，取迁移时间和环境准备时间最大的那一个
                mig_time = max(mig_time1 + mig_time2, mig_prepare_time)

                comm_time1, comm_time2_1, comm_time2_2, local_cellular, comm_hop = (
                    self.communication_time(
                        vec, action))  # 进入通信函数
                comm_time = comm_time1 + comm_time2_1 + comm_time2_2  # 全部通信时间

                comp_time, capability = self.computation_time(
                    vec, action)  # 进入计算函数

            all_time = comm_time + comp_time + mig_time  # 所有时间消耗
            vec_result.append([comm_time, comp_time, mig_time, all_time])
            reward -= all_time

        # 修改所有蜂窝网络的服务APP数量
        for vec in self.vehicle_list:
            if vec.application.instance_belong_cellular == action[vec.vehicle_index]:
                pass  # 如果不迁移，就不进行处理
            else:
                # 统一删除先删除原服务器
                if vec.application.instance_belong_cellular == -1:
                    pass  # 最开始那个时隙，不做特殊处理
                else:
                    self.cellular_list[vec.application.instance_belong_cellular].server_app -= 1
                # 在目标处添加
                self.cellular_list[action[vec.vehicle_index]].server_app += 1
        return reward, vec_result

    def step(self, action):
        """
        执行动作函数，严格遵循st，at，rt，st+1的操作序列

        Args:
            action (list): 当前时隙的动作列表

        Returns:
            tuple: (下一状态, 奖励, 详细结果, 是否结束)
        """
        done = False  # 记录epoch是否结束训练

        # Step 1: 根据当前动作，获取时隙t的负载：用户属于哪个蜂窝，每个蜂窝网络的通信负载和节点负载
        self.get_this_load(self.time_slot, action)
        if action is None:
            final_action = self.vehicle_this_belong
        else:
            final_action = action

        # Step 2: 执行收益函数计算
        reward, vec_result = self.vehicle_all_consume(final_action)

        # Step3：判断结束条件（目前的截止条件是当所有的时隙运行完毕）
        self.time_slot += 1
        if self.time_slot == self.max_time_slot:
            done = True
            self.time_slot = 0

        # Step 4：更新下一状态（需要显式调用的只有用户的任务以及服务实例，其余信息，只要我们更新timeslot就可获得）
        for vec in self.vehicle_list:
            vec.application.generate_task()  # 车辆上的智能应用产生新任务
            vec.application.instance_change()  # 智能应用对应的服务实例大小变化
            vec.application.instance_belong_cellular = final_action[
                vec.vehicle_index]  # 车辆的服务所在的边缘服务器
        return self.get_state_normalize(), reward, vec_result, done


# 测试代码（已注释）
# env = Env()
# i = 0
# ep_reward = 0
# print("********************************************************时隙", i,
# "*********************************************")
# # print(env.get_state())
# nex, re, _, do = env.step(None)
# print(nex)
# ep_reward += re
# # print("时隙", i, "的下一时刻", nex)
# i += 1
# vehicle_this_belong = [-1] * env.vehicle_number
# # for vec in env.vehicle_list:
# #     # print("车辆", vec.user_index, vec.application.task[0], vec.application.task[1],
# vec.application.task[
# #     # 0]*vec.application.task[1])
# #     vehicle_this_belong[vec.vehicle_index], _ = env.belong_cellular(vec.label_data[
# env.time_slot * env.loc_page])  # 车辆属于哪个基站
# while not do:
#     stat = env.get_state()
#     print("********************************************************时隙", i,
#     "***************************************")
#     # print(stat)
#     # ac1 = np.random.randint(0, 16, 100)
#     # print(ac1)
#
#     vehicle_this_belong = [-1] * env.vehicle_number
#     for vec in env.vehicle_list:
#         # print("车辆", vec.user_index, vec.application.task[0], vec.application.task[1],
#         vec.application.task[
#         # 0]*vec.application.task[1])
#         vehicle_this_belong[vec.vehicle_index], _ = env.belong_cellular(vec.label_data[
#         env.time_slot * env.loc_page])  # 车辆属于哪个基站
#     nex, reward, _, do = env.step(vehicle_this_belong)
#     print(reward)
#     ep_reward += reward
#     print(ep_reward)
#     # print("时隙", i, "的下一时刻", nex)
#     i += 1
