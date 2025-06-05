import math
import os
import random
import torch
import numpy as np
from scipy.spatial import KDTree
from Environments.cellular_node import cellular_node
from Environments.vehicles import vehicle
from Environments.config import *


# 随机种子设置函数，设置后可复现结果
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Env:
    """此环境为方形环境，每台服务器的覆盖范围为正方形范围"""

    def __init__(self):
        seed_everything(RANDOM_SEED)  # 设置随机种子，保障可复现
        self.Ground_Length = GROUND_LENGTH  # 场地长度
        self.Ground_Width = GROUND_WIDTH  # 场地宽度
        self.cellular_number = CELLULAR_NUMBER  # 蜂窝基站数量
        self.vehicle_number = VEHICLE_NUMBER  # 车辆数量
        self.max_time_slot = MAX_TIME_SLOT  # 最大时隙数
        self.migration_one_hop = MIGRATION_ONE_HOP  # 迁移跳数
        self.backhaul_one_hop = BACKHAUL_ONE_HOP
        self.migration_prepare_time = MIGRATION_PREPARE_TIME  # 迁移准备时间
        self.cellular_list = self.get_cellulars()  # 获得蜂窝节点列表（坐标列表）

        self.vehicle_list = self.get_vehicles()  # 获得车辆列表
        
        self.cellular_locations = np.array(
            [cel.cellular_loc for cel in self.cellular_list])  # 车辆位置
        # 将所有基站的数据封装成kdtree,通过kdtree可查询到距离其最近的节点
        self.kdtree = KDTree(self.cellular_locations)
        self.loc_page = LOC_PAGE  # 位置偏移量
        self.loc_init = LOC_INIT  # 初始位置偏移
        self.state = None  # 环境初始状态为空
        self.time_slot = 0  # 系统运行到哪个时隙
        self.cellular_computation_load = [
            0] * self.cellular_number  # 统计蜂窝节点在该时隙的计算负载
        self.cellular_communication_load = [
            0] * self.cellular_number  # 统计蜂窝节点在该时隙的通信负载
        self.vehicle_this_belong = [-1] * self.vehicle_number  # 用户在该时隙属于哪个蜂窝站点
        self.vehicle_distance_this_belong = [
            0] * self.vehicle_number  # 用户与距离其最近基站的距离

    def get_cellulars(self):
        # 传入的都是基站的编号
        # 返回的是蜂窝基站的
        return np.array([cellular_node(cel) for cel in range(self.cellular_number)])

    def get_vehicles(self):
        # 传入的都是车辆的编号
        return np.array([vehicle(vec) for vec in range(self.vehicle_number)])

    def get_state(self):
        state = []
        for vec in self.vehicle_list:
            state.append(
                vec.label_data[self.loc_init + self.time_slot * self.loc_page][0])  # 车辆坐标 X轴
            state.append(
                vec.label_data[self.loc_init + self.time_slot * self.loc_page][1])  # 车辆坐标 Y轴
            state.append(vec.application.task[0])  # 任务数据量
            state.append(vec.application.task[1])  # 任务计算密度
            state.append(vec.application.instance)  # 服务实例数据大小
            # 当前服务实例位于哪个基站
            state.append(vec.application.instance_belong_cellular)
        self.state = np.array(state)
        return self.state

    def get_state_normalize(self):
        state = []
        for vec in self.vehicle_list:
            state.append(vec.label_data[self.loc_init + self.time_slot * self.loc_page][
                0] / self.Ground_Width)
            state.append(vec.label_data[self.loc_init + self.time_slot * self.loc_page][
                1] / self.Ground_Width)
            state.append(
                (vec.application.task[0] - 4194304) / (12582912 - 4194304))
            state.append((vec.application.task[1] - 200) / (1000 - 200))
            state.append((vec.application.instance - 4194304) /
                         (419430400 - 4194304))
            state.append(vec.application.instance_belong_cellular / 15)
        self.state = np.array(state)
        return self.state

    """重置环境函数"""

    def reset_environments(self):
        self.__init__()  # 重新设置随机数，获得基站，获得车辆
        return self.get_state()

    """重置环境函数（标准差）"""

    def reset_environment_normalize(self):
        self.__init__()
        return self.get_state_normalize()

    """车辆属于哪个蜂窝网络"""

    def belong_cellular(self, location):
        distance, nearest = self.kdtree.query(location)
        return nearest, distance  # 返回的是节点编号，以及距离节点的距离

    """计算两个节点之间的曼哈顿距离"""

    def manhattan_distance(self, i, j):
        """计算旧服务器与新服务器之间的曼哈顿距离"""
        """返回的是二者之间距离多少跳数"""
        """旧服务器编号，新服务器编号"""
        old_location_mec = self.cellular_list[i].cellular_loc
        new_location_mec = self.cellular_list[j].cellular_loc
        # print(abs(old_location_mec[0] - new_location_mec[0]))
        # print(abs(old_location_mec[step1] - new_location_mec[step1]))
        # print("二者之间的跳数为：", (abs(old_location_mec[0] - new_location_mec[0]) + abs(
        # old_location_mec[step1] - new_location_mec[step1])) / 1000)
        return (abs(old_location_mec[0] - new_location_mec[0]) + abs(
            old_location_mec[1] - new_location_mec[1])) / 1000

    """根据action，获取该实习所有蜂窝节点的负载：计算负载以及通信负载"""

    def get_this_load(self, time_slot, action):
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
            # print(self.cellular_computation_load)
            # print(np.int64(np.int64(vec.application.task[0]) * np.int64(vec.application.task[1])))
            # print(vec.application.task[0])
            # print(vec.application.task[1])
            if action is None:  # 如果action为None，则代表是第一个时隙，此时应该根据车辆的位置统计负载
                self.cellular_computation_load[cellular_index] += np.sqrt(np.int64(
                    np.int64(vec.application.task[0]) * np.int64(
                        vec.application.task[1])))  # 统计负载，到时候先按加权分配
            else:  # action不为空，根据action统计负载
                self.cellular_computation_load[action[vec.vehicle_index]] += np.sqrt(np.int64(
                    np.int64(vec.application.task[0]) * np.int64(
                        vec.application.task[1])))  # 统计负载，到时候先按加权分配
        # print("车辆属于哪个基站：", self.vehicle_this_belong)
        # print("蜂窝网络通信负载：", self.cellular_communication_load)
        # print("蜂窝网络计算负载：", self.cellular_computation_load)

    """通信时间"""

    def communication_time(self, vec, action):
        # print("-------------------------------》", vec.vehicle_index, "进入了通信函数")
        """Step 1 : 得出本地基站的位置以及二者之间的距离"""
        local_cellular = self.vehicle_this_belong[vec.vehicle_index]  # 本地基站的编号
        # print("local_cellular:", local_cellular)
        # 直接计算距离，不用开根号（方便后面的计算）
        length = self.vehicle_distance_this_belong[vec.vehicle_index] ** 2
        # print("车辆位置：", vec.label_data[self.time_slot])
        # print("本地基站位置", self.cellular_list[local_cellular].cellular_loc)
        # print("距离的平方:", length)

        """Step 2 : 算出用户与本地基站之间的传输速率，当前基站给用户分配了多少通信资源"""
        bandwidth = (1 / self.cellular_communication_load[local_cellular]) * self.cellular_list[
            local_cellular].bandwidth
        # print("带宽:", bandwidth)
        channel_gain = abs(
            self.cellular_list[local_cellular].channel_gain / length)
        # print("信道增益:", channel_gain)
        trans_rate = bandwidth * math.log2(
            1 + vec.P * channel_gain / self.cellular_list[local_cellular].gaussian_noise)
        # print("传输速率:", trans_rate)

        """Step 3 : 用户将任务传输到本地基站"""
        communication_time1 = vec.application.task[0] / trans_rate
        # print("任务数据量：", vec.application.task[0])

        """Step 4 ： 任务从本地基站通过回程链路传输到服务基站（前提条件：本地基站与服务基站不是一个基站）"""
        hop = 0  # 到目标服务器的条数
        # 代表是其他时隙，需要判断一下，服务实例位于哪个位置，然后转发
        if local_cellular != action[vec.vehicle_index]:
            hop = self.manhattan_distance(
                local_cellular, action[vec.vehicle_index])
            # _, hop_distance, _ = dijkstra(self.cellular_matrix, local_cellular)
            communication_time2_1 = vec.application.task[0] / self.cellular_list[
                local_cellular].backhaul_network
            communication_time2_2 = self.backhaul_one_hop * hop
            # print("目标基站：", action[vec.vehicle_index])
            # print("跳数：", hop)
        else:  # 服务实例就位于本地基站位置，不需要转发
            communication_time2_1 = 0
            communication_time2_2 = 0
            hop = 0
        # print("从本地基站发出任务的时间：", communication_time1)
        # print("从本地基站传输到服务基站的时间：", communication_time2)
        return round(communication_time1, 8), round(communication_time2_1, 8), round(
            communication_time2_2, 8), local_cellular, hop

    """计算时间"""

    def computation_time(self, vec, action):
        """之前已经预先统计以下信息：用户该时隙所属蜂窝节点，各个蜂窝节点的计算负载，通信负载"""
        # print("------------------------------------->车辆", vec.vehicle_index, "进入了computation函数")
        # print("所有蜂窝节点的计算负载：", self.cellular_computation_load)
        """Step1 : 目标action基站给用户分配了多少计算资源"""
        compute_vol = np.int64(
            vec.application.task[0]) * np.int64(vec.application.task[1])  # 计算量
        ac = action[vec.vehicle_index]  # 在哪里计算

        capability = (np.sqrt(compute_vol) / self.cellular_computation_load[ac]) * \
            self.cellular_list[ac].capability
        # print("目标计算基站：", action[vec.vehicle_index])
        # print("分配的计算能力：", capability)
        computation_time = compute_vol / capability
        # print("任务计算量：", compute_vol)
        # print("计算时间：", computation_time)
        return round(computation_time, 8), capability

    """迁移时间"""

    def migration_time(self, vec, action):
        """之前已经预先统计以下信息：用户该时隙所属蜂窝节点，各个蜂窝节点的计算负载，通信负载"""
        # print("------------------------------------->车辆", vec.vehicle_index, "进入了migration函数")
        # print("服务迁移之前，服务实例所在的服务器：", vec.application.instance_belong_cellular)
        ins_belong = vec.application.instance_belong_cellular  # 迁移之前服务实例所在的MEC服务器编号
        ac = action[vec.vehicle_index]

        # s, hop_distance, _ = dijkstra(self.cellular_matrix, ins_belong)
        hop = self.manhattan_distance(ins_belong, ac)
        migration_time1 = vec.application.instance / \
            self.cellular_list[ins_belong].backhaul_network
        migration_time2 = hop * self.migration_one_hop

        # print("服务实例数据：", vec.application.instance)
        # print("目标基站：", action[vec.vehicle_index])
        # print("跳数：", hop)
        # print("服务数据从本地基站发出的时间：", migration_time1)
        # print("服务数据从本地基站经过回程链路传输的时间：", migration_time2)
        return round(migration_time1, 8), round(migration_time2, 8), hop

    """计算单辆车完成任务的消耗"""

    def vehicle_all_consume(self, action):  # 将动作输入,计算收益，目前支出按照能耗计算，皆与时间有关，可以先计算时间
        """目前不涉及流量感知，因此暂时蜂窝网络的拓扑图是固定的,在环境初始化时就生成了拓扑图"""
        # print("------------------------------------->进入了效益计算函数")
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
                """代表此时是第0个时隙，应该创建服务实例"""
                mig_time = self.migration_prepare_time

                comm_time1, comm_time2_1, comm_time2_2, local_cellular, comm_hop = (
                    self.communication_time(
                        vec, action))  # 进入通信函数
                comm_time = comm_time1 + comm_time2_1 + comm_time2_2  # 全部通信时间

                comp_time, capability = self.computation_time(
                    vec, action)  # 进入计算函数计算
                # print("用户", vec.vehicle_index, "通信时间1：", comm_time2_1, "通信时间2：", comm_time2_2,
                # "计算时间：", comp_time, "频率", capability)

            elif vec.application.instance_belong_cellular == ac:
                """服务实例上一时隙所在的服务器与新的动作指向的是同一个服务器，不进行迁移"""
                comm_time1, comm_time2_1, comm_time2_2, local_cellular, comm_hop = (
                    self.communication_time(
                        vec, action))  # 进入通信函数
                comm_time = comm_time1 + comm_time2_1 + comm_time2_2  # 全部通信时间

                comp_time, capability = self.computation_time(
                    vec, action)  # 进入计算函数计算
                # print("用户", vec.vehicle_index, "通信时间1：", comm_time2_1, "通信时间2：", comm_time2_2,
                # "计算时间：", comp_time, "频率", capability)

            elif vec.application.instance_belong_cellular != ac:
                """服务实例上一时隙所在的服务器与新的动作指向的不是同一个服务器，应当进行迁移，迁移到action处"""
                if self.cellular_list[ac].server_app == 0:
                    mig_prepare_time = self.migration_prepare_time
                mig_time1, mig_time2, mig_hop = self.migration_time(
                    vec, action)  # 进入迁移函数
                # print("目标环境准备时间：", mig_prepare_time)
                # 全部迁移时间,取迁移时间和环境准备时间最大的那一个
                mig_time = max(mig_time1 + mig_time2, mig_prepare_time)

                comm_time1, comm_time2_1, comm_time2_2, local_cellular, comm_hop = (
                    self.communication_time(
                        vec, action))  # 进入通信函数
                comm_time = comm_time1 + comm_time2_1 + comm_time2_2  # 全部通信时间

                comp_time, capability = self.computation_time(
                    vec, action)  # 进入计算函数
                # print("用户", vec.vehicle_index, "通信时间1：", comm_time2_1, "通信时间2：", comm_time2_2,
                # "计算时间：", comp_time, "频率", capability, "迁移时间：",mig_time1)

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

    """执行动作函数，严格遵循st，at，rt，st+1的操作序列"""

    def step(self, action):
        """此刻传入的action，是t时刻的动作，时隙应该在函数结束后再+1，而action代表的是每个智能体的动作"""
        done = False  # 记录epoch是否结束训练

        """Step 1: 根据当前动作，获取时隙t的负载：用户属于哪个蜂窝，每个蜂窝网络的通信负载和节点负载"""
        self.get_this_load(self.time_slot, action)
        # print("属于哪个基站：", self.vehicle_this_belong)
        if action is None:
            final_action = self.vehicle_this_belong
        else:
            final_action = action
        # print("真正的动作:", final_action)

        """Step 2: 执行收益函数计算"""
        reward, vec_result = self.vehicle_all_consume(final_action)

        """Step3：判断结束条件（目前的截止条件是当所有的时隙运行完毕）"""
        self.time_slot += 1
        if self.time_slot == self.max_time_slot:
            done = True
            self.time_slot = 0

        """Step 4：更新下一状态（需要显式调用的只有用户的任务以及服务实例,其余信息，只要我们更新timeslot就可获得）"""
        for vec in self.vehicle_list:
            vec.application.generate_task()  # 车辆上的智能应用产生新任务
            vec.application.instance_change()  # 智能应用对应的服务实例大小变化
            vec.application.instance_belong_cellular = final_action[
                vec.vehicle_index]  # 车辆的服务所在的边缘服务器
        return self.get_state_normalize(), reward, vec_result, done


env = Env()
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
