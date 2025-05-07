#!/usr/bin/env python-- coding: utf-8 --
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Environments.env import Env  # 环境


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(1)


# --------------------- Replay Buffer ---------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """存储一个经验样本"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --------------------- Dueling DQN 网络 ---------------------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, vehicle_number, action_dim):
        """ 
        输入：state_dim 为环境状态维度（例如6*vehicle_number），
        vehicle_number 表示车辆数量， a
        ction_dim 表示每辆车的动作空间维度（蜂窝节点数量，16）。 
        输出：对于每辆车，输出 action_dim 个 Q 值，即输出 shape 为 (vehicle_number, action_dim) 
        """
        super(DuelingDQN, self).__init__()
        self.vehicle_number = vehicle_number
        hidden_size = 512  # 共享的前馈层，提取状态特征
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.relu = nn.ReLU()

        # 分为 Value 和 Advantage 两个分支
        self.value_fc = nn.Linear(hidden_size, vehicle_number)  # 每辆车对应一个状态值
        self.advantage_fc = nn.Linear(hidden_size, vehicle_number * action_dim)

        self.action_dim = action_dim

    def forward(self, state):
        """
        state: shape (batch_size, state_dim)
        输出：Q 值矩阵，shape (batch_size, vehicle_number, action_dim)
        """
        x = self.relu(self.fc1(state))
        # Value 部分：输出 shape (batch_size, vehicle_number, 1)
        value = self.value_fc(x).unsqueeze(-1)

        # Advantage 部分：输出 shape (batch_size, vehicle_number * action_dim)，重塑为 (batch_size,
        # vehicle_number, action_dim)
        advantage = self.advantage_fc(x).view(-1, self.vehicle_number, self.action_dim)
        # 去均值：对每辆车的 advantage 做均值消减
        advantage_mean = advantage.mean(dim=2, keepdim=True)
        qvals = value + (advantage - advantage_mean)
        return qvals


# --------------------- Agent 类 ---------------------
class Agent:
    def __init__(self, state_dim, vehicle_number, action_dim, lr=1e-4, gamma=0.99,
             buffer_capacity=10000, batch_size=64, epsilon_start=1.0, epsilon_final=0.1,
             epsilon_decay=5000):
        self.vehicle_number = vehicle_number
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

        # 策略网络与目标网络（双网络结构）
        self.policy_net = DuelingDQN(state_dim, vehicle_number, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, vehicle_number, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state):
        """
        输入：state 为当前状态，形状 (1, state_dim)
        输出：对每辆车选择 action（0 到 action_dim-1），返回 list 或 numpy 数组，长度为 vehicle_number
        """
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  math.exp(-1. * self.total_steps / self.epsilon_decay)
        self.total_steps += 1

        if random.random() < epsilon:
            # 随机选取动作：对于每辆车均随机选一个蜂窝节点编号
            return [random.randrange(self.action_dim) for _ in range(self.vehicle_number)]
        else:
            # 利用策略网络计算 Q 值，然后对每辆车选取最大 Q 值对应的动作
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, state_dim)
            with torch.no_grad():
                qvals = self.policy_net(state)  # shape: (1, vehicle_number, action_dim)
            qvals = qvals.squeeze(0).cpu().numpy()  # shape: (vehicle_number, action_dim)
            actions = [int(np.argmax(qvals[i])) for i in range(self.vehicle_number)]
            return actions

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)

        # 对于动作，由于每个样本的动作是一个 list，转换为 tensor，形状 (batch, vehicle_number)
        action = torch.LongTensor(np.array(action)).to(device)

        # 计算当前 Q 值（策略网络输出 Q，选取对应每辆车的动作）
        qvals = self.policy_net(state)  # (batch, vehicle_number, action_dim)
        # gather操作沿着最后维度选取动作对应 Q 值
        # 先扩展 action 至 (batch, vehicle_number, 1)
        action = action.unsqueeze(-1)
        current_q = qvals.gather(2, action)  # (batch, vehicle_number, 1)
        # 将每个样本 60 辆车的 Q 值求平均，得到单个标量评价
        current_q = current_q.mean(dim=1)  # (batch, 1)

        # 计算下一状态 Q 值：Double DQN，先用策略网络选择动作，再用目标网络评估
        with torch.no_grad():
            next_qvals = self.policy_net(next_state)  # (batch, vehicle_number, action_dim)
            next_actions = next_qvals.argmax(dim=2, keepdim=True)  # (batch, vehicle_number, 1)
            next_q_target = self.target_net(next_state)  # (batch, vehicle_number, action_dim)
            next_q = next_q_target.gather(2, next_actions)  # (batch, vehicle_number, 1)
            next_q = next_q.mean(dim=1)  # (batch, 1)
            expected_q = reward + (1 - done) * self.gamma * next_q

        # 计算均方差损失
        loss = nn.MSELoss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

#--------------------- 主训练循环 ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(num_episodes=200, target_update_interval=10):
    env = Env()
    # 状态维度：我们使用 get_state_normalize 返回的状态，形状为 (6 * vehicle_number,)
    state_dim = len(env.get_state_normalize())
    vehicle_number = env.vehicle_number
    action_dim = env.cellular_number # 每辆车可选择的蜂窝节点数量
    agent = Agent(state_dim, vehicle_number, action_dim)
    episode_rewards = []
    for ep in range(num_episodes):
        state = env.reset_environment_normalize()
        ep_reward = 0
        done = False
        step = 0
        losses = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, vec_result, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            step += 1

        if ep % target_update_interval == 0:
            agent.update_target()

        print("Episode: {}, Steps: {}, Episode Reward: {:.4f}, Loss: {:.6f}".format(
            ep, step, ep_reward, np.mean(losses) if losses else 0
        ))
        episode_rewards.append(ep_reward)
    return episode_rewards

if __name__ == 'main':
    rewards = train(num_episodes=100) # 可以保存模型和奖励曲线
    torch.save(rewards, "rewards.pt")
    torch.save(Agent, "modrl_cop_agent.pt")
    print("训练结束，模型已保存。")