import numpy as np
import sys
import os
from typing import Tuple, List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Environments.env import Env
from T_Pattern_Tree.predictor import TrajectoryPredictor


class VECEnvironmentAdapter:
    """
    车联网环境适配器
    集成现有环境与TPPT预测，实现状态空间扩展和多目标奖励计算
    """
    
    def __init__(self, original_env: Env, trajectory_predictor: TrajectoryPredictor = None):
        """
        初始化环境适配器
        Args:
            original_env: 原始车联网环境
            trajectory_predictor: TPPT轨迹预测器
        """
        self.original_env = original_env
        self.trajectory_predictor = trajectory_predictor
        
        # 环境参数
        self.vehicle_number = original_env.vehicle_number
        self.cellular_number = original_env.cellular_number
        
        # 状态空间扩展参数
        self.original_state_dim = len(original_env.get_state_normalize())
        self.tppt_prediction_dim = 1  # TPPT预测的RSU ID
        self.enhanced_state_dim = self.original_state_dim + self.tppt_prediction_dim * self.vehicle_number
        
        # 奖励计算参数
        self.delay_weight = 1.0
        self.energy_weight = 1.0
        
        # 历史轨迹记录
        self.vehicle_trajectories = [[] for _ in range(self.vehicle_number)]
        
    def reset(self) -> np.ndarray:
        """
        重置环境并返回增强状态
        Returns:
            增强状态向量
        """
        # 重置原始环境
        state = self.original_env.reset_environment_normalize()
        
        # 清空轨迹记录
        self.vehicle_trajectories = [[] for _ in range(self.vehicle_number)]
        
        # 构建增强状态
        enhanced_state = self._build_enhanced_state(state)
        
        return enhanced_state
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], bool]:
        """
        执行动作并返回结果
        Args:
            actions: 每辆车的动作列表
        Returns:
            (next_state, reward_vector, info, done)
        """
        # 执行原始环境动作
        next_state, reward, vec_result, done = self.original_env.step(actions)
        
        # 更新车辆轨迹
        self._update_trajectories(actions)
        
        # 构建增强状态
        enhanced_next_state = self._build_enhanced_state(next_state)
        
        # 计算多目标奖励
        reward_vector = self._compute_multi_objective_rewards(actions, vec_result)
        
        # 构建信息字典
        info = {
            'original_reward': reward,
            'vec_result': vec_result,
            'actions': actions,
            'vehicle_trajectories': self.vehicle_trajectories.copy()
        }
        
        return enhanced_next_state, reward_vector, info, done
    
    def _build_enhanced_state(self, original_state: np.ndarray) -> np.ndarray:
        """
        构建增强状态空间
        st = (Lv(t), PredTPPT(t), ChSt, RSUS(t), TaskC(t))
        Args:
            original_state: 原始状态
        Returns:
            增强状态向量
        """
        enhanced_state = list(original_state)
        
        if self.trajectory_predictor is not None:
            # 获取TPPT预测
            tppt_predictions = self._get_tppt_predictions()
            
            # 将预测结果添加到状态中
            for pred in tppt_predictions:
                enhanced_state.append(pred)
        else:
            # 如果没有TPPT预测器，使用默认值
            for _ in range(self.vehicle_number):
                enhanced_state.append(0.0)
        
        return np.array(enhanced_state, dtype=np.float32)
    
    def _get_tppt_predictions(self) -> List[float]:
        """
        获取TPPT预测结果
        Returns:
            TPPT预测的RSU ID列表
        """
        predictions = []
        
        for vehicle_id in range(self.vehicle_number):
            if len(self.vehicle_trajectories[vehicle_id]) > 0:
                # 使用TPPT预测下一个RSU
                try:
                    predicted_rsu = self.trajectory_predictor.predict_next_rsu(
                        self.vehicle_trajectories[vehicle_id]
                    )
                    predictions.append(float(predicted_rsu))
                except Exception as e:
                    # 预测失败时使用默认值
                    print(f"TPPT预测失败: {e}")
                    predictions.append(0.0)
            else:
                # 轨迹为空时使用默认值
                predictions.append(0.0)
        
        return predictions
    
    def _update_trajectories(self, actions: List[int]):
        """
        更新车辆轨迹记录
        Args:
            actions: 执行的动作
        """
        for vehicle_id, action in enumerate(actions):
            # 将动作作为轨迹的一部分记录
            self.vehicle_trajectories[vehicle_id].append(action)
            
            # 限制轨迹长度
            if len(self.vehicle_trajectories[vehicle_id]) > 50:
                self.vehicle_trajectories[vehicle_id] = self.vehicle_trajectories[vehicle_id][-50:]
    
    def _compute_multi_objective_rewards(self, actions: List[int], vec_result: Dict[str, Any]) -> np.ndarray:
        """
        计算多目标奖励向量 rt = [rtD, rtE]
        Args:
            actions: 执行的动作
            vec_result: 环境执行结果
        Returns:
            多目标奖励向量 [延迟奖励, 能耗奖励]
        """
        # 获取车辆状态信息
        vehicles = self.original_env.vehicles
        
        delay_rewards = []
        energy_rewards = []
        
        for i, vehicle in enumerate(vehicles):
            action = actions[i]  # 0: 本地执行, 1: 卸载
            
            if action == 0:  # 本地执行
                delay = self._compute_local_delay(vehicle)
                energy = self._compute_local_energy(vehicle)
            else:  # 卸载执行
                delay = self._compute_offload_delay(vehicle, action)
                energy = self._compute_offload_energy(vehicle, action)
            
            # 延迟奖励（负延迟，最小化延迟）
            delay_reward = -delay * self.delay_weight
            # 能耗奖励（负能耗，最小化能耗）
            energy_reward = -energy * self.energy_weight
            
            delay_rewards.append(delay_reward)
            energy_rewards.append(energy_reward)
        
        # 计算总奖励
        total_delay_reward = sum(delay_rewards)
        total_energy_reward = sum(energy_rewards)
        
        return np.array([total_delay_reward, total_energy_reward], dtype=np.float32)
    
    def _compute_local_delay(self, vehicle) -> float:
        """
        计算本地执行延迟
        Args:
            vehicle: 车辆对象
        Returns:
            本地执行延迟
        """
        # 这里需要根据实际的车辆模型计算延迟
        # 暂时使用简化的计算方式
        task_size = getattr(vehicle, 'task_size', 1000)  # 任务大小
        cpu_freq = getattr(vehicle, 'cpu_freq', 1000)    # CPU频率
        
        local_delay = task_size / cpu_freq
        return local_delay
    
    def _compute_local_energy(self, vehicle) -> float:
        """
        计算本地执行能耗
        Args:
            vehicle: 车辆对象
        Returns:
            本地执行能耗
        """
        # 简化的能耗计算
        task_size = getattr(vehicle, 'task_size', 1000)
        cpu_freq = getattr(vehicle, 'cpu_freq', 1000)
        power_consumption = getattr(vehicle, 'power_consumption', 1.0)
        
        local_energy = (task_size / cpu_freq) * power_consumption
        return local_energy
    
    def _compute_offload_delay(self, vehicle, action: int) -> float:
        """
        计算卸载执行延迟
        Args:
            vehicle: 车辆对象
            action: 卸载动作
        Returns:
            卸载执行延迟
        """
        # 简化的卸载延迟计算
        task_size = getattr(vehicle, 'task_size', 1000)
        transmission_rate = getattr(vehicle, 'transmission_rate', 100)
        server_freq = getattr(vehicle, 'server_freq', 2000)
        
        # 传输延迟 + 计算延迟
        transmission_delay = task_size / transmission_rate
        computation_delay = task_size / server_freq
        
        offload_delay = transmission_delay + computation_delay
        return offload_delay
    
    def _compute_offload_energy(self, vehicle, action: int) -> float:
        """
        计算卸载执行能耗
        Args:
            vehicle: 车辆对象
            action: 卸载动作
        Returns:
            卸载执行能耗
        """
        # 简化的卸载能耗计算
        task_size = getattr(vehicle, 'task_size', 1000)
        transmission_rate = getattr(vehicle, 'transmission_rate', 100)
        transmission_power = getattr(vehicle, 'transmission_power', 0.5)
        
        # 传输能耗
        transmission_time = task_size / transmission_rate
        offload_energy = transmission_time * transmission_power
        
        return offload_energy
    
    def get_enhanced_state(self) -> np.ndarray:
        """
        获取当前增强状态
        Returns:
            当前增强状态
        """
        current_state = self.original_env.get_state_normalize()
        return self._build_enhanced_state(current_state)
    
    def get_state_dim(self) -> int:
        """获取增强状态维度"""
        return self.enhanced_state_dim
    
    def get_action_dim(self) -> int:
        """获取动作维度"""
        return self.cellular_number
    
    def get_vehicle_number(self) -> int:
        """获取车辆数量"""
        return self.vehicle_number
    
    def set_reward_weights(self, delay_weight: float, energy_weight: float):
        """
        设置奖励权重
        Args:
            delay_weight: 延迟权重
            energy_weight: 能耗权重
        """
        self.delay_weight = delay_weight
        self.energy_weight = energy_weight
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        获取环境信息
        Returns:
            环境信息字典
        """
        return {
            'vehicle_number': self.vehicle_number,
            'cellular_number': self.cellular_number,
            'original_state_dim': self.original_state_dim,
            'enhanced_state_dim': self.enhanced_state_dim,
            'tppt_enabled': self.trajectory_predictor is not None,
            'delay_weight': self.delay_weight,
            'energy_weight': self.energy_weight
        } 