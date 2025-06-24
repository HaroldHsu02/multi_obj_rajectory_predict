import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

from ..Models.ddqn_agent import DDQNAgent
from ..Adapters.vec_environment_adapter import VECEnvironmentAdapter


class DDQNTrainer:
    """
    DDQN训练器，实现算法2的完整训练流程
    负责偏好采样、经验收集、网络训练和性能评估
    """
    
    def __init__(self, env_adapter: VECEnvironmentAdapter, agent: DDQNAgent, config: Dict[str, Any]):
        """
        初始化训练器
        Args:
            env_adapter: 环境适配器
            agent: DDQN智能体
            config: 训练配置
        """
        self.env_adapter = env_adapter
        self.agent = agent
        self.config = config
        
        # 训练配置
        training_config = config.get('training', {})
        self.max_episodes = training_config.get('max_episodes', 5000)
        self.horizon_steps = training_config.get('horizon_steps', 200)
        self.eval_interval = training_config.get('eval_interval', 100)
        self.save_interval = training_config.get('save_interval', 500)
        
        # 日志配置
        self.logging_enabled = training_config.get('logging_enabled', True)
        if self.logging_enabled:
            self._setup_logging()
        
        # 训练状态
        self.current_episode = 0
        self.total_steps = 0
        self.training_start_time = None
        
        # 性能记录
        self.episode_rewards = []
        self.episode_losses = []
        self.evaluation_results = []
        
        # 偏好统计
        self.preference_usage = {}
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'ddqn_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self, max_episodes: int = None) -> Dict[str, Any]:
        """
        主训练循环
        Args:
            max_episodes: 最大训练轮数，None表示使用配置中的值
        Returns:
            训练结果统计
        """
        if max_episodes is None:
            max_episodes = self.max_episodes
        
        self.training_start_time = time.time()
        
        if self.logging_enabled:
            self.logger.info(f"开始训练，最大轮数: {max_episodes}")
            self.logger.info(f"环境信息: {self.env_adapter.get_environment_info()}")
        
        try:
            for episode in range(max_episodes):
                self.current_episode = episode
                
                # 训练一个episode
                episode_reward, episode_loss = self.train_episode(episode)
                
                # 记录性能
                self.episode_rewards.append(episode_reward)
                if episode_loss is not None:
                    self.episode_losses.append(episode_loss)
                
                # 定期评估
                if episode % self.eval_interval == 0 and episode > 0:
                    eval_result = self.evaluate_performance(eval_episodes=10)
                    self.evaluation_results.append({
                        'episode': episode,
                        'result': eval_result
                    })
                    
                    if self.logging_enabled:
                        self.logger.info(f"Episode {episode}: 评估结果 - {eval_result}")
                
                # 定期保存模型
                if episode % self.save_interval == 0 and episode > 0:
                    self._save_checkpoint(episode)
                
                # 打印进度
                if episode % 100 == 0:
                    self._print_progress(episode, max_episodes)
        
        except KeyboardInterrupt:
            if self.logging_enabled:
                self.logger.info("训练被用户中断")
        
        # 训练完成
        training_time = time.time() - self.training_start_time
        final_stats = self._get_final_stats(training_time)
        
        if self.logging_enabled:
            self.logger.info(f"训练完成，总时间: {training_time:.2f}秒")
            self.logger.info(f"最终统计: {final_stats}")
        
        return final_stats
    
    def train_episode(self, episode_idx: int) -> Tuple[float, float]:
        """
        训练一个episode
        Args:
            episode_idx: episode索引
        Returns:
            (episode_reward, episode_loss)
        """
        # 采样当前偏好
        current_preference = self.agent.preference_manager.sample_preference()
        
        # 记录偏好使用
        pref_key = tuple(np.round(current_preference, 2))
        self.preference_usage[pref_key] = self.preference_usage.get(pref_key, 0) + 1
        
        # 重置环境
        state = self.env_adapter.reset()
        
        episode_reward = 0.0
        episode_loss = None
        
        # 收集经验
        for step in range(self.horizon_steps):
            # 选择动作
            action = self.agent.select_action(state, current_preference)
            
            # 执行动作
            next_state, reward_vector, info, done = self.env_adapter.step(action)
            
            # 存储经验
            self.agent.store_experience(state, action, reward_vector, next_state, done, current_preference)
            
            # 计算标量化奖励
            scalarized_reward = np.dot(reward_vector, current_preference)
            episode_reward += scalarized_reward
            
            # 训练网络
            if self.agent.can_train():
                batch_data = self.agent.sample_batch()
                loss = self.agent.train_step(batch_data)
                episode_loss = loss if episode_loss is None else (episode_loss + loss) / 2
            
            state = next_state
            self.total_steps += 1
            
            if done:
                break
        
        # 更新目标网络
        if episode_idx % self.agent.target_update_freq == 0:
            self.agent.update_target_network()
        
        return episode_reward, episode_loss
    
    def evaluate_performance(self, eval_episodes: int = 10) -> Dict[str, float]:
        """
        评估性能
        Args:
            eval_episodes: 评估episode数量
        Returns:
            评估结果
        """
        eval_rewards = []
        eval_delay_rewards = []
        eval_energy_rewards = []
        
        # 测试不同的偏好
        test_preferences = [
            np.array([1.0, 0.0]),  # 只关注延迟
            np.array([0.0, 1.0]),  # 只关注能耗
            np.array([0.5, 0.5]),  # 平衡偏好
        ]
        
        for pref in test_preferences:
            pref_rewards = []
            pref_delay_rewards = []
            pref_energy_rewards = []
            
            for _ in range(eval_episodes):
                state = self.env_adapter.reset()
                episode_reward = 0.0
                episode_delay_reward = 0.0
                episode_energy_reward = 0.0
                
                for step in range(self.horizon_steps):
                    # 使用贪婪策略
                    action = self.agent.select_action(state, pref, epsilon=0.0)
                    
                    next_state, reward_vector, info, done = self.env_adapter.step(action)
                    
                    # 记录各目标奖励
                    episode_delay_reward += reward_vector[0]
                    episode_energy_reward += reward_vector[1]
                    episode_reward += np.dot(reward_vector, pref)
                    
                    state = next_state
                    if done:
                        break
                
                pref_rewards.append(episode_reward)
                pref_delay_rewards.append(episode_delay_reward)
                pref_energy_rewards.append(episode_energy_reward)
            
            eval_rewards.extend(pref_rewards)
            eval_delay_rewards.extend(pref_delay_rewards)
            eval_energy_rewards.extend(pref_energy_rewards)
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_delay_reward': np.mean(eval_delay_rewards),
            'avg_energy_reward': np.mean(eval_energy_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }
    
    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint_path = f"checkpoints/ddqn_episode_{episode}.pth"
        
        # 确保目录存在
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        # 保存模型
        self.agent.save_model(checkpoint_path)
        
        # 保存训练状态
        training_state = {
            'episode': episode,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'evaluation_results': self.evaluation_results,
            'preference_usage': self.preference_usage
        }
        
        import pickle
        with open(f"checkpoints/training_state_{episode}.pkl", 'wb') as f:
            pickle.dump(training_state, f)
        
        if self.logging_enabled:
            self.logger.info(f"保存检查点: {checkpoint_path}")
    
    def _print_progress(self, episode: int, max_episodes: int):
        """打印训练进度"""
        progress = (episode + 1) / max_episodes * 100
        
        # 计算平均奖励
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        # 计算平均损失
        recent_losses = self.episode_losses[-100:] if self.episode_losses else []
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        # 获取训练统计
        training_stats = self.agent.get_training_stats()
        
        print(f"Episode {episode + 1}/{max_episodes} ({progress:.1f}%) - "
              f"Avg Reward: {avg_reward:.3f}, Avg Loss: {avg_loss:.3f}, "
              f"Buffer: {training_stats['buffer_size']}, "
              f"Preferences: {training_stats['encountered_preferences']}")
    
    def _get_final_stats(self, training_time: float) -> Dict[str, Any]:
        """获取最终统计信息"""
        return {
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'training_time': training_time,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'final_avg_loss': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0.0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0.0,
            'min_reward': np.min(self.episode_rewards) if self.episode_rewards else 0.0,
            'encountered_preferences': len(self.agent.preference_manager.get_encountered_preferences()),
            'preference_usage': self.preference_usage,
            'evaluation_results': self.evaluation_results
        }
    
    def load_checkpoint(self, episode: int):
        """加载检查点"""
        checkpoint_path = f"checkpoints/ddqn_episode_{episode}.pth"
        training_state_path = f"checkpoints/training_state_{episode}.pkl"
        
        # 加载模型
        self.agent.load_model(checkpoint_path)
        
        # 加载训练状态
        import pickle
        with open(training_state_path, 'rb') as f:
            training_state = pickle.load(f)
        
        self.current_episode = training_state['episode']
        self.total_steps = training_state['total_steps']
        self.episode_rewards = training_state['episode_rewards']
        self.episode_losses = training_state['episode_losses']
        self.evaluation_results = training_state['evaluation_results']
        self.preference_usage = training_state['preference_usage']
        
        if self.logging_enabled:
            self.logger.info(f"加载检查点: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'evaluation_results': self.evaluation_results,
            'preference_usage': self.preference_usage,
            'agent_stats': self.agent.get_training_stats(),
            'environment_info': self.env_adapter.get_environment_info()
        } 