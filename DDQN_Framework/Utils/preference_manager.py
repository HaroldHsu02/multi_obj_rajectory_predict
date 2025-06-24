import numpy as np
import random
from typing import List, Tuple, Optional


class PreferenceManager:
    """
    偏好管理器，负责管理候选偏好空间和遇到偏好
    实现偏好采样、验证、非支配排序等功能
    """
    
    def __init__(self, preference_space_config: dict):
        """
        初始化偏好管理器
        Args:
            preference_space_config: 偏好空间配置
                - preference_space_size: 偏好空间大小
                - non_dominated_sorting: 是否使用非支配排序
                - validation_enabled: 是否启用偏好验证
        """
        self.preference_space_size = preference_space_config.get('preference_space_size', 11)
        self.non_dominated_sorting = preference_space_config.get('non_dominated_sorting', True)
        self.validation_enabled = preference_space_config.get('validation_enabled', True)
        
        # 生成候选偏好空间Ω
        self.preference_space = self._generate_preference_space(self.preference_space_size)
        
        # 遇到的偏好集合W
        self.encountered_preferences = []
        
        # 偏好性能记录
        self.preference_performance = {}
        
    def _generate_preference_space(self, size: int) -> List[np.ndarray]:
        """
        生成候选偏好空间
        Args:
            size: 偏好空间大小
        Returns:
            偏好空间列表，每个偏好为[ωD, ωE]格式
        """
        preferences = []
        step = 1.0 / (size - 1) if size > 1 else 1.0
        
        for i in range(size):
            omega_d = i * step  # 延迟偏好权重
            omega_e = 1.0 - omega_d  # 能耗偏好权重
            preference = np.array([omega_d, omega_e])
            preferences.append(preference)
            
        return preferences
    
    def sample_preference(self) -> np.ndarray:
        """
        从候选偏好空间Ω中采样偏好
        Returns:
            采样的偏好权重 [ωD, ωE]
        """
        preference = random.choice(self.preference_space)
        
        # 验证偏好
        if self.validation_enabled:
            self.validate_preference(preference)
            
        return preference
    
    def add_encountered_preference(self, preference: np.ndarray, performance: Optional[dict] = None):
        """
        添加遇到的偏好到集合W中
        Args:
            preference: 偏好权重 [ωD, ωE]
            performance: 偏好性能指标（可选）
        """
        if self.validation_enabled:
            self.validate_preference(preference)
        
        # 检查是否已存在
        preference_tuple = tuple(preference)
        if preference_tuple not in [tuple(p) for p in self.encountered_preferences]:
            self.encountered_preferences.append(preference)
            
            # 记录性能
            if performance is not None:
                self.preference_performance[preference_tuple] = performance
    
    def select_encountered_preference(self, non_dominated_sorting: bool = None) -> np.ndarray:
        """
        从遇到偏好中选择偏好
        Args:
            non_dominated_sorting: 是否使用非支配排序，None表示使用默认设置
        Returns:
            选择的偏好权重 [ωD, ωE]
        """
        if not self.encountered_preferences:
            # 如果没有遇到偏好，从候选空间采样
            return self.sample_preference()
        
        if non_dominated_sorting is None:
            non_dominated_sorting = self.non_dominated_sorting
        
        if non_dominated_sorting and len(self.encountered_preferences) > 1:
            # 使用非支配排序选择偏好
            sorted_preferences = self._non_dominated_sorting(self.encountered_preferences)
            return random.choice(sorted_preferences[0])  # 选择第一层非支配解
        else:
            # 随机选择
            return random.choice(self.encountered_preferences)
    
    def validate_preference(self, preference: np.ndarray) -> bool:
        """
        验证偏好权重是否有效
        Args:
            preference: 偏好权重 [ωD, ωE]
        Returns:
            是否有效
        Raises:
            ValueError: 如果偏好无效
        """
        if not isinstance(preference, np.ndarray) or preference.shape != (2,):
            raise ValueError("偏好必须是形状为(2,)的numpy数组")
        
        omega_d, omega_e = preference
        
        # 检查权重范围
        if omega_d < 0 or omega_e < 0:
            raise ValueError("偏好权重必须非负")
        
        # 检查权重和为1
        if abs(omega_d + omega_e - 1.0) > 1e-6:
            raise ValueError("偏好权重和必须为1")
        
        return True
    
    def _non_dominated_sorting(self, preferences: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        对偏好进行非支配排序
        Args:
            preferences: 偏好列表
        Returns:
            分层排序结果，每层包含非支配偏好
        """
        if not self.preference_performance:
            # 如果没有性能记录，返回原始偏好
            return [preferences]
        
        # 构建性能矩阵
        performance_matrix = []
        valid_preferences = []
        
        for pref in preferences:
            pref_tuple = tuple(pref)
            if pref_tuple in self.preference_performance:
                performance_matrix.append(self.preference_performance[pref_tuple])
                valid_preferences.append(pref)
        
        if not performance_matrix:
            return [preferences]
        
        # 执行非支配排序
        fronts = self._fast_non_dominated_sort(performance_matrix, valid_preferences)
        
        return fronts
    
    def _fast_non_dominated_sort(self, performance_matrix: List[dict], 
                                preferences: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        快速非支配排序算法
        Args:
            performance_matrix: 性能矩阵
            preferences: 对应的偏好列表
        Returns:
            分层排序结果
        """
        n = len(performance_matrix)
        domination_count = [0] * n  # 被支配次数
        dominated_solutions = [[] for _ in range(n)]  # 支配的解
        
        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(performance_matrix[i], performance_matrix[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(performance_matrix[j], performance_matrix[i]):
                        domination_count[i] += 1
        
        # 构建前沿
        fronts = []
        current_front = []
        
        for i in range(n):
            if domination_count[i] == 0:
                current_front.append(preferences[i])
        
        fronts.append(current_front)
        
        # 构建后续前沿
        while current_front:
            next_front = []
            for i, pref in enumerate(current_front):
                pref_idx = preferences.index(pref)
                for dominated_idx in dominated_solutions[pref_idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(preferences[dominated_idx])
            
            current_front = next_front
            if current_front:
                fronts.append(current_front)
        
        return fronts
    
    def _dominates(self, perf1: dict, perf2: dict) -> bool:
        """
        判断性能1是否支配性能2
        Args:
            perf1: 性能1
            perf2: 性能2
        Returns:
            是否支配
        """
        # 假设性能指标越小越好
        at_least_one_better = False
        
        for key in perf1:
            if key in perf2:
                if perf1[key] > perf2[key]:  # 性能1更差
                    return False
                elif perf1[key] < perf2[key]:  # 性能1更好
                    at_least_one_better = True
        
        return at_least_one_better
    
    def get_preference_space(self) -> List[np.ndarray]:
        """获取候选偏好空间"""
        return self.preference_space.copy()
    
    def get_encountered_preferences(self) -> List[np.ndarray]:
        """获取遇到偏好集合"""
        return self.encountered_preferences.copy()
    
    def get_preference_performance(self) -> dict:
        """获取偏好性能记录"""
        return self.preference_performance.copy()
    
    def clear_encountered_preferences(self):
        """清空遇到偏好集合"""
        self.encountered_preferences = []
        self.preference_performance = {}
    
    def update_preference_performance(self, preference: np.ndarray, performance: dict):
        """
        更新偏好性能记录
        Args:
            preference: 偏好权重
            performance: 性能指标
        """
        preference_tuple = tuple(preference)
        self.preference_performance[preference_tuple] = performance 