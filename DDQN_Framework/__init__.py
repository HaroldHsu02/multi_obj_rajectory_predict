"""
DDQN_Framework - 基于改进DRL的动态权重算法实现
"""

__version__ = "1.0.0"
__author__ = "DDQN Team"

from .Models.ddqn_agent import DDQNAgent
from .Models.neural_networks import MultiObjectiveQNetwork, DuelingMultiObjectiveQNetwork
from .Utils.preference_manager import PreferenceManager
from .Utils.experience_buffer import MultiObjectiveExperienceBuffer
from .Adapters.vec_environment_adapter import VECEnvironmentAdapter
from .Training.trainer import DDQNTrainer
from .config import get_config, validate_config, print_config_summary

__all__ = [
    'DDQNAgent',
    'MultiObjectiveQNetwork',
    'DuelingMultiObjectiveQNetwork',
    'PreferenceManager',
    'MultiObjectiveExperienceBuffer',
    'VECEnvironmentAdapter',
    'DDQNTrainer',
    'get_config',
    'validate_config',
    'print_config_summary'
] 