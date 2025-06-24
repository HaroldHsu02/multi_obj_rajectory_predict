"""
DDQN算法配置文件
包含网络、训练、偏好、缓冲池等所有配置参数
"""

DDQN_CONFIG = {
    'network': {
        'hidden_dims': [256, 128, 64],
        'learning_rate': 1e-4,
        'target_update_freq': 100,
        'gamma': 0.99,
        'use_dueling': True,  # 是否使用Dueling架构
        'dropout_rate': 0.1
    },
    'training': {
        'max_episodes': 5000,
        'horizon_steps': 200,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'epsilon_decay': 5000,
        'eval_interval': 100,  # 评估间隔
        'save_interval': 500,  # 保存间隔
        'logging_enabled': True
    },
    'preference': {
        'preference_space_size': 11,  # ω ∈ {0.0, 0.1, ..., 1.0}
        'non_dominated_sorting': True,
        'validation_enabled': True
    },
    'buffer': {
        'capacity': 10000
    },
    'tppt': {
        'min_support': 0.1,
        'window_size': 10,
        'enabled': True  # 是否启用TPPT预测
    },
    'environment': {
        'delay_weight': 1.0,
        'energy_weight': 1.0,
        'reward_scale': 1.0
    }
}

# 不同场景的配置变体
DDQN_CONFIGS = {
    'default': DDQN_CONFIG,
    
    'fast_training': {
        'network': {
            'hidden_dims': [128, 64],
            'learning_rate': 5e-4,
            'target_update_freq': 50,
            'gamma': 0.99,
            'use_dueling': False,
            'dropout_rate': 0.1
        },
        'training': {
            'max_episodes': 1000,
            'horizon_steps': 100,
            'batch_size': 16,
            'epsilon_start': 1.0,
            'epsilon_final': 0.1,
            'epsilon_decay': 2000,
            'eval_interval': 50,
            'save_interval': 200,
            'logging_enabled': True
        },
        'preference': {
            'preference_space_size': 6,  # 减少偏好空间大小
            'non_dominated_sorting': False,
            'validation_enabled': True
        },
        'buffer': {
            'capacity': 5000
        },
        'tppt': {
            'min_support': 0.1,
            'window_size': 10,
            'enabled': False  # 禁用TPPT以加快训练
        },
        'environment': {
            'delay_weight': 1.0,
            'energy_weight': 1.0,
            'reward_scale': 1.0
        }
    },
    
    'high_precision': {
        'network': {
            'hidden_dims': [512, 256, 128, 64],
            'learning_rate': 5e-5,
            'target_update_freq': 200,
            'gamma': 0.99,
            'use_dueling': True,
            'dropout_rate': 0.2
        },
        'training': {
            'max_episodes': 10000,
            'horizon_steps': 300,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_final': 0.05,
            'epsilon_decay': 8000,
            'eval_interval': 200,
            'save_interval': 1000,
            'logging_enabled': True
        },
        'preference': {
            'preference_space_size': 21,  # 更细粒度的偏好空间
            'non_dominated_sorting': True,
            'validation_enabled': True
        },
        'buffer': {
            'capacity': 20000
        },
        'tppt': {
            'min_support': 0.05,
            'window_size': 20,
            'enabled': True
        },
        'environment': {
            'delay_weight': 1.0,
            'energy_weight': 1.0,
            'reward_scale': 1.0
        }
    },
    
    'delay_focused': {
        'network': {
            'hidden_dims': [256, 128, 64],
            'learning_rate': 1e-4,
            'target_update_freq': 100,
            'gamma': 0.99,
            'use_dueling': True,
            'dropout_rate': 0.1
        },
        'training': {
            'max_episodes': 5000,
            'horizon_steps': 200,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_final': 0.1,
            'epsilon_decay': 5000,
            'eval_interval': 100,
            'save_interval': 500,
            'logging_enabled': True
        },
        'preference': {
            'preference_space_size': 11,
            'non_dominated_sorting': True,
            'validation_enabled': True
        },
        'buffer': {
            'capacity': 10000
        },
        'tppt': {
            'min_support': 0.1,
            'window_size': 10,
            'enabled': True
        },
        'environment': {
            'delay_weight': 2.0,  # 更重视延迟
            'energy_weight': 0.5,
            'reward_scale': 1.0
        }
    },
    
    'energy_focused': {
        'network': {
            'hidden_dims': [256, 128, 64],
            'learning_rate': 1e-4,
            'target_update_freq': 100,
            'gamma': 0.99,
            'use_dueling': True,
            'dropout_rate': 0.1
        },
        'training': {
            'max_episodes': 5000,
            'horizon_steps': 200,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_final': 0.1,
            'epsilon_decay': 5000,
            'eval_interval': 100,
            'save_interval': 500,
            'logging_enabled': True
        },
        'preference': {
            'preference_space_size': 11,
            'non_dominated_sorting': True,
            'validation_enabled': True
        },
        'buffer': {
            'capacity': 10000
        },
        'tppt': {
            'min_support': 0.1,
            'window_size': 10,
            'enabled': True
        },
        'environment': {
            'delay_weight': 0.5,  # 更重视能耗
            'energy_weight': 2.0,
            'reward_scale': 1.0
        }
    }
}

def get_config(config_name: str = 'default') -> dict:
    """
    获取指定配置
    Args:
        config_name: 配置名称
    Returns:
        配置字典
    """
    if config_name not in DDQN_CONFIGS:
        print(f"警告: 配置 '{config_name}' 不存在，使用默认配置")
        return DDQN_CONFIGS['default']
    
    return DDQN_CONFIGS[config_name]

def merge_config(base_config: dict, custom_config: dict) -> dict:
    """
    合并配置
    Args:
        base_config: 基础配置
        custom_config: 自定义配置
    Returns:
        合并后的配置
    """
    import copy
    
    merged_config = copy.deepcopy(base_config)
    
    def merge_dict(base, custom):
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value
    
    merge_dict(merged_config, custom_config)
    return merged_config

def validate_config(config: dict) -> bool:
    """
    验证配置的有效性
    Args:
        config: 配置字典
    Returns:
        是否有效
    """
    try:
        # 检查必需的配置项
        required_sections = ['network', 'training', 'preference', 'buffer', 'tppt', 'environment']
        for section in required_sections:
            if section not in config:
                print(f"错误: 缺少配置节 '{section}'")
                return False
        
        # 验证网络配置
        network_config = config['network']
        if network_config['learning_rate'] <= 0:
            print("错误: 学习率必须大于0")
            return False
        
        if network_config['gamma'] <= 0 or network_config['gamma'] >= 1:
            print("错误: gamma必须在(0,1)范围内")
            return False
        
        # 验证训练配置
        training_config = config['training']
        if training_config['max_episodes'] <= 0:
            print("错误: max_episodes必须大于0")
            return False
        
        if training_config['batch_size'] <= 0:
            print("错误: batch_size必须大于0")
            return False
        
        # 验证偏好配置
        preference_config = config['preference']
        if preference_config['preference_space_size'] <= 1:
            print("错误: preference_space_size必须大于1")
            return False
        
        # 验证缓冲池配置
        buffer_config = config['buffer']
        if buffer_config['capacity'] <= 0:
            print("错误: buffer capacity必须大于0")
            return False
        
        return True
        
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False

def print_config_summary(config: dict):
    """
    打印配置摘要
    Args:
        config: 配置字典
    """
    print("=== DDQN配置摘要 ===")
    print(f"网络架构: {'Dueling' if config['network']['use_dueling'] else 'Standard'}")
    print(f"隐藏层: {config['network']['hidden_dims']}")
    print(f"学习率: {config['network']['learning_rate']}")
    print(f"训练轮数: {config['training']['max_episodes']}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"偏好空间大小: {config['preference']['preference_space_size']}")
    print(f"缓冲池容量: {config['buffer']['capacity']}")
    print(f"TPPT启用: {config['tppt']['enabled']}")
    print(f"延迟权重: {config['environment']['delay_weight']}")
    print(f"能耗权重: {config['environment']['energy_weight']}")
    print("==================") 