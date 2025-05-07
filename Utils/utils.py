import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.File_operation import FileOperation


def create_directory(path: str, sub_path_list: list):
    for sub_path in sub_path_list:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Path: {} create successfully!'.format(path + sub_path))
        else:
            print('Path: {} is already existence!'.format(path + sub_path))


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, color='b', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_


def save_result(dataset, flag, MEC, UE):
    filepath = FileOperation.get_BASE_DIR() + "/Result" + "/" + str(flag) + "_" + str(MEC) + "_" + str(UE)  # 获取存储路径
    np.save(filepath, dataset)  # 保存为npy文件
    print("单次奖励保存成功")


def make_dir(*paths):
    """创建文件夹"""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print('Path: {} create successfully!'.format(path))
        else:
            print('Path: {} is already existence!'.format(path))


def del_empty_dir(*paths):
    """删除目录下所有空文件夹"""
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    """保存奖励"""
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')
