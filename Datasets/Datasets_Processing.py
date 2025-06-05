from numpy import double
from idlelib.window import add_windows_to_menu
from collections import namedtuple
from File_operation import FileOperation
import numpy as np
import pandas as pd
import sys
import os

# 添加项目根目录到Python路径
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

# 先添加路径后导入

Point = namedtuple("Point", ["x", "y"])


# 内部函数
def _read_traces_from_the_txt(file_path):
    """从已知文件中读取用户的轨迹信息"""
    f = open(file_path, "r")
    users_traces = {}
    lines = f.readlines()
    for line in lines:
        if line[0] == "[":
            continue
        items = line.split()
        x = float(items[1])
        y = float(items[2])
        if items[0] not in users_traces.keys():
            users_traces[items[0]] = []
            users_traces[items[0]].append(np.array([x, y]))
        else:
            users_traces[items[0]].append(np.array([x, y]))
    f.close()
    """将其转换为csv文件存储"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pd.DataFrame(users_traces).to_csv(
        os.path.join(current_dir, "rome_traces_coordinate.csv"), index=False
    )
    print("完成数据预处理！")


def generate_dataset(filepath):
    """
    对罗马数据进行处理，生成数据集，
    格式为(用户数, 时隙数)
    每个元素是一个字符串，表示该用户在该时隙的坐标（例如 "[x y]"）
    """
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_filepath = os.path.join(current_dir, filepath)

    print(f"尝试打开文件: {full_filepath}")

    # 确保文件存在
    if not os.path.exists(full_filepath):
        print(f"错误: 文件不存在 {full_filepath}")
        return

    with open(full_filepath, encoding="utf-8") as f:
        # 读入罗马数据的所有用户在所有时隙的轨迹数据，横轴是时隙，竖轴是用户编号
        data = np.loadtxt(f, str, delimiter=",", skiprows=1)
        # data 是一个二维数组，其形状为 (时隙数,用户数)
        # 每一行代表一个用户
        # 每一列代表一个时隙
        # 每个元素是一个字符串，表示该用户在该时隙的坐标（例如 "[x y]"）

    print("车辆数量为：", len(data[0]))  # 车辆数量
    print("数据集的形状为：", len(data), len(data[0]))
    datasets = []  # 最终要保存的数据集合
    ###################################################################################
    # 获取存储路径
    output_dir = os.path.join(FileOperation.get_BASE_DIR(), "Datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rome_trajectory.npy")
    ###################################################################################
    for i in range(len(data)):  # 遍历时隙  data(时隙数,用户数)
        temp = []
        for j in range(len(data[0])):  # 遍历用户
            arrays = data[i][j].split()  # 将当前用户在当前时隙的数据字符串按空格分割成数组
            if (len(arrays) == 2):  # 第一个元素去掉开头的'['作为x坐标    第二个元素去掉结尾的']'作为y坐标
                # print(np.shape(arrays))
                temp.append(
                    np.array([double(arrays[0][1:]),
                             double(arrays[1][:-1])]) / 2
                )
            elif len(arrays) == 4:
                temp.append(
                    np.array([double(arrays[1]), double(arrays[2])]) / 2)
            elif len(arrays) == 3:
                if arrays[0] == "[":
                    temp.append(
                        np.array(
                            [double(arrays[1]), double(arrays[2][:-1])]) / 2
                    )
                else:
                    temp.append(
                        np.array([double(arrays[0][1:]),
                                 double(arrays[1])]) / 2
                    )
        """我认为每个用户的任务计算密度以及服务数据大小在这个时间段内不可改变，改变的只有每个时间点生成的任务
            数据大小"""
        datasets.append(temp)  # 将其加入dataset
    np.save(output_path, np.array(datasets))  # 保存为npy文件
    # 打印dataset形状
    print("生成的数据集形状:", np.array(datasets).shape)
    print(f"数据集已保存到: {output_path}")
    print("数据集生成完毕")


if __name__ == "__main__":
    # 首先尝试寻找rome_traces_coordinate.csv文件
    # 如果找不到，可以先处理原始的.txt文件
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # csv_path是rome_traces_coordinate.csv文件的路径
        csv_path = os.path.join(current_dir, "Datasets",
                                "rome_traces_coordinate.csv")
        if os.path.exists(csv_path):
            generate_dataset(os.path.join(
                "Datasets", "rome_traces_coordinate.csv"))
        else:
            # 尝试寻找并处理txt文件
            txt_path = os.path.join(
                current_dir, "Datasets", "rome_traces_coordinate.txt")
            if os.path.exists(txt_path):
                _read_traces_from_the_txt(txt_path)
                generate_dataset(os.path.join(
                    "Datasets", "rome_traces_coordinate.csv"))
            else:
                print(f"错误: 找不到数据文件 {csv_path} 或 {txt_path}")
    except Exception as e:
        print(f"处理数据时出错: {e}")
