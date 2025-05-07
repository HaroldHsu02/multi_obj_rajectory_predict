import os

def generate_tree(directory, prefix=""):
    """
    生成指定目录的树状结构图并打印到控制台。

    参数:
        directory: 要遍历的目录路径
        prefix: 用于显示层级的前缀（默认为空）
    """
    # 获取目录中的所有文件和文件夹
    entries = os.listdir(directory)
    entries.sort()  # 排序以确保顺序一致

    # 遍历每个条目
    for i, entry in enumerate(entries):
        # 构建完整路径
        path = os.path.join(directory, entry)
        # 判断是否是最后一个条目
        is_last = (i == len(entries) - 1)

        # 打印当前条目
        if is_last:
            print(f"{prefix}└── {entry}")
            new_prefix = prefix + "    "
        else:
            print(f"{prefix}├── {entry}")
            new_prefix = prefix + "│   "
            

        # 如果是文件夹，则递归遍历其内容
        if os.path.isdir(path):
            generate_tree(path, new_prefix)

def main():
    # 获取用户输入的目标目录
    target_directory = input("请输入要生成树状结构的目录路径: ")

    # 检查目录是否存在
    if not os.path.exists(target_directory):
        print(f"错误: 目录 '{target_directory}' 不存在!")
        return

    # 打印树状结构
    print(f"\n目录结构: {target_directory}")
    generate_tree(target_directory)
    print("树状结构生成完成")


    # # 将输出保存到文件
    # output_file = "directory_tree.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     print(f"目录结构: {target_directory}", file=f)
    #     generate_tree(target_directory, file=f)

    # print(f"树状结构已保存到 {output_file}")

if __name__ == "__main__":
    main()  