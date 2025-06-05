import os
import sys
import numpy as np
# 添加项目根目录到Python路径（当直接运行此文件时）
if __name__ == "__main__":
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))

from T_Pattern_Tree.config import RSU_COORDS, DATA_FILE_PATH, TRANSFORMED_DATA_PATH, TRAIN_RATIO
from T_Pattern_Tree.data_processor import load_and_transform_data
from T_Pattern_Tree.utils import split_data
from T_Pattern_Tree.tree import T_tree
from T_Pattern_Tree.predictor import predict_next_rsu


def main(force_transform=False):
    """
    主函数，执行RSU轨迹预测
    Args:
        force_transform (bool): 是否强制重新执行转换，即使转换后的文件已存在
    """
    print("Starting RSU-based trajectory prediction adaptation...")
    #打印RSU坐标
    for rsu_id, coords in RSU_COORDS.items():
        print(f"{rsu_id}: {coords}")
    ########################################################################
    # 1. Load and transform data
    print("\n#########################1. Loading and transforming data#########################\n")
    all_rsu_sequences = load_and_transform_data(
        DATA_FILE_PATH,
        RSU_COORDS,
        transformed_data_path=TRANSFORMED_DATA_PATH,
        force_transform=force_transform
    )

    if all_rsu_sequences.size == 0:
        print("No RSU sequences loaded or generated. Exiting.")
        return

    print(f"RSU sequences shape: {all_rsu_sequences.shape}")
    ########################################################################
    # 2. Split data
    # 划分数据集为训练集和测试集
    print("\n#########################2. Splitting data#########################\n")
    train_sequences, test_sequences = split_data(
        all_rsu_sequences, train_ratio=TRAIN_RATIO)
    # 打印训练集和测试集的形状
    if len(train_sequences) == 0:
        print("No training sequences available after split. Exiting.")
        return
    ########################################################################
    # 3. Build T_tree
    print("\n#########################3. Building T_tree#########################\n")
    prediction_tree_root = T_tree("root")

    for rsu_seq in train_sequences:
        current_t_node = prediction_tree_root
        for rsu_id_key in rsu_seq:
            child_found = False
            for child in current_t_node.child:
                if child.key == rsu_id_key:
                    child.support += 1
                    current_t_node = child
                    child_found = True
                    break
            if not child_found:
                new_child = T_tree(rsu_id_key)
                current_t_node.child.append(new_child)
                current_t_node = new_child

    print("T_tree built successfully from training data.")
    ########################################################################
    # 4. Evaluate predictions
    print("\n#########################4. Evaluating predictions#########################\n")
    correct_predictions = 0
    total_predictions = 0

    if len(test_sequences) == 0:
        print("No test sequences to evaluate.")
        return

    print(f"Evaluating on {len(test_sequences)} test sequences...")
    for test_idx, full_rsu_sequence in enumerate(test_sequences):
        if len(full_rsu_sequence) < 2:
            continue

        history = list(full_rsu_sequence[:-1])
        actual_next_rsu = full_rsu_sequence[-1]

        predicted_rsu = None
        temp_history = list(history)

        while not predicted_rsu and temp_history:
            predicted_rsu = predict_next_rsu(
                prediction_tree_root, temp_history)
            if not predicted_rsu:
                temp_history = temp_history[1:]

        if predicted_rsu:
            total_predictions += 1
            if predicted_rsu == actual_next_rsu:
                correct_predictions += 1

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(
            f"\nPrediction Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    else:
        print("\nNo predictions could be made on the test set.")

 
if __name__ == "__main__":
    # 默认使用已转换的数据（如果存在）
    # 如需强制重新执行转换，可以设置force_transform=True
    force_transform = True

    # 解析命令行参数
    if len(sys.argv) > 1 and sys.argv[1].lower() == "force":
        force_transform = True
        print("Forcing data transformation...")

    main(force_transform)
