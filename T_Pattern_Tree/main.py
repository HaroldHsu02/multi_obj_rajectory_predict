import os
import sys
import numpy as np
# 添加项目根目录到Python路径（当直接运行此文件时）
if __name__ == "__main__":
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))

from T_Pattern_Tree.config import RSU_COORDS, DATA_FILE_PATH, TRANSFORMED_DATA_PATH, TRAIN_RATIO, USE_TIMESLOT_MODE, SLIDING_WINDOW_SIZE
from T_Pattern_Tree.data_processor import load_and_transform_data, validate_timeslot_sequence, get_sequence_statistics
from T_Pattern_Tree.utils import split_data
from T_Pattern_Tree.tree import T_tree
from T_Pattern_Tree.predictor import predict_next_rsu


def validate_configuration():
    """
    验证配置参数的有效性
    """
    if SLIDING_WINDOW_SIZE <= 0:
        raise ValueError(
            f"SLIDING_WINDOW_SIZE must be positive, got {SLIDING_WINDOW_SIZE}")

    if not isinstance(USE_TIMESLOT_MODE, bool):
        raise ValueError(
            f"USE_TIMESLOT_MODE must be boolean, got {type(USE_TIMESLOT_MODE)}")

    print(f"Configuration validation passed:")
    print(f"  - USE_TIMESLOT_MODE: {USE_TIMESLOT_MODE}")
    print(f"  - SLIDING_WINDOW_SIZE: {SLIDING_WINDOW_SIZE}")
    print(f"  - TRAIN_RATIO: {TRAIN_RATIO}")


def main(force_transform=False):
    """
    主函数，执行RSU轨迹预测
    Args:
        force_transform (bool): 是否强制重新执行转换，即使转换后的文件已存在
    """
    print("Starting RSU-based trajectory prediction adaptation...")

    # 验证配置
    try:
        validate_configuration()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # 检查数据文件是否存在
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        print("Please ensure the trajectory data file exists.")
        return

    # 打印RSU坐标
    for rsu_id, coords in RSU_COORDS.items():
        print(f"{rsu_id}: {coords}")
    ########################################################################
    # 1. Load and transform data
    print("\n#########################1. Loading and transforming data#########################\n")
    try:
        all_rsu_sequences = load_and_transform_data(
            DATA_FILE_PATH,
            RSU_COORDS,
            transformed_data_path=TRANSFORMED_DATA_PATH,
            force_transform=force_transform
        )
    except Exception as e:
        print(f"Error during data loading and transformation: {e}")
        return

    if all_rsu_sequences.size == 0:
        print("No RSU sequences loaded or generated. Exiting.")
        return

    print(f"RSU sequences shape: {all_rsu_sequences.shape}")

    # 验证时序数据
    if USE_TIMESLOT_MODE:
        print("Validating timeslot sequences...")
        # 存储有效的RSU序列
        valid_sequences = []
        # 记录无效序列的数量
        invalid_count = 0
        # 遍历所有RSU序列
        for i, seq in enumerate(all_rsu_sequences):
            # 验证时序序列是否有效
            if validate_timeslot_sequence(seq):
                valid_sequences.append(seq)
            else:
                invalid_count += 1
                if invalid_count <= 5:  # 只显示前5个无效序列
                    print(f"Warning: Invalid sequence {i}: {seq[:10]}...")

        if invalid_count > 0:
            print(
                f"Filtered {invalid_count} invalid sequences out of {len(all_rsu_sequences)}")
            all_rsu_sequences = np.array(valid_sequences, dtype=object)

        # 打印统计信息
        try:
            stats = get_sequence_statistics(all_rsu_sequences)
            print(f"Sequence statistics: {stats}")
        except Exception as e:
            print(f"Warning: Could not generate statistics: {e}")

    ########################################################################
    # 2. Split data
    # 划分数据集为训练集和测试集
    print("\n#########################2. Splitting data#########################\n")
    try:
        train_sequences, test_sequences = split_data(
            all_rsu_sequences, train_ratio=TRAIN_RATIO)
        # 打印训练集和测试集的形状
        print(f"Train sequences shape: {len(train_sequences)}")
        print(f"Test sequences shape: {len(test_sequences)}")   
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return

    # 打印训练集和测试集的形状
    if len(train_sequences) == 0:
        print("No training sequences available after split. Exiting.")
        return
    ########################################################################
    # 3. Build T_tree
    print("\n#########################3. Building T_tree#########################\n")
    try:
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
    except Exception as e:
        print(f"Error during T_tree construction: {e}")
        return
    ########################################################################
    # 4. Evaluate predictions
    print("\n#########################4. Evaluating predictions#########################\n")
    correct_predictions = 0
    total_predictions = 0
    failed_predictions = 0

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
        try:
            if USE_TIMESLOT_MODE:
                # 时序模式：直接使用滑动窗口预测
                predicted_rsu = predict_next_rsu(
                    prediction_tree_root, history, SLIDING_WINDOW_SIZE)
            else:
                # 原有模式：逐步缩短历史
                temp_history = list(history)
                while not predicted_rsu and temp_history:
                    predicted_rsu = predict_next_rsu(
                        prediction_tree_root, temp_history)
                    if not predicted_rsu:
                        temp_history = temp_history[1:]
        except Exception as e:
            print(f"Warning: Prediction failed for sequence {test_idx}: {e}")
            failed_predictions += 1
            continue

        if predicted_rsu:
            total_predictions += 1
            if predicted_rsu == actual_next_rsu:
                correct_predictions += 1
        else:
            failed_predictions += 1

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(
            f"\nPrediction Results:")
        print(f"  - Total predictions: {total_predictions}")
        print(f"  - Correct predictions: {correct_predictions}")
        print(f"  - Failed predictions: {failed_predictions}")
        print(f"  - Accuracy: {accuracy:.2f}%")

        # 保存预测结果统计
        save_prediction_results(
            correct_predictions, total_predictions, failed_predictions, accuracy)

        # 打印详细统计信息
        print_detailed_statistics(test_sequences, prediction_tree_root)
    else:
        print("\nNo predictions could be made on the test set.")


def save_prediction_results(correct_predictions, total_predictions, failed_predictions, accuracy):
    """
    保存预测结果统计
    Args:
        correct_predictions (int): 正确预测数
        total_predictions (int): 总预测数
        failed_predictions (int): 失败预测数
        accuracy (float): 准确率
    """
    try:
        results = {
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'failed_predictions': failed_predictions,
            'accuracy': accuracy,
            'use_timeslot_mode': USE_TIMESLOT_MODE,
            'sliding_window_size': SLIDING_WINDOW_SIZE,
            'train_ratio': TRAIN_RATIO
        }

        # 确保结果目录存在
        results_dir = os.path.join(os.path.dirname(
            DATA_FILE_PATH), "..", "..", "Results")
        os.makedirs(results_dir, exist_ok=True)

        # 生成文件名
        mode_str = "timeslot" if USE_TIMESLOT_MODE else "original"
        filename = f"prediction_results_{mode_str}_window{SLIDING_WINDOW_SIZE}.npy"
        filepath = os.path.join(results_dir, filename)

        # 保存结果
        np.save(filepath, results)
        print(f"Prediction results saved to {filepath}")

    except Exception as e:
        print(f"Warning: Could not save prediction results: {e}")


def print_detailed_statistics(test_sequences, prediction_tree_root):
    """
    打印详细的预测统计信息
    Args:
        test_sequences (list): 测试序列
        prediction_tree_root (T_tree): 预测树根节点
    """
    if not USE_TIMESLOT_MODE:
        return

    print("\n#########################5. Detailed Statistics#########################\n")

    # 统计各RSU的预测准确率
    rsu_accuracy = {}
    rsu_predictions = {}

    for test_idx, full_rsu_sequence in enumerate(test_sequences):
        if len(full_rsu_sequence) < 2:
            continue

        history = list(full_rsu_sequence[:-1])
        actual_next_rsu = full_rsu_sequence[-1]

        try:
            predicted_rsu = predict_next_rsu(
                prediction_tree_root, history, SLIDING_WINDOW_SIZE)

            if predicted_rsu:
                # 统计实际RSU的预测情况
                if actual_next_rsu not in rsu_accuracy:
                    rsu_accuracy[actual_next_rsu] = {'correct': 0, 'total': 0}

                rsu_accuracy[actual_next_rsu]['total'] += 1
                if predicted_rsu == actual_next_rsu:
                    rsu_accuracy[actual_next_rsu]['correct'] += 1

                # 统计预测RSU的分布
                rsu_predictions[predicted_rsu] = rsu_predictions.get(
                    predicted_rsu, 0) + 1

        except Exception:
            continue

    # 打印各RSU的准确率
    print("RSU-wise Prediction Accuracy:")
    for rsu_id, stats in rsu_accuracy.items():
        if stats['total'] > 0:
            acc = (stats['correct'] / stats['total']) * 100
            print(
                f"  {rsu_id}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    # 打印预测分布
    print(f"\nPrediction Distribution:")
    total_preds = sum(rsu_predictions.values())
    for rsu_id, count in sorted(rsu_predictions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_preds) * 100 if total_preds > 0 else 0
        print(f"  {rsu_id}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    # 默认使用已转换的数据（如果存在）
    # 如需强制重新执行转换，可以设置force_transform=True
    force_transform = True

    # 解析命令行参数
    if len(sys.argv) > 1 and sys.argv[1].lower() == "force":
        force_transform = True
        print("Forcing data transformation...")

    main(force_transform)
