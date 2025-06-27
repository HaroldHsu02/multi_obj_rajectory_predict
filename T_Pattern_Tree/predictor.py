'''
Author: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
Date: 2025-05-07 20:17:27
LastEditors: HaroldHsu02 88320487+HaroldHsu02@users.noreply.github.com
LastEditTime: 2025-06-27 16:25:42
FilePath: \multi_obj\T_Pattern_Tree\predictor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import sys

# 添加项目根目录到Python路径（当直接运行此文件时）
if __name__ == "__main__":
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))

from T_Pattern_Tree.tree import T_tree
from T_Pattern_Tree.config import SLIDING_WINDOW_SIZE, USE_TIMESLOT_MODE


def get_sliding_window_history(history_rsu_sequence, window_size=None):
    """
    从历史RSU序列中提取滑动窗口
    Args:
        history_rsu_sequence (list): 完整的历史RSU序列
        window_size (int, optional): 滑动窗口大小，默认使用配置中的SLIDING_WINDOW_SIZE
    Returns:
        list: 滑动窗口内的RSU序列
    """
    if window_size is None:
        window_size = SLIDING_WINDOW_SIZE

    if not history_rsu_sequence:
        return []

    # 如果历史长度小于窗口大小，返回全部历史
    if len(history_rsu_sequence) <= window_size:
        return history_rsu_sequence

    # 返回最近window_size个时隙的历史
    return history_rsu_sequence[-window_size:]


def t_tree_dfs_prediction_recursive(
    current_path_nodes, current_treenode, target_sequence_keys, all_predicted_paths_list
):
    """
    使用深度优先搜索(DFS)递归地在T树中寻找可能的下一个RSU预测。

    参数:
        current_path_nodes (list): 当前已匹配的T树节点列表,记录了与target_sequence_keys匹配的路径。
        current_treenode (T_tree): 当前遍历到的T树节点。
        target_sequence_keys (list): 需要匹配的历史RSU序列(字符串列表)。
        all_predicted_paths_list (list): 用于存储所有找到的完整预测路径。

    工作流程:
    1. 如果已经完全匹配了目标序列:
       - 检查当前节点是否有子节点(可能的预测)
       - 将每个子节点作为预测添加到匹配路径中
       - 将完整路径添加到预测列表

    2. 如果遇到死胡同(无子节点)且未完全匹配,则返回

    3. 递归步骤:
       - 获取下一个需要匹配的key
       - 在当前节点的子节点中寻找匹配
       - 找到匹配时,将节点加入路径并继续递归
       - 回溯时移除节点,以便探索其他可能的路径
    """
    # 基本情况1: 成功匹配整个目标序列
    if len(current_path_nodes) == len(target_sequence_keys):
        if current_treenode.child:  # 如果有子节点,这些都是潜在的预测
            for next_potential_node in current_treenode.child:
                # 复制当前匹配路径
                path_with_prediction = list(current_path_nodes)
                path_with_prediction.append(
                    next_potential_node)  # 添加预测节点
                all_predicted_paths_list.append(path_with_prediction)
        return

    # 基本情况2: 在完全匹配前遇到不匹配或死胡同
    if not current_treenode.child and len(current_path_nodes) < len(target_sequence_keys):
        return

    # 递归步骤: 尝试匹配目标序列中的下一个key
    if len(current_path_nodes) < len(target_sequence_keys):
        key_to_match = target_sequence_keys[len(current_path_nodes)]
        found_match_in_children = False
        for child_node in current_treenode.child:
            if child_node.key == key_to_match:
                found_match_in_children = True
                current_path_nodes.append(child_node)
                t_tree_dfs_prediction_recursive(
                    current_path_nodes,
                    child_node,
                    target_sequence_keys,
                    all_predicted_paths_list,
                )
                current_path_nodes.pop()  # 回溯:移除节点以探索其他兄弟路径


def predict_next_rsu(tree_root, history_rsu_sequence, window_size=None):
    """
    基于历史RSU序列预测下一个RSU,支持滑动窗口方式。
    
    参数:
        tree_root (T_tree): 预测树的根节点
        history_rsu_sequence (list): RSU ID的历史序列
        window_size (int, optional): 滑动窗口大小,默认使用SLIDING_WINDOW_SIZE
        
    返回:
        str: 预测的下一个RSU ID,如果无法预测则返回None
    """
    # 如果历史序列为空,无法预测
    if not history_rsu_sequence:
        return None

    if USE_TIMESLOT_MODE:
        # 时序模式：使用滑动窗口进行预测
        # 获取滑动窗口内的历史序列
        window_history = get_sliding_window_history(
            history_rsu_sequence, window_size)
        if not window_history:
            return None

        # 使用深度优先搜索在T树中寻找所有可能的预测路径
        all_predicted_paths = []
        t_tree_dfs_prediction_recursive(
            [], tree_root, window_history, all_predicted_paths)

        if not all_predicted_paths:
            return None

        # 统计每个候选RSU的总支持度
        candidate_predictions = {}  # RSU_key -> total_support
        for path in all_predicted_paths:
            if path:
                predicted_node = path[-1]  # 获取路径最后一个节点
                support = predicted_node.support  # 获取节点的支持度
                key = predicted_node.key  # 获取RSU ID
                # 累加该RSU的总支持度
                candidate_predictions[key] = candidate_predictions.get(
                    key, 0) + support

        if not candidate_predictions:
            return None

        # 选择支持度最高的RSU作为预测结果
        best_rsu_key = None
        highest_score = -1
        for rsu_key, total_support in candidate_predictions.items():
            if total_support > highest_score:
                highest_score = total_support
                best_rsu_key = rsu_key

        return best_rsu_key
    else:
        # 非时序模式：使用完整历史序列进行预测
        # 流程与时序模式相同,但使用完整历史而非滑动窗口
        all_predicted_paths = []
        t_tree_dfs_prediction_recursive(
            [], tree_root, history_rsu_sequence, all_predicted_paths)

        if not all_predicted_paths:
            return None

        candidate_predictions = {}
        for path in all_predicted_paths:
            if path:
                predicted_node = path[-1]
                support = predicted_node.support
                key = predicted_node.key
                candidate_predictions[key] = candidate_predictions.get(
                    key, 0) + support

        if not candidate_predictions:
            return None

        best_rsu_key = None
        highest_score = -1
        for rsu_key, total_support in candidate_predictions.items():
            if total_support > highest_score:
                highest_score = total_support
                best_rsu_key = rsu_key

        return best_rsu_key
