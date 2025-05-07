from .config import RSU_COORDS, DATA_FILE_PATH, TRAIN_RATIO
from .data_processor import load_and_transform_data
from .utils import split_data
from .tree import T_tree
from .predictor import predict_next_rsu


def main():
    print("Starting RSU-based trajectory prediction adaptation...")
    print(f"RSU coordinates: {RSU_COORDS}")

    # 1. Load and transform data
    all_rsu_sequences = load_and_transform_data(DATA_FILE_PATH, RSU_COORDS)

    if not all_rsu_sequences:
        print("No RSU sequences loaded or generated. Exiting.")
        return

    # 2. Split data
    train_sequences, test_sequences = split_data(
        all_rsu_sequences, train_ratio=TRAIN_RATIO)

    if not train_sequences:
        print("No training sequences available after split. Exiting.")
        return

    # 3. Build T_tree
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

    # 4. Evaluate predictions
    correct_predictions = 0
    total_predictions = 0

    if not test_sequences:
        print("No test sequences to evaluate.")
        return

    print(f"\nEvaluating on {len(test_sequences)} test sequences...")
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
    main()
