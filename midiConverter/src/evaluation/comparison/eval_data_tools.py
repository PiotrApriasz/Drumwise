import json
import os

from src.constants import EVALUATION_RESULTS_PATH


def load_evaluation_results(results_dir=EVALUATION_RESULTS_PATH):
    """
    Loads all evaluation results from the specified directory.

    Returns:
        Dict mapping experiment names to their results data
    """
    all_results = {}

    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    exp_name = data.get('experiment_name', os.path.splitext(filename)[0])
                    all_results[exp_name] = data
                except json.JSONDecodeError:
                    print(f"Error loading {filename}: Invalid JSON format")

    return all_results


def filter_results(all_results, model_type=None, feature_type=None):
    """Filters results by model type and/or feature type"""

    # zaimplementowac pobieranie tylko najnowszych dla danego modelu i feature type
    filtered = {}

    for exp_name, data in all_results.items():
        current_model = data.get('model_type', exp_name.split('_')[0])
        current_feature = data.get('feature_type', exp_name.split('_')[1] if len(exp_name.split('_')) > 1 else None)

        if (model_type is None or current_model == model_type) and \
                (feature_type is None or current_feature == feature_type):
            filtered[exp_name] = data

    return filtered