import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def perform_mcnemar_test(results_dict):
    """
    Performs McNemar's test to compare model predictions.
    Returns p-values matrix for model pairs.
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    p_values = np.zeros((n_models, n_models))
    
    # Extract predictions from each model
    predictions = {}
    labels_of = {}
    
    for name, data in results_dict.items():
        if 'test_set_evaluation' in data:
            eval_data = data['test_set_evaluation']
        else:
            eval_data = data
            
        preds = eval_data.get('predictions', [])
        labels = eval_data.get('labels', [])
        
        if len(preds) > 0:
            predictions[name] = np.array(preds)
            labels_of[name] = np.array(labels)
    
    # Perform McNemar's test for all pairs; it is only valid when both models saw the same test samples
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j or model1 not in predictions or model2 not in predictions:
                continue
            if not np.array_equal(labels_of[model1], labels_of[model2]):
                p_values[i, j] = np.nan
                continue
                
            # Create contingency table
            pred1_correct = predictions[model1] == labels_of[model1]
            pred2_correct = predictions[model2] == labels_of[model2]
            
            both_correct = np.sum((pred1_correct & pred2_correct))
            only_1_correct = np.sum((pred1_correct & ~pred2_correct))
            only_2_correct = np.sum((~pred1_correct & pred2_correct))
            both_incorrect = np.sum((~pred1_correct & ~pred2_correct))
            
            table = np.array([[both_correct, only_2_correct], 
                               [only_1_correct, both_incorrect]])
            
            # Calculate McNemar's test
            p_values[i, j] = mcnemar(table, exact=True).pvalue
    
    return pd.DataFrame(p_values, index=model_names, columns=model_names)