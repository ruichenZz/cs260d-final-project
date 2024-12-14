import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

Data_dir="./jigsaw-dataset"

# Define your helper functions for computing metrics
def compute_auc(y_true, y_pred):
    """Compute ROC-AUC score."""
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_bias_aucs(y_true, y_pred, subgroup_mask):
    """
    Compute Bias AUCs: Subgroup AUC, BPSN AUC, BNSP AUC.
    
    Args:
        y_true (np.array): True binary labels (0/1).
        y_pred (np.array): Predicted scores or probabilities.
        subgroup_mask (np.array): Mask for the specific subgroup.
    
    Returns:
        dict: Subgroup AUC, BPSN AUC, and BNSP AUC.
    """
    subgroup_true = y_true[subgroup_mask]
    subgroup_pred = y_pred[subgroup_mask]

    background_mask = ~subgroup_mask
    background_true = y_true[background_mask]
    background_pred = y_pred[background_mask]

    # Subgroup AUC
    subgroup_auc = compute_auc(subgroup_true, subgroup_pred)

    # BPSN AUC (Background Positive, Subgroup Negative)
    bpsn_mask = background_true == 1
    bpsn_auc = compute_auc(
        np.concatenate([subgroup_true, background_true[bpsn_mask]]),
        np.concatenate([subgroup_pred, background_pred[bpsn_mask]])
    )

    # BNSP AUC (Background Negative, Subgroup Positive)
    bnsp_mask = background_true == 0
    bnsp_auc = compute_auc(
        np.concatenate([subgroup_true, background_true[bnsp_mask]]),
        np.concatenate([subgroup_pred, background_pred[bnsp_mask]])
    )

    return {"subgroup_auc": subgroup_auc, "bpsn_auc": bpsn_auc, "bnsp_auc": bnsp_auc}

def generalized_mean(values, p):
    """Compute the generalized mean."""
    values = np.array(values)
    values = values[~np.isnan(values)]  # Ignore NaNs
    return (np.mean(values**p))**(1/p)

def compute_final_metric(overall_auc, bias_aucs, p=-5, weight=0.25):
    """
    Compute the final metric combining overall AUC and bias AUCs.
    
    Args:
        overall_auc (float): ROC-AUC score for the entire dataset.
        bias_aucs (list of dict): List of AUC dictionaries for each subgroup.
        p (float): Power for the generalized mean.
        weight (float): Weight for each component in the final score.
    
    Returns:
        float: Final metric score.
    """
    subgroup_aucs = [bias["subgroup_auc"] for bias in bias_aucs]
    bpsn_aucs = [bias["bpsn_auc"] for bias in bias_aucs]
    bnsp_aucs = [bias["bnsp_auc"] for bias in bias_aucs]

    gm_subgroup_auc = generalized_mean(subgroup_aucs, p)
    gm_bpsn_auc = generalized_mean(bpsn_aucs, p)
    gm_bnsp_auc = generalized_mean(bnsp_aucs, p)

    bias_score = (weight * gm_subgroup_auc +
                  weight * gm_bpsn_auc +
                  weight * gm_bnsp_auc)

    return (weight * overall_auc) + bias_score

# Example usage with your dataframe
def evaluate_metrics(dataframe, subgroups, true_label_col, predicted_label_col):
    """
    Evaluate metrics on a dataframe with true and predicted labels.
    
    Args:
        dataframe (pd.DataFrame): Dataframe with predictions and subgroups.
        subgroups (list): List of subgroup column names.
        true_label_col (str): Column name for true labels.
        predicted_label_col (str): Column name for predicted probabilities/scores.
    
    Returns:
        dict: A dictionary with overall AUC, bias AUCs, and final metric.
    """
    y_true = dataframe[true_label_col].values
    y_pred = dataframe[predicted_label_col].values

    # Compute overall AUC
    overall_auc = compute_auc(y_true, y_pred)

    # Compute bias AUCs for each subgroup
    bias_aucs = []
    for subgroup in subgroups:
        if subgroup in dataframe:
            subgroup_mask = dataframe[subgroup].values > 0.3  # Assuming float subgroup values
            bias_auc = compute_bias_aucs(y_true, y_pred, subgroup_mask)
            bias_aucs.append(bias_auc)

    # Compute final metric
    final_metric = compute_final_metric(overall_auc, bias_aucs)

    return {
        "overall_auc": overall_auc,
        "bias_aucs": bias_aucs,
        "final_metric": final_metric
    }

def merge_train_val(train_df, val_df, id_col, predicted_label_col):
    """
    Merge the train and validation datasets based on the id column.
    
    Args:
        train_df (pd.DataFrame): The train dataset with subgroup information.
        val_df (pd.DataFrame): The validation dataset with predictions.
        id_col (str): The name of the id column common to both datasets.
        predicted_label_col (str): The name of the column in val_df with predicted labels.
    
    Returns:
        pd.DataFrame: A merged DataFrame containing rows from train_df with matching ids in val_df,
                      and an additional column for predicted labels.
    """
    # Merge the two datasets on the id column
    merged_df = train_df.merge(
        val_df[[id_col, predicted_label_col]],
        on=id_col,
        how="inner"  # Keep only rows from train_df that have matching ids in val_df
    )
    return merged_df

# Ensure both IDs are of the same type (e.g., integers or strings)
test_result_data = pd.read_csv("test_predictions_crest.csv")

subgroups = [
    "asian", "atheist", "bisexual", "black", "buddhist", "christian",
    "female", "heterosexual", "hindu", "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability", "jewish", "latino", "male",
    "muslim", "other_disability", "other_gender", "other_race_or_ethnicity",
    "other_religion", "other_sexual_orientation", "physical_disability",
    "psychiatric_or_mental_illness", "transgender", "white"
]
metrics = evaluate_metrics(test_result_data, subgroups, "label", "predicted_label")
print(metrics)
