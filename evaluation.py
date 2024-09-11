import pandas as pd
from typing import List, Set, Dict, Union, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def calculate_metrics(df: pd.DataFrame, 
                      label_col: str = "truth", 
                      pred_col: str = "prediction") -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculates accuracy, precision, recall, and F1-score globally and per class.
    
    Args:
        df (pd.DataFrame): DataFrame containing the true labels and predictions.
        label_col (str): Column name for true labels. Default is "truth".
        pred_col (str): Column name for predicted labels. Default is "prediction".
        
    Returns:
        Dict: A dictionary containing global and per-class metrics.
    """
    # Extract true labels and predictions
    y_true = df[label_col]
    y_pred = df[pred_col]
    
    # Global accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.unique(y_true))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    
    # Metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "per_class": {
            "precision": dict(zip(np.unique(y_true), precision)),
            "recall": dict(zip(np.unique(y_true), recall)),
            "f1": dict(zip(np.unique(y_true), f1))
        }
    }
    
    return metrics


#######################################""

def metrics_to_dataframe(metrics: Dict[str, Union[float, Dict[str, float]]]) -> pd.DataFrame:
    """
    Converts the calculated metrics into a pandas DataFrame for easier saving to Excel.
    
    Args:
        metrics (Dict): Dictionary containing global and per-class metrics.
        
    Returns:
        pd.DataFrame: DataFrame with the metrics in a tabular format.
    """
    # Global metrics (macro/micro averages)
    global_metrics = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)', 
                   'Precision (Micro)', 'Recall (Micro)', 'F1 Score (Micro)'],
        'Score': [
            metrics['accuracy'], 
            metrics['precision_macro'], 
            metrics['recall_macro'], 
            metrics['f1_macro'], 
            metrics['precision_micro'], 
            metrics['recall_micro'], 
            metrics['f1_micro']
        ]
    }
    
    global_df = pd.DataFrame(global_metrics)
    
    # Per-class metrics
    per_class_metrics = []
    for class_label, precision in metrics['per_class']['precision'].items():
        recall = metrics['per_class']['recall'][class_label]
        f1 = metrics['per_class']['f1'][class_label]
        per_class_metrics.append([class_label, precision, recall, f1])
    
    per_class_df = pd.DataFrame(per_class_metrics, columns=['Class', 'Precision', 'Recall', 'F1 Score'])
    
    return global_df, per_class_df

#######################################""

def save_metrics_to_excel(global_df: pd.DataFrame, 
                          per_class_df: pd.DataFrame, 
                          excel_filename: str):
    """
    Saves the global and per-class metrics DataFrames into an Excel file.
    
    Args:
        global_df (pd.DataFrame): DataFrame containing global metrics (accuracy, macro/micro averages).
        per_class_df (pd.DataFrame): DataFrame containing per-class metrics (precision, recall, F1-score).
        excel_filename (str): The filename where the Excel file will be saved.
    """
    with pd.ExcelWriter(excel_filename) as writer:
        global_df.to_excel(writer, sheet_name='Global Metrics', index=False)
        per_class_df.to_excel(writer, sheet_name='Per Class Metrics', index=False)
    print(f"Metrics saved to {excel_filename}")
    
#######################################""    

def plot_confusion_matrix(df: pd.DataFrame, 
                          label_col: str = "truth", 
                          pred_col: str = "prediction", 
                          normalize: bool = False, 
                          save_path: str = "confusion_matrix.png"):
    """
    Plots the confusion matrix using seaborn heatmap and saves it to an image file.
    
    Args:
        df (pd.DataFrame): DataFrame containing the true labels and predictions.
        label_col (str): Column name for true labels. Default is "truth".
        pred_col (str): Column name for predicted labels. Default is "prediction".
        normalize (bool): Whether to normalize the confusion matrix. Default is False.
        save_path (str): The file path where the confusion matrix image will be saved.
    """
    # Extract true labels and predictions
    y_true = df[label_col]
    y_pred = df[pred_col]
    
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.savefig(save_path)
    plt.show()
    # Save the plot to an image file
    
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
#######################################""


def evaluate_predictions(excel_filename: str, 
                         output_excel_filename: str = "performance_metrics.xlsx", 
                         confusion_matrix_image_path: str = "confusion_matrix.png",
                         normalize_confusion_matrix: bool = False):
    """
    Main function to evaluate predictions from an Excel file, calculate metrics, 
    and save the performance metrics into an Excel file.
    
    Args:
        excel_filename (str): The path to the Excel file containing predictions.
        output_excel_filename (str): The path where the metrics will be saved. Default is "performance_metrics.xlsx".
        normalize_confusion_matrix (bool): Whether to normalize the confusion matrix. Default is False.
    """
    # Read predictions from Excel
    df = pd.read_excel(excel_filename)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Convert metrics to DataFrames
    global_df, per_class_df = metrics_to_dataframe(metrics)
    
    # Save metrics to Excel
    save_metrics_to_excel(global_df, per_class_df, output_excel_filename)
    
    # Plot confusion matrix
    plot_confusion_matrix(df, normalize=normalize_confusion_matrix, save_path=confusion_matrix_image_path)
    
    return metrics
