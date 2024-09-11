import math
import logging
from datetime import datetime
from typing import List, Set, Dict, Union, Tuple
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample, evaluation
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CrossEnc:
    def __init__(self, model_name: str, 
                 train_examples: List[InputExample], 
                 dev_examples: List[InputExample], 
                 test_examples: List[InputExample], 
                 num_labels: int = 1,
                 model_save_path: str = None,
                 train_batch_size: int = 16, 
                 num_epochs: int = 2, 
                 patience: int = 2):
        """
        CrossEncoder model for intent classification with training, evaluation, and prediction methods.

        Args:
            model_name (str): Name of the pre-trained model (e.g., 'camembert-large').
            train_examples (List[InputExample]): Training examples for the model.
            dev_examples (List[InputExample]): Development/validation examples for evaluation.
            test_examples (List[InputExample]): Test examples for evaluation.
            num_labels (int): Number of labels to classify. Default is 1.
            model_save_path (str): Path to save the best model. Defaults to a time-stamped folder.
            train_batch_size (int): Batch size during training. Default is 16.
            num_epochs (int): Number of training epochs. Default is 2.
            patience (int): Early stopping patience (number of epochs without improvement). Default is 2.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.train_batch_size = train_batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.warmup_steps = 0
        self.train_scores = []
        self.dev_scores = []

        # Path for saving the model
        self.model_save_path = model_save_path or f"models/CrossEncoder-{model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # Prepare datasets
        self.train_samples = train_examples
        self.dev_samples = dev_examples
        self.test_samples = test_examples

        # Initialize the model
        self.model = CrossEncoder(model_name, num_labels=num_labels)




    def train(self):
        """Train the CrossEncoder model with early stopping."""

        train_dataloader = DataLoader(self.train_samples, shuffle=True, batch_size=self.train_batch_size)
        dev_evaluator = CECorrelationEvaluator.from_input_examples(self.dev_samples, name="eval-dev")
        train_evaluator = CECorrelationEvaluator.from_input_examples(self.train_samples, name="eval-train")
        evaluator = evaluation.SequentialEvaluator([train_evaluator, dev_evaluator])


        self.warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1) 
        logger.info("Warmup-steps: {}".format(self.warmup_steps))
        
        best_dev_score = -float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            self.model.fit(
                train_dataloader=train_dataloader,
                epochs=1,
                warmup_steps=self.warmup_steps,
                output_path=self.model_save_path,
            )
            train_score = train_evaluator(self.model)
            dev_score = dev_evaluator(self.model)
            self.train_scores.append(train_score)
            self.dev_scores.append(dev_score)
            logger.info(f"Train score: {train_score}")
            logger.info(f"Dev score: {dev_score}")
            

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                epochs_without_improvement = 0
                # Optionally, save the best model
                self.model.save(self.model_save_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
    def plot_performance(self):
        """Plot the training and development scores across epochs."""

        epochs = list(range(1,len(self.train_scores) + 1))
        
        train_scores = [score for score in self.train_scores]
        dev_scores = [score for score in self.dev_scores]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, label='Train Score')
        plt.plot(epochs, dev_scores, label='Dev Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score: Spearman correlation')
        plt.title('Training and Development Score per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
        



    def predict(self, 
                texts: List[str], 
                true_labels: Union[List[str], None], 
                classes: Set[str], 
                threshold: float = 0.5, 
                out_of_scope_class: str = "None", 
                calculate_perf: bool = True,
                save_to_excel: bool = False,
                excel_filename: str = "results.xlsx") -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, float]]:
        """
        Predicts the top-k classes for a list of input texts and evaluates the performance (if requested).
        Optionally saves the results to an Excel file.

        Args:
            texts (List[str]): A list of input text samples to classify.
            true_labels (Union[List[str], None]): The ground-truth labels corresponding to each input text. If `None`, defaults to 'Unknown'.
            classes (Set[str]): A set of class labels the model can predict from.
            threshold (float, optional): The confidence threshold above which a prediction is considered valid. Defaults to 0.5.
            out_of_scope_class (str, optional): Label assigned if the highest prediction score is below the threshold. Defaults to "None".
            calculate_perf (bool, optional): Whether to calculate and return top-k performance metrics. Defaults to True.
            save_to_excel (bool, optional): Whether to save the prediction results to an Excel file. Defaults to False.
            excel_filename (str, optional): The filename for the Excel file. Defaults to "results.xlsx".

        Returns:
            Tuple: 
                - List[Dict[str, Union[str, float]]]: A list of dictionaries where each dictionary contains the text, predicted class, true label, 
                                                    top-3 predicted classes, and their corresponding scores.
                - Dict[str, float]: A dictionary containing top-1, top-2, and top-3 accuracy values if `calculate_perf` is True.
        """
        
        # Initialize top-k accuracy tracker and results
        top_k_accuracies = {'top-1': [], 'top-2': [], 'top-3': []}
        results = []

        # Verify that if calculating performance, `true_labels` should not be empty and should match the size of `texts`
        if calculate_perf:
            assert len(true_labels) == len(texts), "Length of true_labels and texts must be the same when calculating performance."

        # If true_labels is None or an empty list, fill it with "Unknown" for the same size as `texts`
        if len(true_labels)<len(texts) or not true_labels or true_labels is None:
            true_labels = ["Unknown"] * len(texts)

        # Iterate through each text and true label
        for text, true_label in zip(texts, true_labels):
            scores = []

            # Predict scores for all possible classes
            for label in classes:
                score = self.model.predict([(text, str(label))])[0]
                scores.append((str(label), score))

            # Sort scores in descending order
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            # Select prediction based on threshold
            if scores[0][1] >= threshold:
                prediction = scores[0][0]
            else:
                prediction = out_of_scope_class

            # Append result for this example
            results.append({
                "text": text,
                "truth": true_label,
                "prediction": prediction,
                "top-1": scores[0][0],
                "top-2": scores[1][0] if len(scores) > 1 else None,
                "top-3": scores[2][0] if len(scores) > 2 else None,
                "score-top-1": scores[0][1],
                "score-top-2": scores[1][1] if len(scores) > 1 else None,
                "score-top-3": scores[2][1] if len(scores) > 2 else None
            })

            # Calculate top-k accuracy if requested
            if calculate_perf:
                if true_label in classes:
                    top_k_accuracies['top-1'].append(int(scores[0][0] == true_label))
                    top_k_accuracies['top-2'].append(int(true_label in [scores[0][0], scores[1][0]]))
                    top_k_accuracies['top-3'].append(int(true_label in [scores[0][0], scores[1][0], scores[2][0] if len(scores) > 2 else None]))
                else:
                    print(f"Warning: The true label '{true_label}' for text '{text}' is not in the defined classes. It will not be counted towards performance.")

        # Compute top-k accuracies
        if calculate_perf:
            top_k_accuracies = {k: sum(v) / len(v) for k, v in top_k_accuracies.items() if len(v) > 0}

        # Save results to Excel if requested
        if save_to_excel:
            df = pd.DataFrame(results)
            df.to_excel(excel_filename, index=False)
            print(f"Results saved to {excel_filename}")

        return results, top_k_accuracies
    
