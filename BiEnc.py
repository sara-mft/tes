


import math
import logging
import torch
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class BiEnc:
    def __init__(self, model_name: str, train_examples, dev_examples, test_examples):
        """
        Initializes the BiEncoder class for sentence embedding and training.

        Args:
            model_name (str): Pre-trained model name or path.
            train_examples (List[InputExample]): Training examples.
            dev_examples (List[InputExample]): Development (validation) examples.
            test_examples (List[InputExample]): Test examples.
            
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.train_samples = train_examples
        self.dev_samples = dev_examples
        self.test_samples = test_examples

        self.train_scores = []
        self.dev_scores = []
        self.train_loss_per_epoch = []  # List to store average loss per epoch
        self.model_save_path = "best_bi_encoder_model"
    
    def train(self, batch_size: int = 16, num_epochs: int = 4, patience: int = 3, output_path: str = 'fine_tuned_model'):
        """
        Fine-tunes the sentence transformer model with manual early stopping.

        Args:
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs to train.
            output_path (str): Path to save the best model.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        logger.info(f"Fine-tuning started for model: {self.model_name}")
        
        train_dataset = SentencesDataset(self.train_samples, self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        
        train_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(self.train_samples, name="train-dev")

        if self.dev_samples:
            dev_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(self.dev_samples, name="eval-dev")
        else:
            logger.warning("No dev set provided for evaluation. Early stopping will not work without a dev set.")
            return

        # Setup the loss function
        loss_function = losses.CoSENTLoss(model=self.model)
        #Alternatives:  
            #losses.ContrastiveLoss, 
        #OnlineContrastiveLoss
        #CoSENTLoss
            #losses.MarginMSELoss, 
            #SoftmaxLoss(model=self.model, self.model.get_sentence_embedding_dimension(), num_labels=2), 
            #MultipleNegativesRankingLoss
        

        # Calculate warmup steps
        self.warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
        logger.info(f"Warmup steps: {self.warmup_steps}")
        
        best_dev_score = -float('inf')  # Initialize the best score
        epochs_without_improvement = 0  # Count of epochs without improvement
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0  # Initialize epoch loss
            batch_count = 0
            
            # Train the model for 1 epoch, tracking the loss
            self.model.fit(
                train_objectives=[(train_dataloader, loss_function)],
                epochs=1,
                warmup_steps=self.warmup_steps,
                show_progress_bar=True,
                output_path=None  # Don't save after every epoch
            )




            train_score = train_evaluator(self.model)
            dev_score = dev_evaluator(self.model)
            self.train_scores.append(train_score)
            self.dev_scores.append(dev_score)
            logger.info(f"Train score: {train_score}")
            logger.info(f"Dev score: {dev_score}")

            # Early stopping logic
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                epochs_without_improvement = 0
                logger.info("New best dev score, saving the model.")
                # Save the best model
                self.model.save(output_path)
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break  # Stop training if no improvement for 'patience' epochs

        logger.info(f"Training finished. Best model saved to {output_path}")



    
    def train_2(self, batch_size: int = 16, num_epochs: int = 4, patience: int = 3, output_path: str = 'fine_tuned_model'):
        """
        Fine-tunes the sentence transformer model with manual early stopping.

        Args:
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs to train.
            output_path (str): Path to save the best model.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        logger.info(f"Fine-tuning started for model: {self.model_name}")
        
        train_dataset = SentencesDataset(self.train_samples, self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        
        train_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(self.train_samples, name="train-dev")

        if self.dev_samples:
            dev_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(self.dev_samples, name="eval-dev")
            evaluator = evaluation.SequentialEvaluator([train_evaluator, dev_evaluator])
        else:
            logger.warning("No dev set provided for evaluation. Early stopping will not work without a dev set.")
            return

        # Setup the loss function
        loss_function = losses.MultipleNegativesRankingLoss(model=self.model)

        # Calculate warmup steps
        self.warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
        logger.info(f"Warmup steps: {self.warmup_steps}")
        
        best_dev_score = -float('inf')  # Initialize the best score
        epochs_without_improvement = 0  # Count of epochs without improvement
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Train the model for 1 epoch
            self.model.fit(
                train_objectives=[(train_dataloader, loss_function)],
                epochs=1,
                warmup_steps=self.warmup_steps,
                show_progress_bar=True,
                output_path=None  # Don't save after every epoch
            )

            # Evaluate the model on the dev set
            train_score = train_evaluator(self.model)
            dev_score = dev_evaluator(self.model)
            self.train_scores.append(train_score)
            self.dev_scores.append(dev_score)
            logger.info(f"Train score: {train_score}")
            logger.info(f"Dev score: {dev_score}")

            # Early stopping logic
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                epochs_without_improvement = 0
                logger.info("New best dev score, saving the model.")
                # Save the best model
                self.model.save(output_path)
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break  # Stop training if no improvement for 'patience' epochs

        logger.info(f"Training finished. Best model saved to {output_path}")




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




    def evaluate(self, batch_size: int = 16) -> Dict[str, float]:
        """
        Evaluates the model on the test set.

        Args:
            batch_size (int): Batch size for evaluation. Default is 16.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics (e.g., accuracy).
        """
        if not self.test_samples:
            logger.warning("Test samples are not available. Evaluation cannot proceed.")
            return {}

        logger.info("Evaluating model on the test set...")
        
        # Prepare the test dataset and evaluator
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(self.test_samples, name="test")
        test_accuracy = test_evaluator(self.model, output_path=None)

        logger.info(f"Test set accuracy: {test_accuracy}")
        return {"test_accuracy": test_accuracy}

    def get_sentence_embedding(self, sentence: str) -> List[float]:
        """
        Obtains the embedding vector for a single sentence.

        Args:
            sentence (str): Sentence for which to obtain the embedding.

        Returns:
            List[float]: Embedding vector for the input sentence.
        """
        sentence_embedding = self.model.encode([sentence], convert_to_tensor=True, device=self.device)
        return sentence_embedding.cpu().numpy()[0].tolist()

    def get_batch_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """
        Obtains embeddings for a batch of sentences.

        Args:
            sentences (List[str]): List of sentences for which to obtain embeddings.

        Returns:
            List[List[float]]: A list of embedding vectors, one per sentence.
        """
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
        return sentence_embeddings.cpu().numpy().tolist()

    def save_embeddings_to_csv(self, sentences: List[str], output_csv: str = "sentence_embeddings.csv"):
        """
        Saves the sentence embeddings to a CSV file.

        Args:
            sentences (List[str]): List of sentences to generate embeddings for.
            output_csv (str): Path to save the CSV file containing embeddings.
        """
        embeddings = self.get_batch_sentence_embeddings(sentences)
        df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(len(embeddings[0]))])
        df['sentence'] = sentences
        df.to_csv(output_csv, index=False)
        logger.info(f"Embeddings saved to {output_csv}")


