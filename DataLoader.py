import os
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
import random

class IntentDataLoader:
    def __init__(self, pos_file: str, neg_file: Optional[str] = None, topic_columns: Optional[List[str]] = None):
        """
        Initialize the IntentDataLoader
        Parameters:
            pos_file (str): Path to the positive examples file (CSV or Excel).
            neg_file (str, optional): Path to the negative examples file (CSV or Excel). Default is None.
            topic_columns (List[str], optional): List of column names to concatenate for the topic. 
                                                 If None, all topic columns will be used.
        """
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.topic_columns = topic_columns
        self.pos_df = None
        self.neg_df = None
        self.classes=[]

        self._validate_files()
        self._load_data()
        self._validate_columns()
    
    def _validate_files(self):
        """Validate that the specified files exist."""
        if not os.path.exists(self.pos_file):
            raise FileNotFoundError(f"The positive examples file '{self.pos_file}' does not exist.")
        if self.neg_file and not os.path.exists(self.neg_file):
            raise FileNotFoundError(f"The negative examples file '{self.neg_file}' does not exist.")

    def _load_data(self):
        """Load the data from the specified files."""
        if self.pos_file.endswith('.csv'):
            self.pos_df = pd.read_csv(self.pos_file)
        else:
            self.pos_df = pd.read_excel(self.pos_file)

        

        if self.neg_file:
            if self.neg_file.endswith('.csv'):
                self.neg_df = pd.read_csv(self.neg_file)
            else:
                self.neg_df = pd.read_excel(self.neg_file)
        else:
            self.neg_df = pd.DataFrame(columns=self.pos_df.columns)

        
    
    def _validate_columns(self):
        """Validate the presence of required columns and consistency between files."""
        if 'text' not in self.pos_df.columns:
            raise ValueError(f"The 'text' column is missing in the positive examples file '{self.pos_file}'.")

        if self.neg_file and 'text' not in self.neg_df.columns:
            raise ValueError(f"The 'text' column is missing in the negative examples file '{self.neg_file}'.")

        if self.neg_file and not self.pos_df.columns.equals(self.neg_df.columns):
            raise ValueError("The columns in the positive and negative files do not match.")

        if self.topic_columns is None:
            self.topic_columns = [col for col in self.pos_df.columns if col.startswith('topic')]
        
        if not self.topic_columns:
            raise ValueError("No columns starting with 'topic' found in the positive examples file.")

        missing_columns = [col for col in self.topic_columns if col not in self.pos_df.columns]
        if missing_columns:
            raise ValueError(f"The following topic columns are missing: {', '.join(missing_columns)}")

    def _create_topic_column(self, df: pd.DataFrame) -> pd.Series:
        """Concatenate the selected topic columns to create a topic label."""
        return df[self.topic_columns].apply(lambda row: ' - '.join(row.values.astype(str)), axis=1)
    
    def get_data(self, 
                 data_type: str = "both", 
                 include_synthetic_negatives: bool = False, 
                 n_negatives: int = 1, 
                 topic_pairs: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str, int]]:
        """
        Retrieve data, optionally including synthetic negative examples.

        Parameters:
            data_type (str): Specify which data to return. Options are "positive", "negative", or "both".
            include_synthetic_negatives (bool): Whether to include synthetic negatives. Default is False.
            n_negatives (int): Number of synthetic negative examples to generate per topic (if applicable).
            topic_pairs (List[Tuple[str, str]], optional): Specific pairs of topics to use for generating negatives.

        Returns:
            List[Tuple[str, str, int]]: A list of examples in the format (text, topic, label).
        """
        data = []
        self.pos_df['topic'] = self._create_topic_column(self.pos_df)
        self.neg_df['topic'] = self._create_topic_column(self.neg_df)

        self.labels = set(self.pos_df['topic'].tolist())

        pos_data = list(zip(self.pos_df['text'], self.pos_df['topic'], [1] * len(self.pos_df)))
        neg_data = list(zip(self.neg_df['text'], self.neg_df['topic'], [0] * len(self.neg_df)))

        

    
        if data_type == "positive":
            data=pos_data
        elif data_type == "negative":
            if self.neg_df is not None:
                data= neg_data
            else:
                data=[]
        else:  # "both"
            if self.neg_df is not None:
                data= pos_data + neg_data
            else:
                data=pos_data

        if include_synthetic_negatives and data_type in ["positive", "both"]:
            synthetic_negatives = self.generate_synthetic_negatives(n=n_negatives, topic_pairs=topic_pairs)
            data.extend(synthetic_negatives)
            
        return data

    def generate_synthetic_negatives(self, n: int = 1, topic_pairs: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str, int]]:
        """
        Generate synthetic negative examples by mixing topics between positive examples.

        Parameters:
            n (int): Maximum number of synthetic negative examples to retain per topic.
            topic_pairs (List[Tuple[str, str]], optional): Specific pairs of topics to use for generating negatives.
                                                          If None, all topics will be randomly paired.

        Returns:
            List[Tuple[str, str, int]]: A list of synthetic negative examples in the format (text, topic, 0).
        """
        synthetic_negatives = defaultdict(list)  # Collect negatives by topic

        pos_data = list(zip(self.pos_df['text'], self.pos_df['topic']))
        topics = self.pos_df['topic'].unique()

        for text, topic in pos_data:
            available_topics = [t for t in topics if t != topic]

            for new_topic in available_topics:
                # If topic_pairs is specified, check if the current pair is valid
                if topic_pairs:
                    if (topic, new_topic) not in topic_pairs and (new_topic, topic) not in topic_pairs:
                        continue

                # Add to synthetic_negatives by topic
                synthetic_negatives[new_topic].append((text, new_topic, 0))

        # Now randomly sample up to `n` examples per topic
        final_synthetic_negatives = []
        for topic, examples in synthetic_negatives.items():
            if len(examples) > n:
                final_synthetic_negatives.extend(random.sample(examples, n))
            else:
                final_synthetic_negatives.extend(examples)

        return final_synthetic_negatives

    def get_data_for_encoder(self, 
                             data_type: str = "both", 
                             include_synthetic_negatives: bool = False, 
                             n_negatives: int = 1, 
                             topic_pairs: Optional[List[Tuple[str, str]]] = None,
                             split: bool = False, 
                             test_size: float = 0.2, 
                             random_state: Optional[int] = 42
                             ) -> Union[List[InputExample], Tuple[List[InputExample], List[InputExample]]]:
        """
        Prepare data in a format compatible with encoders, with an optional train/dev split.
        The train/dev split is performed separately for positive and negative examples with stratification.

        Parameters:
            data_type (str): Specify whether to return 'positive', 'negative', or 'both' types of data.
                             Default is 'both'.
            include_synthetic_negatives (bool): Whether to include synthetic negatives. Default is False.
            n_negatives (int): Number of synthetic negative examples to generate per topic (if applicable).
            topic_pairs (List[Tuple[str, str]], optional): Specific pairs of topics to use for generating negatives.
            split (bool): Whether to split the data into train and dev sets. Default is False.
            test_size (float): Proportion of the data to include in the dev set if splitting. Default is 0.2.
            random_state (Optional[int]): Random state for reproducibility of the split. Default is None.

        Returns:
            Union[List[InputExample], Tuple[List[InputExample], List[InputExample]]]: 
                - If `split` is False: A list of InputExample instances.
                - If `split` is True: A tuple containing two lists of InputExample instances (train and dev sets).
        """
        # Retrieve data
        data = self.get_data(data_type, include_synthetic_negatives, n_negatives, topic_pairs)

        # Convert data to InputExample format
        examples = [InputExample(texts=[text, topic], label=float(label)) for text, topic, label in data]

        # If splitting is not required, return the full dataset
        if not split:
            return examples

        # Separate positive and negative examples
        positive_examples = [example for example in examples if example.label == 1.0]
        negative_examples = [example for example in examples if example.label == 0.0]
        


        # Function to split data and maintain stratification
        def stratified_split(examples):
            texts_labels = [(example.texts, example.label) for example in examples]
            #print(texts_labels)
            #print(zip(*texts_labels))
            #texts, labels = zip(*texts_labels)
            
            texts=[text for text,label in texts_labels]
            labels=[text[1] for text in texts_labels]
            
            
            if len(texts) == 0:
                return [],[],[],[]
            
            else:
            
                return train_test_split(
                    texts, labels, test_size=test_size, stratify=labels, random_state=random_state
                )

        # Split positive and negative examples separately
        pos_train_texts, pos_dev_texts, pos_train_labels, pos_dev_labels = stratified_split(positive_examples)
        

        
        neg_train_texts, neg_dev_texts, neg_train_labels, neg_dev_labels = stratified_split(negative_examples)
          


        # Combine positive and negative train sets
        train_texts = pos_train_texts + neg_train_texts
        train_labels = pos_train_labels + neg_train_labels

        # Combine positive and negative dev sets
        dev_texts = pos_dev_texts + neg_dev_texts
        dev_labels = pos_dev_labels + neg_dev_labels

        # Create InputExample instances for the train and dev sets
        train_examples = [InputExample(texts=text, label=label) for text, label in zip(train_texts, train_labels)]
        dev_examples = [InputExample(texts=text, label=label) for text, label in zip(dev_texts, dev_labels)]

        return train_examples, dev_examples

    
    def get_data_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Generate statistics about the data, including the number of sentences by topic
        for both positive and negative examples.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing two DataFrames:
                                      - 'positive': Statistics for positive examples.
                                      - 'negative': Statistics for negative examples (if available).
        """
        statistics = {}

        # Positive examples statistics
        pos_stats = self.pos_df['topic'].value_counts().reset_index()
        pos_stats.columns = ['Topic', 'Number of Sentences']
        statistics['positive'] = pos_stats

        # Negative examples statistics (if available)
        if self.neg_df is not None:
            neg_stats = self.neg_df['topic'].value_counts().reset_index()
            neg_stats.columns = ['Topic', 'Number of Sentences']
            statistics['negative'] = neg_stats

        return statistics    
    

    def print_data_statistics(self) -> None:
        """
        Print the statistics about the data, including the number of sentences by topic
        for both positive and negative examples.
        """
        stats = self.get_data_statistics()

        print("Positive Examples Statistics:")
        print(stats['positive'].to_string(index=False))
        print("\n")

        if 'negative' in stats:
            print("Negative Examples Statistics:")
            print(stats['negative'].to_string(index=False))
        else:
            print("No negative examples provided.")