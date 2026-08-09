#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

from sentence_transformers import evaluation
from typing import Tuple, Generator, List
import json
import os
import glob
import random
import itertools
from datasets import Dataset, concatenate_datasets
from ruamel.yaml import CommentedMap

def yield_data(filename: str) -> Generator[Tuple[int, str], None, None]:
    """Generator of tuples of label, text

    Args:
        filename (str): input file

    Yields:
        Generator[Tuple[int, str], None, None]: Generator of tuples of label and text
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # Assign numerical labels based on subject
            label = 0
            if "Physics" == data["label"]:
                label = 1
            elif "History" == data["label"]:
                label = 2
                
            yield label, data["text"]

def get_train_test_lists(cfg: CommentedMap, split: float = 0.8) -> Tuple[List[Tuple[int, str]]]:
    """Get the train and test lists

    Args:
        cfg (CommentedMap): configuration read from config file. 
        split (float): split ratio for train and test data

    Returns:
        Tuple[List[Tuple[int, str]]]: The train and test lists of tuples containing label and text
    """
    data_dir = cfg["paths"]["data_dir"]

    # Use glob to find all JSONL files in the chunks directory
    jsonl_files = glob.glob(os.path.join(f"{data_dir}/chunks", "*.json"))

    # Collect examples from all JSONL files
    examples = [(label, text) for jsonl_file in jsonl_files for label, text in yield_data(jsonl_file)]
    examples = [example for example in examples if len(example[1]) > 100]
    random.shuffle(examples)
    train_size = int(len(examples) * split)
    train_list = examples[:train_size]
    test_list = examples[train_size:]
    return train_list, test_list

def tuples_list_to_dataset(tuples_list: List[Tuple[int, str]]) -> Dataset:
    """Converts list of tuples of label, text to Hugging face dataset

    Args:
        tuples_list (List[Tuple[int, str]]): A list of tuples containing subject label along with the text

    Returns:
        Dataset: The hugging face dataset with columns "label" and "text" with these two values
    """
    # Separate the labels and texts
    labels, texts = zip(*tuples_list)
    # Create a dictionary
    data_dict = {"label": labels, "text": texts}
    # Convert to Dataset
    return Dataset.from_dict(data_dict)

def get_evaluator(test_dataset: Dataset, truncate_dim: int = None) -> evaluation.SentenceEvaluator:
    """Gets the embeddings similarity evaluator from a sample of test dataset

    Args:
        test_dataset (Dataset): Test dataset 
        truncate_dim (int): For use with Matryoshka loss training - defaults to None

    Returns:
        evaluation.SentenceEvaluator: The evaluator returned
    """
    # Sample 5000 random entries from the test_dataset
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), 5000)
    sampled_dataset = test_dataset.select(sample_indices)
    sentences1 = sampled_dataset["sentence1"]
    sentences2 = sampled_dataset["sentence2"]
    labels = sampled_dataset["label"]

    # Set up the evaluator with binary similarity scores
    return evaluation.BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=labels,
        truncate_dim=truncate_dim,
    )

def convert_to_pair_dataset(sentence_label_dataset: Dataset) -> Dataset:
    """Generate pair dataset from sentence-label dataset

    Args:
        sentence_label_dataset (Dataset): sentence to class

    Returns:
        Dataset: pairs of sentences with a label indicating if they are same or different
    """
    sentences = sentence_label_dataset["text"]
    labels = sentence_label_dataset["label"]
    sentence_pair_dataset = {"sentence1":[], "sentence2":[], "label":[]}
    for (i, sent1), (j, sent2) in itertools.combinations(enumerate(sentences), 2):
        # If labels are the same, it's a similar pair (1), else dissimilar (0)
        score = 1 if labels[i] == labels[j] else 0
        sentence_pair_dataset["sentence1"].append(sent1)
        sentence_pair_dataset["sentence2"].append(sent2)
        sentence_pair_dataset["label"].append(score)  
    
    return Dataset.from_dict(sentence_pair_dataset) 

def sampled_dataset(dataset: Dataset) -> Dataset:
    """Sample incoming dataset to max of 500 per label

    Args:
        dataset (Dataset): incoming dataset

    Returns:
        Dataset: sampled dataset containing 500 per label
    """
    # Initialize an empty list to hold sampled subsets
    samples = []

    # Loop through each label (assuming labels are 0, 1, and 2)
    for label in set(dataset['label']):
        # Filter rows with the current label
        label_subset = dataset.filter(lambda x, label=label : x['label'] == label)
        # Shuffle and select the first 500 samples (or fewer if less are available)
        label_sample = label_subset.shuffle(seed=42).select(range(min(500, len(label_subset))))
        # Append to samples list
        samples.append(label_sample)
    
    return concatenate_datasets(samples)  