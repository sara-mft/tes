import os
from typing import List
import re
from collections import defaultdict




def create_matching_dict(lista, listb):
    matching_dict = defaultdict(list)

    for sentence_a, sentence_b in zip(lista, listb):
        matching_dict[sentence_a].append(sentence_b)

    return matching_dict


def adapt_file_path(file_path:str) -> str:
    """Adapts a file path based on the current operating system (Windows or Linux).

    Args:
        file_path (str): The original file path.

    Returns:
        str: The adapted file path for the current operating system.
    """
    current_os = os.name

    if current_os == 'nt':
        adapted_file_path = file_path.replace('/', '\\')
    else:
        adapted_file_path = file_path.replace('\\', '/')

    return adapted_file_path




def remove_special_characters(sentence):

    pattern = r'[^a-zA-Z0-9\s]'  # Matches any character that is not alphanumeric or whitespace

    # Remove special characters from the sentence
    cleaned_sentence = re.sub(pattern, '', sentence)

    return cleaned_sentence

def divide_last_four_elements(data: tuple) -> list:
    """
    Divides the last 4 elements of a tuple by the first element and returns a list of the results.

    Args:
        data (tuple): A tuple of 5 values.

    Returns:
        list: A list of the last 4 elements divided by the first element.
    """
    first_value = data[0]
    last_four_values = data[1:]

    result = [(value / first_value)*100 for value in last_four_values]
    return result