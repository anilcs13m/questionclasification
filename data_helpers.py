import re
import os
import time
import datetime
import itertools
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from collections import Counter
from tensorflow.contrib import learn

"""
Reference https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
"""

def clean_str(string):
    """
    Reference from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().lower() 
    string = re.sub("'s","is",string)
    string = re.sub("n't","not",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\ ]", " ", string)
    string = re.sub("what's","what is",string)
    # string = re.sub("'s","is",string)
    string = re.sub("``", " ",string)
    string = re.sub("where 's","where is",string)
    string = re.sub("what time","when",string)
    string = re.sub("don't","do not",string)
    string = re.sub("doesn't","does not",string)
    string = re.sub("n't"," not",string)
    string = re.sub("u s","your",string)
    string = re.sub("in what","what",string)
    string = re.sub("in which","which",string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string

def load_data_and_labels():
    """
    Splits the data into words and generates labels.
    """
    who_examples = []
    when_examples = []
    what_examples = []
    affi_example = []
    Unknow = []
    dataset = list(open("train_1000.label", "r").readlines())
    dataset = [s.strip() for s in dataset]
    dataset = [clean_str(sent) for sent in dataset]
    x_text = [s.split(" ") for s in dataset]
    dataset = [s[2:] for s in x_text]
    dataset_clean = []
    for s in dataset:
        dataset_clean.append( [' '.join(ww for ww in s)])

    for s in dataset_clean:
        for w in s:
            if 'who' in w:
                who_examples.append(w)
            elif 'what' in w:
                what_examples.append(w)
            elif 'when' in w:
                when_examples.append(w)
            elif 'not' in w and (not 'what' in w or not 'who' in w or not 'when' in w):
                Unknow.append(w)
            else:
                affi_example.append(w)
                
    x_text = who_examples + when_examples + what_examples + affi_example + Unknow
    x_text = [clean_str(sent) for sent in x_text] 
    x_text = [s.split(" ") for s in x_text] 
    who_labels =  [[1, 0, 0, 0] for _ in who_examples]
    when_labels = [[0, 1, 0, 0] for _ in when_examples]
    what_labels = [[0, 0, 1, 0] for _ in what_examples]
    affi_labels = [[0, 0, 0, 1] for _ in affi_example]
    
    y = np.concatenate([who_labels, when_labels,what_labels,affi_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. 
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    """
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]  
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  
    return [vocabulary, vocabulary_inv]

def load_data():
    """
    Loads and preprocessed data for dataset.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels() 
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = np.array([[vocabulary[word] for word in sent] for sent in sentences_padded])
    y = np.array(labels)
    return [x, y, vocabulary, vocabulary_inv] 

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
