import os
import pickle

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import BertModel, BertTokenizer, BertConfig
import random
import pandas as pd

def collect_from_coha(target_words,
                      decades,
                      sequence_length=128,
                      pretrained_weights='bert-base-uncased',
                      file_path=[],
                      output_path=None,
                      buffer_size=1024,
                      device='cuda:0'):
    """
    Collect usages of target words from the COHA dataset.

    :param target_words: list of words whose usages are to be collected
    :param decades: [source year, target year]
    :param sequence_length: the number of tokens in the context of a word occurrence
    :param pretrained_weights: path to model folder with weights and config file
    :param file_path: [source file path, target file path]
    :param output_path: path to output file for `usages` dictionary. If provided, data is stored
                        in this file incrementally (use e.g. to avoid out of memory errors)
    :param buffer_size: (max) number of usages to process in a single model run
    :return: usages: a dictionary from target words to lists of usage tuples
                     lemma -> [(vector, sentence, word_position, decade), (v, s, p, d), ...]
    """

    # load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    config = BertConfig(output_hidden_states=True)
    model = BertModel.from_pretrained(pretrained_weights, config=config)
    if torch.cuda.is_available():
        model.to(device)
    # print(model)

    # build word-index vocabulary for target words
    i2w = {}

    for word in target_words:
        i2w[tokenizer.encode(word)[1]] = word

    batch_input_ids = []
    batch_tokens = []
    batch_pos = []
    batch_snippets = []
    batch_decades = []

    usages = defaultdict(list)  # w -> (vector, sentence, word_position, decade)

    # do collection
    for T, decade in enumerate(decades):
        # one time interval at a time
        print('Decade {}...'.format(decade))
        path = file_path[T]

        # read in data, shuffle and choose
        if path.split('.')[-1] == 'txt' or path.split('.')[-1] == 'en':
            with open(path) as f:
                lines = f.readlines()
        elif path.split('.')[-1] == 'csv':
            df = pd.read_csv(path)
            # print(df)
            lines = []
            if len(df.columns)==2:
                lines=list(df['clean_text'])
            else:
                for s1,s2 in zip(list(df['sentence1']),list(df['sentence2'])):
                    lines.append(s1+s2)

        random.shuffle(lines)
        fre = defaultdict(int)

        for i in range(len(lines)):
            L = i
            line = lines[i]

            # tokenize line and convert to token ids
            tokens = tokenizer.encode(line, max_length=sequence_length, padding='max_length', truncation=True)
            # print(tokens)

            pos_lst = []
            tokens_lst = []
            decade_lst = []
            for pos, token in enumerate(tokens):

                # store usage info of target words only
                # only choose 100 sentence for every word
                if token in i2w:
                    if fre[token] > 100:
                        continue
                    fre[token] += 1

                    tokens_lst.append(i2w[token])
                    pos_lst.append(pos)
                    decade_lst.append(decade)

                    # convert later to save storage space
                    # snippet = tokenizer.convert_ids_to_tokens(context_ids)

            # add usage info to buffers


            batch_input_ids.append(tokens) # id
            batch_tokens.append(tokens_lst) # token's id
            batch_pos.append(pos_lst) # token's position
            batch_snippets.append(tokens) # sentence's id
            batch_decades.append(decade_lst) # year

            # if the buffers are full...             or if we're at the end of the dataset
            if (len(batch_input_ids) >= buffer_size) or (L == len(lines) - 1 and T == len(decades) - 1):

                with torch.no_grad():
                    # collect list of input ids into a single batch tensor
                    input_ids_tensor = torch.tensor(batch_input_ids)
                    if torch.cuda.is_available():
                        input_ids_tensor = input_ids_tensor.to(device)
                    print(input_ids_tensor.shape)
                    # print(input_ids_tensor)

                    # run usages through language model
                    outputs = model(input_ids_tensor)
                    # print(outputs.hidden_states.shape)
                    if torch.cuda.is_available():
                        hidden_states = [l.detach().cpu().clone().numpy() for l in outputs.hidden_states]
                    else:
                        hidden_states = [l.clone().numpy() for l in outputs.hidden_states]

                    # print(hidden_states.)
                    # get usage vectors from hidden states
                    hidden_states = np.stack(hidden_states)  # (13, B, |s|, 768)
                    print(hidden_states.shape)
                    # print('Expected hidden states size: (13, B, |s|, 768). Got {}'.format(hidden_states.shape))
                    # usage_vectors = np.sum(hidden_states, 0)  # (B, |s|, 768)
                    # usage_vectors = hidden_states.view(hidden_states.shape[1],
                    #                                    hidden_states.shape[2],
                    #                                    -1)
                    usage_vectors = np.sum(hidden_states[1:, :, :, :], axis=0)
                    # usage_vectors = hidden_states.reshape((hidden_states.shape[1], hidden_states.shape[2], -1))

                if output_path and os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        usages = pickle.load(f)

                # store usage tuples in a dictionary: lemma -> (vector, snippet, position, decade)
                for b in np.arange(len(batch_input_ids)):
                    for i in np.arange(len(batch_pos[b])):
                        # print(batch_tokens[b][i])
                        usage_vector = usage_vectors[b, batch_pos[b][i], :]
                        usages[batch_tokens[b][i]].append(
                            (usage_vector, batch_snippets[b][i], batch_pos[b][i], batch_decades[b][i]))
                # print(usages.keys())
                # finally, empty the batch buffers
                batch_input_ids, batch_tokens, batch_pos, batch_snippets, batch_decades = [], [], [], [], []

                # and store data incrementally
                if output_path:
                    with open(output_path, 'wb') as f:
                        pickle.dump(usages, file=f)
    # print(usages)

    return usages


