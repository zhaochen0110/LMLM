import csv
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import pandas as pd
from metrics import jsd_timeseries, avg_pairwise_distance_timeseries, entropy_difference_timeseries, js_distance, \
    js_divergence
from cluster import usage_distribution


def gulordava_baroni_correlation(clusterings, usages, jsd_output):
    """
    :param clusterings: dictionary mapping target words to their best clustering model
    :param usages: dictionary mapping target words to a 4-place data structure containing
                   usage vectors, sentences, sentence positions, and time labels
    :param word2shift: dictionary mapping target words to human-rated shift scores
    :return: dataframe containing measurements for all JSD, entropy difference, and average pairwise distance
    """
    dic={}
    for word in tqdm(clusterings):
        _, _, _, t_labels = usages[word]
        clustering = clusterings[word]
        print(clusterings[word])

        # create usage distribution based on clustering results
        if clustering == None:
            dic[word]=1
            continue
        usage_distr = usage_distribution(clustering.labels_, t_labels)
        usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)

        # JSD
        jsd = jsd_timeseries(usage_distr, dfunction=js_divergence) / usage_distr.shape[0]
        dic[word]=jsd
    targets=list(dic.keys())
    targets.sort(key=lambda x: -dic[x])
    # print(targets[:20])
    with open(jsd_output, 'w') as f:
        for ele in targets:
            f.write(ele + '\n')
    #


if __name__ == '__main__':
    # Load usages
    with open('/data1/pivot-extraction/saved_usage/arxiv_09_21.dict', 'rb') as f:
        usages = pickle.load(f)

    # Load clusterings
    with open('/data1/pivot-extraction/saved_usage/arxiv_09_21.clustering.dict', 'rb') as f:
        clusterings = pickle.load(f)


    def format_usages(usages):
        new_usages = {}
        for ele in usages:
            lst = [np.array([x[0] for x in usages[ele]]), [x[1] for x in usages[ele]], [x[2] for x in usages[ele]],
                   np.array([x[3] for x in usages[ele]])]
            new_usages[ele] = lst
        return new_usages


    new_usages = format_usages(usages)
    df = gulordava_baroni_correlation(clusterings, new_usages, 'saved_usage/reverse2009.txt')
