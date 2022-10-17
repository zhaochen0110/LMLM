from word_representation import *
from cluster import *

with open('data/arxiv_word/extract_word.csv') as f:
    pivot = [word.strip() for word in f.readlines()]

usages=collect_from_coha(pivot,
                  [2009, 2021],
                  sequence_length=128,
                  file_path=['data/arxiv/arxiv_2009_train.csv',
                             'data/arxiv/arxiv_2021.csv'],
                  output_path='data/arxiv/arxiv_word/arxiv_09_21.dict'
                  )

clusterings = obtain_clusterings(
    usages,
    out_path='data/arxiv/arxiv_word/arxiv_09_21.clustering.dict',
    method='kmeans',
    criterion='silhouette'
)