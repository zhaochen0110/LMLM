from word_representation import *
from cluster import *

with open('/data1/pivot-extraction/data/reverse_pivot.csv') as f:
    pivot = [word.strip() for word in f.readlines()]

usages=collect_from_coha(pivot,
                  [2009, 2021],
                  sequence_length=128,
                  file_path=['/data1/pivot-extraction/data/arxiv_2009_part_train.csv',
                             '/data1/pivot-extraction/data/arxiv_2021.csv'],
                  output_path='/data1/pivot-extraction/saved_usage/arxiv_09_21.dict'
                  )

clusterings = obtain_clusterings(
    usages,
    out_path='/data1/pivot-extraction/saved_usage/arxiv_09_21.clustering.dict',
    method='kmeans',
    criterion='silhouette'
)