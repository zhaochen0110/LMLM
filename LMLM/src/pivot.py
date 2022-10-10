import pandas as pd
from collections import defaultdict


def get_fre(file):
    dic = defaultdict(int)
    with open(file) as f:
        for line in f.readlines():
            lst = line.split()
            for ele in lst:
                dic[ele] += 1
    return dic


def get_pivot_word(important_words, unlabel_path,pivot_path):
    with open(important_words) as f:
        word_lst = [word.strip() for word in f.readlines()]

    fre = get_fre(unlabel_path)

    cnt = 0
    res = []
    for word in word_lst:
        if fre[word] > 100:
            res.append(word)
            cnt += 1

    with open(pivot_path, 'w') as f:
        for word in res:
            f.write(word + '\n')