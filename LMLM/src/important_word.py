import yake
import pandas as pd
import random
import re
from extract_word import get_top_word
from sklearn.feature_extraction.text import CountVectorizer
# def get_MI_word(file_path, save_path, topk_freq):

def get_frequency_word(file_path, save_path, topk_freq):
    frequency_dict = {}
    if file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path)
        text = ''
        s1 = list(df['sentence1'])
        s2 = list(df['sentence2'])
        for i in range(len(s1)):
            text = s1[i] + s2[i]
            text_str = text.split()
            for word in text_str:
                word = re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", word)
                frequency_dict[word] = frequency_dict.get(word, 0) + 1
    frequency_word_index = sorted(frequency_dict.keys(), key=lambda x: frequency_dict[x], reverse=True)[:topk_freq]
    print(frequency_word_index)
    with open(save_path, 'w') as f:
        for kw in frequency_word_index:
            f.write(kw + '\n')


def get_important_word(file_path, save_path):
    if file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path)
        text = ''
        s1 = list(df['sentence1'])
        s2 = list(df['sentence2'])
        for i in range(len(s1)):
            text += s1[i] + s2[i]
    else:
        text = ''
        with open(file_path) as f:
            for ele in f.readlines():
                text += ele

    language = "en"
    max_ngram_size = 1
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 2000
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                         dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                         features=None)
    keywords = kw_extractor.extract_keywords(text)

    # for kw in keywords:
    #     print(kw)
    with open(save_path, 'w') as f:
        for kw in keywords:
            f.write(kw[0] + '\n')


if __name__ == '__main__':
    get_frequency_word('data/arxiv/arxiv_2019.csv',
                       'data/arxiv/arxiv_word/frequency_word.csv', 2000)
    get_top_word('data/arxiv_word/2019/frequency_word.csv', 'data/arxiv_word/2019/extract_word.csv')
    print('finish!')
