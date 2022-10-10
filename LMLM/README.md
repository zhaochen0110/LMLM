## BASELINE
/data1/szc/LMLM/baseline/random_pretrain.py

即为TADA的代码

/data1/szc/LMLM/baseline/understand_mlm.sh

进行analysis的

## 提取关键词

src/important_word.py

```python
from important_word import get_important_word

get_important_word(SOURCE_DATA_PATH, SAVE_IMPORTANT_WORD_PATH) 
```

## 提取pivot

src/pivot.py

```python
get_pivot_word(IMPORTANT_WORD_PATH, UNLABEL_PATH, PIVOT_SAVE_PATH)
```

## 计算word representation

src/word_representation.py

```python
with open('PIVOT PATH') as f:
    pivot = [word.strip() for word in f.readlines()]

collect_from_coha(pivot,
                  [SOURCE_YEAR, TARGET_YEAR],
                  sequence_length=128,
                  file_path=[SOURCE_LABEL_DATA,
                             TARGET_UNLABEL_DATA],
                  output_path='WORD REPRESENTATION SAVE PATH'
                  )
```

## 计算聚类

src/cluster.py

```python
with open( WORD_REPRESENTATION_PATH, 'rb') as f:
    usages = pickle.load(f)

clusterings = obtain_clusterings(
    usages,
    out_path=CLUSTER_SAVE_PATH,
    method='kmeans',
    criterion='silhouette'
)
```

## 计算JSD

src/jsd.py

```python
# Load usages
with open(WORD_REPRESENTATION_PATH, 'rb') as f:
    usages = pickle.load(f)

# Load clusterings
with open(CLUSTER_PATH, 'rb') as f:
    clusterings = pickle.load(f)


def format_usages(usages):
    new_usages = {}
    for ele in usages:
        lst = [np.array([x[0] for x in usages[ele]]), [x[1] for x in usages[ele]], [x[2] for x in usages[ele]],
               np.array([x[3] for x in usages[ele]])]
        new_usages[ele] = lst
    return new_usages


new_usages = format_usages(usages)
df = gulordava_baroni_correlation(clusterings, new_usages, JSD_SAVE_PATH)

```

## pivot训练

sh/test_mlm.sh