#%%
from typing import Optional
from pathlib import Path

acos_base_dir = (Path(__name__).parent.parent.parent / 'dataset/ACOS').resolve()

laptop_acos_dir = acos_base_dir / 'Laptop-ACOS'
restaurant_acos_dir = acos_base_dir / 'Restaurant-ACOS'

laptop_acos_train_file = laptop_acos_dir / 'laptop_quad_train.tsv'
laptop_acos_dev_file = laptop_acos_dir / 'laptop_quad_dev.tsv'

#%%

def get_words_from_indices_str(review: list[str], indices_str: str) -> Optional[str]:
    if indices_str == '-1,-1':
        return None
    start_i, end_i = [int(s) for s in indices_str.split(',')]
    return review[start_i: end_i]

def get_sentiment(sentiment_t: str):
    sentiment_i = int(sentiment_t)
    return {0: 'negative', 1: 'neutral', 2: 'positive'}[sentiment_i]


lines_to_read = 10
for line_i, line in enumerate(open(laptop_acos_dev_file)):
    line_items = line.split('\t')
    review: list[str] = line_items[0].strip().split()
    acos_quads: list[list[str]] = [item.strip().split() for item in line_items[1:]]
    print(' '.join(review))
    print(acos_quads)
    for aspect_t, category_t, sentiment_t, opinion_t in acos_quads:
        aspect = get_words_from_indices_str(review, aspect_t)
        category = category_t
        sentiment = get_sentiment(sentiment_t)
        opinion = get_words_from_indices_str(review, opinion_t)
        print(f"aspect: {aspect}, category: {category}, sentiment: {sentiment}, opinion: {opinion}")
    print('=' * 80)
    if line_i >= lines_to_read-1:
        break



# %%
