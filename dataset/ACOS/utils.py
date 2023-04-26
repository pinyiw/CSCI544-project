from __future__ import annotations
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from textwrap import dedent
import json
import time
import openai

acos_base_dir = Path(__file__).parent.resolve()

laptop_acos_dir = acos_base_dir / 'Laptop-ACOS'
restaurant_acos_dir = acos_base_dir / 'Restaurant-ACOS'

laptop_acos_train_file = laptop_acos_dir / 'laptop_quad_train.tsv'
laptop_acos_dev_file = laptop_acos_dir / 'laptop_quad_dev.tsv'

restaurant_acos_train_file = restaurant_acos_dir / 'rest16_quad_train.tsv'
restaurant_acos_dev_file = restaurant_acos_dir / 'rest16_quad_dev.tsv'


def get_words_from_indices_str(review: list[str], indices_str: str) -> Optional[list[str]]:
    if indices_str == '-1,-1':
        return None
    start_i, end_i = [int(s) for s in indices_str.split(',')]
    return review[start_i: end_i]

def get_sentiment(sentiment_t: str):
    sentiment_i = int(sentiment_t)
    return {0: 'negative', 1: 'neutral', 2: 'positive'}[sentiment_i]

def find_sublist_indices(l: list, sublist: Optional[list]) -> Optional[tuple[int, int]]:
    if sublist is None:
        return -1, -1
    for idx in range(len(l) - len(sublist) + 1):
        if l[idx: idx + len(sublist)] == sublist:
            return idx, idx + len(sublist)
    return None



@dataclass
class ACOSQuad:
    aspect: Optional[list[str]]
    category: str
    opinion: Optional[list[str]]
    sentiment: str

    def __eq__(self, other: ACOSQuad) -> bool:
        return (
            self.aspect == other.aspect 
            # and self.category == other.category 
            and self.opinion == other.opinion 
            and self.sentiment == other.sentiment
        )

    def to_acos_str(self, review: str) -> str:
        review_words: list[str] = review.split()
        # A
        aspect_indices = find_sublist_indices(review_words, self.aspect)
        if aspect_indices is None:
            raise ValueError(f"Aspect not found in review: {self.aspect} {review}")
        aspect_str = f'{aspect_indices[0]},{aspect_indices[1]}'
        # C
        category_str = self.category
        # O
        opinion_indices = find_sublist_indices(review_words, self.opinion)
        if opinion_indices is None:
            raise ValueError(f"Opinion not found in review: {self.opinion} {review}")
        opinion_str = f'{opinion_indices[0]},{opinion_indices[1]}'
        # S
        sent_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
        sentiment_str = sent_to_idx[self.sentiment]
        return f'{aspect_str} {category_str} {sentiment_str} {opinion_str}'


@dataclass
class ACOSReview:
    review: str
    acos_quads: list[ACOSQuad]

    @classmethod
    def from_pred_str(cls, review: str, pred_str: str) -> ACOSReview:
        pred_str = pred_str.strip()
        acos_quads: list[ACOSQuad] = []
        for pred_quad in pred_str.split('\n\n'):
            aspect_str, cat_str, opinion_str, sent_str = pred_quad.split('\n')

            aspect: Optional[list[str]] = aspect_str[len('Aspect: '):].strip().split()
            aspect = aspect if aspect != ['None'] else None

            category: str = cat_str[len('Category: '):].strip()

            opinion: Optional[list[str]] = opinion_str[len('Opinion: '):].strip().split()
            opinion = opinion if opinion != ['None'] else None

            sentiment: str = sent_str[len('Sentiment: '):].strip()

            acos_quad = ACOSQuad(aspect, category, opinion, sentiment)
            acos_quads.append(acos_quad)
        return cls(review, acos_quads)

    def generate_acos_prompt(self) -> str:
        double_newline = '\n\n'
        acos_prompts = []
        for acos_quad in self.acos_quads:
            prompt = dedent(f"""\
                Aspect: {' '.join(a) if (a := acos_quad.aspect) else 'None'}
                Category: {acos_quad.category}
                Opinion: {' '.join(o) if (o := acos_quad.opinion) else 'None'}
                Sentiment: {acos_quad.sentiment}""")
            acos_prompts.append(prompt)
        return f"Input review:\n{self.review}\n\n{double_newline.join(acos_prompts)}"


@dataclass
class ACOSPrediction:
    review: str
    acos_preds: list[ACOSQuad]
    acos_targets: Optional[list[ACOSQuad]]

    @classmethod
    def from_pred_str(cls, review: list[str], pred_str: str, acos_targets: Optional[list[ACOSQuad]] = None) -> ACOSPrediction:
        extracted_acos_review = ACOSReview.from_pred_str(review, pred_str)
        return cls(review, extracted_acos_review.acos_quads, acos_targets)
    
    @classmethod
    def from_dict(cls, d) -> ACOSPrediction:
        return cls(
            d['review'],
            [ACOSQuad(**quad) for quad in d['acos_preds']],
            [ACOSQuad(**quad) for quad in d['acos_targets']] if d['acos_targets'] is not None else None
        )


PRINT_LINES = False
def load_acos_reviews(acos_file: Path, lines_to_read: int) -> list[ACOSReview]:
    acos_reviews: list[ACOSReview] = []
    for line_i, line in enumerate(open(acos_file)):
        line_items = line.split('\t')
        review: list[str] = line_items[0].strip().split()
        acos_quad_strs: list[list[str]] = [item.strip().split() for item in line_items[1:]]
        if PRINT_LINES:
            print(' '.join(review))
            print(acos_quad_strs)
        acos_quads: list[ACOSQuad] = []
        for aspect_t, category_t, sentiment_t, opinion_t in acos_quad_strs:
            aspect: Optional[list[str]] = get_words_from_indices_str(review, aspect_t)
            category = category_t.split('#')[-1]
            opinion: Optional[list[str]] = get_words_from_indices_str(review, opinion_t)
            sentiment = get_sentiment(sentiment_t)
            acos_quad = ACOSQuad(aspect, category, opinion, sentiment)
            acos_quads.append(acos_quad)
            if PRINT_LINES:
                print(acos_quad)
        if PRINT_LINES:
            print('=' * 80)
        acos_reviews.append(ACOSReview(' '.join(review), acos_quads))
        if lines_to_read >= 0 and line_i >= lines_to_read-1:
            break
    return acos_reviews


def generate_acos_pred_prompts(acos_review_examples: list[ACOSReview], review_prompt: str) -> str:
    example_prompts: list[str] = [acos_review.generate_acos_prompt() for acos_review in acos_review_examples]
    example_prompts_sep = '\n\n##\n\n'
    task_prompt = '''Extract all Aspect, Category, Opinion, and Sentiment quadruples from the product review. Aspect has to be a substring of the input reivew or "None". Opinion has to be a substring of the input reivew or "None":'''
    example_prompts.append(f"Input review:\n{review_prompt}")
    return f"{task_prompt}\n\n{example_prompts_sep.join(example_prompts)}\n\n\n"


def openai_pred_acos(acos_reviews: list[ACOSReview], prompt_review_examples: list[ACOSReview], openai_model: str) -> list[ACOSPrediction]:
    '''
    Predicts ACOS quadruples for each review in acos_reviews using OpenAI's API.
    Please set openai.api_key and openai.organization before calling this function.
    '''
    acos_preds: list[ACOSPrediction] = []
    total_tokens_used = 0
    for i, acos_review in enumerate(acos_reviews):
        try:
            prompt = generate_acos_pred_prompts(prompt_review_examples, acos_review.review)

            resp = openai.Completion.create(
                model=openai_model,
                prompt=prompt,
                temperature=0,
                max_tokens=1000
            )

            pred_str = resp['choices'][0]['text']

            acos_pred = ACOSPrediction.from_pred_str(acos_review.review, pred_str, acos_review.acos_quads)
            acos_preds.append(acos_pred)
            total_tokens_used += resp['usage']['total_tokens']
        except:
            print(f"{i=} {acos_review=}")
            raise
        # try:
        #     # trigger error early if prediction format is wrong
        #     [quad_pred.to_acos_str(acos_pred.review) for quad_pred in acos_pred.acos_preds]
        # except:
        #     print(i)
        #     raise

        # sleep for 60s to avoid rate limit
        if i != 0 and i % 60 == 0:
            print(f"Completed {i} reviews")
            time.sleep(60)
    print(f"Total tokens used: {total_tokens_used:,d}")
    print(f"Credit used: {total_tokens_used / 1000 * 0.02:.2f}")
    print(f"Total reviews: {len(acos_preds)}")
    return acos_preds


def evaluate_acos_preds(acos_preds: list[ACOSPrediction]) -> tuple[float, float, float]:
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for acos_pred in acos_preds:
        for acos_pred_quad in acos_pred.acos_preds:
            if acos_pred_quad in acos_pred.acos_targets:
                true_pos += 1
            else:
                false_pos += 1
        for acos_target_quad in acos_pred.acos_targets:
            if acos_target_quad not in acos_pred.acos_preds:
                false_neg += 1

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1
