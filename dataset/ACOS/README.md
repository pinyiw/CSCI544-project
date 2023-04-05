Aspect-Category-Opinion-Sentiment (ACOS) Quadruple Extraction
==============================================================

* Each line consists of review text and its quadruples. Each quadruple is formalized as 'Aspect Category Sentiment Opinion'. The 0, 1, 2 in the Sentiment category represents negative, neutral, and positive, respectively.

* Data acquired from: https://github.com/NUSTM/ACOS

## Citation
If you use the data and code in your research, please cite our paper as follows:
```
@inproceedings{cai2021aspect,
  title={Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions},
  author={Cai, Hongjie and Xia, Rui and Yu, Jianfei},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={340--350},
  year={2021}
}
```

## Dependencies
* Python 3.11
* [dataset/ACOS/requirements.txt](/dataset/ACOS/requirements.txt)
* Add OpenAI keys to [.env](/.env)
  ```
  OPENAI_KEY=
  OPENAI_ORG_ID=
  ```

## OpenAI few shots learning with laptop train on laptop dev data
* Using 5 training examples
  ```
  Total tokens used: 128,340
  Credit used: 2.57
  Total reviews: 326

  Precision: 0.382075
  Recall: 0.368182
  F1: 0.375000
  ```
* Using 10 training examples
  ```
  Total tokens used: 205,704
  Credit used: 4.11
  Total reviews: 326
  
  Using 10 training examples
  Precision: 0.468254
  Recall: 0.402273
  F1: 0.432763
  ```
