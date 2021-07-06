# Natural Language Processing (OII)


This repository contains the code written for the a course paper in Natural Language Processing taken at the Oxford Internet Institute.


## Analysis & Dataset

For this analysis, I used the arXiv abstracts data set which I downloaded from Kaggle ([link](https://www.kaggle.com/Cornell-University/arxiv)). The data set contains the meta data of scholarly articles from different subdisciplines of physics, computer science, mathematics, statistics, electrical engineering, quantitative biology, and economics. I used this dataset to investigate how scientifig topics and concepts evloved and changed over time. 


## Notebooks


* `Clean_Lemmatise_nGrams_Corpus.ipynb` contains python code to clean and pre-process the text data
* `LDA_with_gensim.ipynb` code to train an LDA topic model with gensim
* `Training_embedding.ipynb` code to train Fasttext word embeddings 
* `Analysing_a_Topic_Fasttext.ipynb` contains python code to analyse the topics detected
* `run_tuned_LDA.py` python code to run LDA training on a server


## Script requirements (versions used)

* jupyter (6.1.4)
* python (3.8.5)
* numpy (1.19.5)
* requests (2.24.0)
* json (2.0.9)
* matplotlib (3.3.2)
* pandas (1.2.4)


#### NLP

* re (2.2.1)
* spacy (2.3.5)
* gensim (3.8.3)
* nltk (3.5)
* sklearn (0.23.2)
* torch (1.7.1)
