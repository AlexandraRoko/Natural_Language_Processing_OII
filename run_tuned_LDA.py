from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import TfidfModel
import time
import json
import pandas as pd
from collections import OrderedDict
import csv
import numpy as np



#with open('../../data/written/data_ready.txt', 'r') as infile:
with open('./data_ready_concat_POSTag_all_sorted.txt', 'r') as infile:
    data_ready_concat = json.load(infile)
infile.close()

print("Data loaded")

# Create a dictionary representation of the documents.
dictionary = Dictionary(data_ready_concat)

# Filter out words that occur in less than 20 documents, or more than 50% of the documents.
#dictionary.filter_extremes(no_below=20, no_above=0.5)
dictionary.filter_extremes(no_below=20, no_above=0.4)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in data_ready_concat]

#tfidf = TfidfModel(corpus)
#corpus = tfidf[corpus]

print("Dictionary ready. Training model now.")

# Set training parameters.
num_topics = 100
chunksize = 12000
passes = 25
iterations = 1000
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

start_time = time.time()

lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')

# Save model to disk.
lda_model.save("./lda_model_concat_POSTag_self_tuned_all_sorted")

print("Normal model saved. DONE.")





###### Second model #######



tfidf = TfidfModel(corpus)
corpus = tfidf[corpus]


print("Tfidf dictionary ready")

# Set training parameters.
num_topics = 100
chunksize = 12000
passes = 25
iterations = 1000
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

start_time = time.time()

tfidf_lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')

# Save model to disk.
tfidf_lda_model.save("./tfidf_lda_model_concat_POSTag_self_tuned_all_sorted")

print("Tfidf model saved. DONE.")
