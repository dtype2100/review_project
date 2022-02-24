# -*- coding: utf-8 -*-
"""review_topic

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Gls2ZXk9rSDpSMWW4su42GK79CifSJUq
"""

from google.colab import drive
drive.mount('/content/drive')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

review_ls = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/멀캠-프로젝트/스터디_프로젝트/review_project/olive_app_review_crawling_final.csv')

review_ls.head()

n_topics = 4

tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(review_ls)
svd = TruncatedSVD(n_components=n_topics, n_iter=100)

U = svd.fit_transform(tfidf)
Vt = svd.components_

len(review_ls)

U.shape

Vt.shape

tfidf.shape

vocab =  tfidf_vect.get_feature_names_out()
print(len(vocab))

Vt

n = 4
for i, topic in enumerate(Vt):
    print("Topic {} :".format(i), [ (vocab[i], topic[i].round(3)) for i in topic.argsort()[:-n-1:-1]])

n = 3
temp = [0, 1, 2, 3, 4, 5]
temp[:-n:-1]

import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def calc_similarity_matrix(vectors):
    
    n_word = len(vectors)
    similarity_matrix = np.zeros((n_word, n_word))
    
    for i in range(n_word):
        for j in range(n_word):
            similarity_matrix[i, j] = cosine_similarity(vectors[i], vectors[j]).round(3)
            
    return similarity_matrix

vectors = Vt.T
word_similarity_matrix = calc_similarity_matrix(vectors)
word_similarity_matrix.shape

import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", family="Malgun Gothic") #windows
# plt.rc("font", family="AppleGothic") #mac

def visualize_similarity(similarity_matrix, label):
    mask = np.triu(np.ones_like(similarity_matrix, dtype=np.bool))
    plt.rcParams['figure.figsize'] = [8, 6]
    ax = sns.heatmap(similarity_matrix, mask=mask, xticklabels=label, yticklabels=label, 
                     annot=True, fmt=".2f", annot_kws={"size":8}, cmap="coolwarm")

visualize_similarity(word_similarity_matrix, vocab)

doc_similarity_matrix = calc_similarity_matrix(U)
visualize_similarity(doc_similarity_matrix, review_ls)

from sklearn.manifold import TSNE

def visualize_vectors(vectors, labels):
    tsne = TSNE(n_components=2, n_iter=10000, perplexity=2)
    T = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords="offset points")

visualize_vectors(vectors, vocab)

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


dataset = fetch_20newsgroups(shuffle=True, random_state=42, remove=("headers", "footer", "quotes"))

!pip install pyLDAvis

dataset['data'][:5]

def get_news(news_dataset, apply_split=True):
    documents = news_dataset.data
    news_df = pd.DataFrame({"document": documents})
    news_df["clean_doc"] = news_df["document"].str.replace("[^a-zA-Z]", " ")
    news_df["clean_doc"] = news_df["clean_doc"].apply(lambda x : " ".join([w.lower() for w in x.split() if len(w) > 2]))
    tokenized_doc = news_df["clean_doc"].apply(lambda x: x.split())
    
    stop_words = stopwords.words('english')
    
    if apply_split:
        return tokenized_doc.apply(lambda x : [item for item in x if item not in stop_words])
    else:
        return tokenized_doc.apply(lambda x : " ".join([item for item in x if item not in stop_words]))

tokenized_docs = get_news(dataset, False)

tokenized_docs[0]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(tokenized_docs)
lda = LatentDirichletAllocation(n_components=5, max_iter=50, learning_method="online", random_state=42)
lda_output = lda.fit_transform(tfidf)

import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda, tfidf, tfidf_vect, mds="tsne")
pyLDAvis.display(vis)

