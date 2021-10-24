import os
import pickle
import sys
import nltk
from bs4 import BeautifulSoup
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
import csv

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return [lemmatizer.lemmatize(word) for word in text.split()]

def get_doc_wt(doc, scheme):
    doc_wt = {}
    #doc_set = set(docs[doc])
    if scheme[0] == 'l':
        for token in docs[doc]:
            doc_wt[token] = (1 + math.log10(inv_index[token][doc]))
    elif scheme[0] == 'L':
        ave_tf = np.mean([inv_index[token][doc] for token in docs[doc]])
        for token in docs[doc]:
            doc_wt[token] = (1 + math.log10(inv_index[token][doc])) / (1 + math.log10(ave_tf))
    norm = np.linalg.norm(list(doc_wt.values()))
    for token in doc_wt:
        doc_wt[token] /= norm
    return doc_wt

def get_query_wt(query, scheme):
    query_wt = {}
    if scheme[0] == 'l':
        for token in set(query):
            query_wt[token] = (1 + math.log10(query.count(token)))
    elif scheme[0] == 'L':
        ave_tf = np.mean([query.count(token) for token in set(query)])
        for token in set(query):
            query_wt[token] = (1 + math.log10(query.count(token))) / (1 + math.log10(ave_tf))

    if scheme[1] == 't':
        for token in set(query):
            query_wt[token] *= ((math.log10(N / len(inv_index[token]))) if (token in vocab_set) else 0)
    elif scheme[1] == 'p':
        for token in set(query):
            query_wt[token] *= (max(0, math.log10((N - len(inv_index[token])) / len(inv_index[token]))) if (token in vocab_set) else 0)

    norm = np.linalg.norm(list(query_wt.values()))
    for token in query_wt:
        query_wt[token] /= norm
    return query_wt

def insert_in_sorted_list(a, x, size=50):
    a.append(x)
    for i in range(len(a) - 1, 0, -1):
        if a[i][2] > a[i - 1][2]:
            a[i], a[i - 1] = a[i - 1], a[i]
        else:
            break
    if len(a) > size:
        a.pop()
    return a

def aug(tf, max_tf):
    return 0.5 + (0.5 * tf) / max_tf

def prob(df):
    if df == 0:
        return 0
    return max(0, math.log10((N - df) / df))

def get_scores_aug():
    res = {}
    max_tf_td = {}
    aug1 = {}
    deno1 = {}
    for i, doc in enumerate(docs):
        max_tf_td[doc] = max([inv_index[token][doc] for token in docs[doc]])
        aug1[doc] = {token: aug((inv_index[token][doc]), max_tf_td[doc]) for token in docs[doc]}
        deno1[doc] = sum(np.multiply(list(aug1[doc].values()), list(aug1[doc].values())))
        l = N - len(docs[doc])
        deno1[doc] += l * 0.25
        deno1[doc] = math.sqrt(deno1[doc])
    for query, qid in zip(query_txt, query_ids):
        res[qid] = []
        max_tf_tq = max([query.count(token) for token in set(query)])
        for i, doc in enumerate(docs):
            score = 0
            deno2 = 0
            for token in set(query):
                tf_tq = query.count(token)
                df_t = (len(inv_index[token]) if (token in vocab_set) else 0)
                score += (aug1[doc][token] if (token in docs[doc]) else 0.5) * aug(tf_tq, max_tf_tq) * prob(df_t)
                deno2 += (aug(tf_tq, max_tf_tq) * prob(df_t)) ** 2
            l = N - len(set(query))
            deno2 += l * 0.5 ** 2
            deno2 = math.sqrt(deno2)
            score /= (deno1[doc] * deno2)
            res[qid] = insert_in_sorted_list(res[qid], (qid, doc, score))
    return res

def get_scores(scheme):
    if scheme[0] == 'a' and scheme[4] == 'a':
        return get_scores_aug()
    scores = {}
    doc_wt = {}
    for i, doc in enumerate(docs):
        doc_wt[doc] = get_doc_wt(doc, scheme.split('.')[0])
    query_wt = {}
    for query, qid in zip(query_txt, query_ids):
        scores[qid] = []
        query_wt[qid] = get_query_wt(query, scheme.split('.')[1])
        for i, doc in enumerate(docs):
            score = 0
            for token in set(query_wt[qid].keys()).intersection(set(doc_wt[doc])):
                score += (query_wt[qid][token] * (doc_wt[doc][token] if (token in doc_wt[doc]) else 0))
            scores[qid] = insert_in_sorted_list(scores[qid], (qid, doc, score))
    return scores


if __name__ == "__main__":
    data_folder_path = sys.argv[1]
    inv_index_file_name = sys.argv[2]
    queries_file_name = (sys.argv[3] if (len(sys.argv) == 4) else 'queries_6.txt')

    input_file = open(inv_index_file_name, 'rb')
    inverted_index = pickle.load(input_file)

    inv_index = {key: dict(inverted_index[key]) for key in inverted_index}
    docs = {}
    for token in inv_index:
        for doc in inv_index[token]:
            if doc not in docs:
                docs[doc] = set()
            docs[doc].add(token)

    vocab_set = set(inv_index.keys())
    with open('queries_6.txt') as f:
        queries = f.readlines()
        query_ids = [line.split(',')[0] for line in queries]
        query_txt = [line.split(',')[1][:-1].split(' ') for line in queries]

    N = len(docs)

    schemes = ['lnc.ltc', 'Lnc.Lpc', 'anc.apc']
    scores = {}
    for scheme in schemes:
        scores[scheme] = get_scores(scheme)

    for i, scheme in enumerate(scores):
        char = chr(ord('A') + i)
        with open('PAT2_6_ranked_list_' + char + '.csv', 'w', newline='') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(('Query_ID', 'Document_ID'))
            for qid in scores[scheme]:
                for row in scores[scheme][qid]:
                    csv_out.writerow(row[:2])
