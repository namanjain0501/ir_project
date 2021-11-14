import os
import pickle
import sys
import string
import math
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm


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


# accepts relevant_docs, non_relevant_docs as a list of dictionaries
def modify_query(q, alpha, beta, gamma, relevant_docs, non_relevant_docs):
    qm = dict()
    for token, wt in q.items():
        qm[token] = alpha * wt
    
    for relevant_doc in relevant_docs:
        for token, wt in relevant_doc.items():
            qm[token] = qm.get(token, 0) +(beta/len(relevant_docs)) * wt
    
    for non_relevant_doc in non_relevant_docs:
        for token, wt in non_relevant_doc.items():
            qm[token] = qm.get(token, 0) - (gamma/len(non_relevant_docs)) * wt
    
    return qm 

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

def get_scores_with_query_updated(scheme, alpha, beta, gamma, relevant_docs, non_relevant_docs):
    scores = {}
    doc_wt = {}
    for i, doc in enumerate(docs):
        doc_wt[doc] = get_doc_wt(doc, scheme.split('.')[0])
    query_wt = {}
    for query, qid in tqdm(list(zip(query_txt, query_ids))):
        scores[qid] = []
        query_wt[qid] = get_query_wt(query, scheme.split('.')[1])
        query_wt[qid] = modify_query(query_wt[qid], alpha, beta, gamma, relevant_docs, non_relevant_docs)
        for i, doc in enumerate(docs):
            score = 0
            for token in set(query_wt[qid].keys()).intersection(set(doc_wt[doc])):
                score += (query_wt[qid][token] * (doc_wt[doc][token] if (token in doc_wt[doc]) else 0))
            scores[qid] = insert_in_sorted_list(scores[qid], (qid, doc, score),20)
    return scores

if __name__ == "__main__":
    data_folder_path = sys.argv[1]
    inv_index_file_name = sys.argv[2]
    ranked_list = sys.argv[3]

    retrieved = pd.read_csv(ranked_list)
    #real = pd.read_csv(gold_standard_path)

    ranked_docs = dict()
    for query_id, doc_id in zip(retrieved['Query_ID'], retrieved['Document_ID']): 
        if(query_id not in ranked_docs):
            ranked_docs[query_id] = []
        ranked_docs[query_id].append(doc_id)


    input_file = open(inv_index_file_name, 'rb')
    inverted_index = pickle.load(input_file)

    inv_index = {key: dict(inverted_index[key]) for key in inverted_index}
    docs = {}
    for token in inv_index:
        for doc in inv_index[token]:
            if doc not in docs:
                docs[doc] = set()
            docs[ doc].add(token)

    vocab_set = set(inv_index.keys())
    with open('queries_6.txt') as f:
        queries = f.readlines()
        query_ids = [line.split(',')[0] for line in queries]
        query_txt = [line.split(',')[1][:-1].split(' ') for line in queries]

    N = len(docs)

    schemes = ['PsRF']
    for scheme in schemes :
        for query_id, ranking in ranked_docs.items():
            ranking = ranking[:20]
            relevant_docs = []
            non_relevant_docs = []

            if(scheme == 'RF'):
                for doc in ranking: 
                    if(doc in relevant_docs_gold.get(query_id, set())):
                        relevant_docs.append(get_doc_wt(doc, 'lnc'))
                    else:
                        non_relevant_docs.append(get_doc_wt(doc, 'lnc'))
            else:
                for doc in ranking[:10]: 
                    relevant_docs.append(get_doc_wt(doc, 'lnc'))

        data = []

        # getting the improtant words
        scores = get_scores_with_query_updated('lnc.ltc', alpha=1, beta=1, gamma=0.5, relevant_docs=relevant_docs, non_relevant_docs=non_relevant_docs)
        for query_num in scores:
            m = {}
            sz = 0
            for j in scores[query_num]:
                sz += 1
                if sz > 10:
                    break
                temp = get_doc_wt(j[1], 'lnc.ltc')
                for k in temp:
                    if k not in m:
                        m[k] = 0
                    m[k] += temp[k]
            x = []
            for j in m:
                x.append([m[j], j])
            x.sort(reverse = True)
            f = []
            f.append(query_num)
            for j in range(5):
                f.append(x[j][1])
            data.append(f)

        final = pd.DataFrame(data, columns =['Query_ID', 'Word_1', 'Word_2', 'Word_3', 'Word_4', 'Word_5'])
        final_path = 'PB_6_important_words.csv'
        final.to_csv(final_path, index = False)
