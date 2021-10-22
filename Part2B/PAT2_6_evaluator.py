import pandas as pd
import sys
import math

def averagePrecision(query, doc, number):
    c , sum, true = 0, 0, 0
    for i in range(number):
        c += 1
        if doc[i] in query:
            true += 1
            sum += true/c 
    if sum == 0:
        return 0
    return sum/true

def ndcg(query, score, doc, number):
    rank = []
    for i in range(number):
        if doc[i] in query:
            rank.append(score[query.index(doc[i])])
        else:
            rank.append(0)
    
    dcg_actual = rank[0]
    for i in range(1, number):
        dcg_actual += rank[i]/math.log(i + 1, 2)
    score.sort(reverse= True)
    while len(score) < number:
        score.append(0)
    
    dcg_ideal = score[0]
    for i in range(1, number):
        dcg_ideal += score[i]/math.log(i + 1, 2)
    if dcg_ideal == 0:
        return 0
    return dcg_actual/dcg_ideal

if __name__ == "__main__":
    gold_standard_path = sys.argv[1]
    ranked_list = sys.argv[2]
    df = pd.read_csv(ranked_list)

    real = pd.read_csv(gold_standard_path)
    data = []
    sumAvgQuer10 = 0
    sumAvgQuer20 = 0
    sumndcgQuer10 = 0
    sumndcgQuer20 = 0

    query = list(set(df['Query_ID']))
    for i in range(len(query)):
        avgQuer10 = averagePrecision(list(real[real['Query_ID'] == query[i]]['Document_ID']), list(df[df['Query_ID'] == query[i]]['Document_ID']), 10)
        ndcgQuery10 = ndcg(list(real[real['Query_ID'] == query[i]]['Document_ID']), list(real[real['Query_ID'] == query[i]]['Relevance_Score']), list(df[df['Query_ID'] == query[i]]['Document_ID']), 10)
        avgQuer20 =  averagePrecision(list(real[real['Query_ID'] == query[i]]['Document_ID']), list(df[df['Query_ID'] == query[i]]['Document_ID']), 20)
        ndcgQuery20 = ndcg(list(real[real['Query_ID'] == query[i]]['Document_ID']), list(real[real['Query_ID'] == query[i]]['Relevance_Score']), list(df[df['Query_ID'] == query[i]]['Document_ID']), 20)
        data.append([query[i], avgQuer10, avgQuer20, ndcgQuery10, ndcgQuery20])
        sumAvgQuer10 += avgQuer10
        sumAvgQuer20 += avgQuer20
        sumndcgQuer10 += ndcgQuery10
        sumndcgQuer20 += ndcgQuery20

    data.append([math.nan, sumAvgQuer10/len(query), sumAvgQuer20/len(query), sumndcgQuer10/len(query), sumndcgQuer20/len(query)])

    final = pd.DataFrame(data, columns =['QueryID', 'AP@10', 'AP@20', 'NDCG@10', 'NDCG@20'])
    final_path = 'PAT2_6_metrics_' +  ranked_list[-5] + '.csv'
    final.to_csv(final_path, index = False)