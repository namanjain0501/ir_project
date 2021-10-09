import os
import sys
import pickle

'''
    Function to merge posting lists p1 and p2
    params : 
        p1 : posting1
        p2 : posting2
    returns : 
        intersection of p1 and p2
'''

def merge(p1 , p2):
    i = 0
    j = 0
    result = []
    while(i < len(p1) and j < len(p2)):
        if(p1[i] == p2[j]):
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    
    return result


if __name__ == "__main__":
    model_path = sys.argv[1]
    queries_path = sys.argv[2]
    
    queries = []
    queryIds = []
    
    inv_index = pickle.load(open(model_path, "rb"))

    with open(queries_path) as f:
        for line in f.readlines():
            queryId, query = line.split(',')
            queryIds.append(queryId)
            queries.append(query)
        f.close()
    
    out = open("PAT1_6_results.txt", "w")

    for queryId, query in zip(queryIds, queries):
        query = query.strip()
        words = list(query.split())
        words.sort(key=lambda x: len(inv_index.get(x, [])))
        
        if(len(words) == 1):
            result = inv_index.get(words[0], [])
        else:
            result = merge(inv_index.get(words[0], []), inv_index.get(words[1], []))
            for i in range(2, len(words)):
                result = merge(result, inv_index.get(words[i], []))
        
        out.write(str(queryId) + ":" + ' '.join(result) + '\n')
        
    out.close()