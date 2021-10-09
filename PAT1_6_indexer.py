import os
import sys
import pickle
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

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

if __name__ == "__main__":
    file_path = sys.argv[1]
    
    inv_index = {}
    
    docs = []
    
    for subdir, dirs, files in os.walk(file_path):
        for file in files:
            cur_file = open(os.path.join(subdir, file),"r")
            index = cur_file.read()
            S = BeautifulSoup(index, 'lxml')
            doc_name = S.find("docno").text
            doc_content = S.find("text").text
            
            docs.append((doc_name, doc_content))
    
    docs = [(tup[0],remove_stopwords(tup[1])) for tup in docs]
    docs = [(tup[0],remove_punctuation(tup[1])) for tup in docs]
    docs = [(tup[0],lemmatize_words(tup[1])) for tup in docs]
    
    for doc in docs:
        for token in doc[1]:
            if(token not in inv_index.keys()):
                inv_index[token]=[]
            inv_index[token].append(doc[0])
    
    
    for key in inv_index.keys():
        inv_index[key] = set(inv_index[key])
        inv_index[key] = list(inv_index[key])
        inv_index[key].sort()
    
    out_file = open("model_queries_6.pth","wb")
    pickle.dump(inv_index, out_file)