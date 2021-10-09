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
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

if __name__ == "__main__":
    file_path = sys.argv[1]
    
    in_file = open(file_path,"r")
    out_file = open("queries_6.txt","w")
    
    index = in_file.read()
    S = BeautifulSoup(index, 'lxml')
    query_ids = S.find_all("num")
    query_txt = S.find_all("title")
    
    query_ids = [query_id.text for query_id in query_ids]
    query_txt = [query.text for query in query_txt]
    query_txt = [remove_stopwords(query) for query in query_txt]
    query_txt = [remove_punctuation(query) for query in query_txt]
    query_txt = [lemmatize_words(query) for query in query_txt]
    
    for i in range(min(len(query_ids), len(query_txt))):
        out_file.write(str(query_ids[i])+","+query_txt[i]+'\n')
    
    in_file.close()
    out_file.close()