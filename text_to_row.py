import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import seaborn as sns
import re
from nltk import pos_tag, word_tokenize
from bs4 import BeautifulSoup


def count_words(x):
    x=x.split()
    return(len(x))
def count_chars(text):
    return len(text)
def count_unique_words(text):
    return len(set(text.split()))
def Find(string):
  
    # findall() has been used 
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)  
    j=[x[0] for x in url]
    return len(j)
    
def check_number_of_noun(x):
    x=x.split()
    x=pos_tag(x)
    noun=0
    verb=0
    adjective=0
    for i in x:
        i=list(i)
        if(i[1] == 'NN' or i[1] == 'NNS' or i[1] == 'NNPS' or i[1] == 'NNP'):
            noun=noun+1
    #print(noun)        
    return (noun)
def check_number_of_verb(x):
    x=x.split()
    x=pos_tag(x)
    verb=0
    for i in x:
        i=list(i)
        if(i[1] == 'VB' or i[1] == 'VBG' or i[1] == 'VBD' or i[1] == 'VBN' or i[1]=='VBP' or i[1]=='VBZ'):
            verb=verb+1
            
    #print(verb)        
    return (verb)
    
def check_number_of_adj(x):
    x=x.split()
    x=pos_tag(x)
    adjective=0
    for i in x:
        i=list(i)
        if(i[1] == 'JJ' or i[1] == 'JJR' or i[1] == 'JJS' or i[1] == 'VBN' or i[1]=='VBP' or i[1]=='VBZ'):
            adjective=adjective+1
            
    #print(adjective)        
    return (adjective)
# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess_with_stopwords(x):
    x = str(x).lower()
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,x)  
    j=[x[0] for x in url]
    for i in j:
        x=x.replace(i,"url")
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x=re.sub('[^a-zA-Z]', ' ', x)
    
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    #print(x)
    return x
    
def row_gen(tit,text):
    text_length=count_words(text)
    title_length=count_words(tit)
    text_chqar_len=count_chars(text)
    uniqu_word=count_unique_words(text)
    avg_Wordlen=text_chqar_len/text_length
    uniqe_vs_word=uniqu_word/text_length
    num_of_link=Find(text)
    num_of_noun=check_number_of_noun(text)
    num_of_verb=check_number_of_verb(text)
    num_of_adj=check_number_of_adj(text)
    text=preprocess_with_stopwords(text)
    tit=preprocess_with_stopwords(tit)
    all_txt=tit+" "+text
    result=[text_length,title_length,text_chqar_len,uniqu_word,avg_Wordlen,uniqe_vs_word,num_of_link,num_of_noun,num_of_verb,num_of_adj,all_txt]
    return result
    
    
    

    

    