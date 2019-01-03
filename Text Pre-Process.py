import numpy as np
import pandas as pd
import wordninja
from scipy.sparse import hstack
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')
import re
from nltk.stem.porter import PorterStemmer

def read_dataset():
    dataset = pd.read_csv('socialmedia_disaster_tweets.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
    dataset.columns
    dataset = dataset.loc[dataset.choose_one!="Can't Decide",:]
    dataset.reset_index(drop=True, inplace=True)
    dataset['choose_one'].replace(['Relevant','Not Relevant'],[1,0],inplace=True)
    return dataset

def remove_emoji(string):
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\u200d"
                       u"\u2640-\u2642" 
                       "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
corpus = []

def clean_text():
    for i in range(0 , 10860):
        dataset ['text'][i]
        
        review = re.sub(r"http\S+", "", dataset ['text'][i])
        review = remove_emoji(review)
        review = " ".join([a for a in re.split('([A-Z][a-z]+)', review) if a])
        review = re.sub('[^a-zA-Z]' , ' ' , review)
        review = ' '.join(wordninja.split(review) )
        review = review.lower()
        review = re.sub(r"i'm", "i am",review)
        review = re.sub(r"he's", "he is",review)
        review = re.sub(r"she's", "she is",review)
        review = re.sub(r"that's", "that is",review)
        review = re.sub(r"where's", "where is",review)
        review = re.sub(r"what's", "what is",review)
        review = re.sub(r"\'ll", "will",review)
        review = re.sub(r"\'ve", "have",review)
        review = re.sub(r"\'re", "are",review)
        review = re.sub(r"won't", "will not",review)
        review = re.sub(r"can't", "can not",review)
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review) 
        corpus.append(review)
    return corpus

dataset = read_dataset()
cor = clean_text()
df = pd.DataFrame({'text':cor})
df['choose_one'] = dataset['choose_one']

df.to_csv('clean_df.csv', sep=',', encoding='utf-8')
