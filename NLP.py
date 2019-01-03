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
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_dataset1():
    dataset = pd.read_csv('tweets.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
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

def read_dataset2():
    dataset = pd.read_csv('Clean_Disasters_T_79187_.csv',delimiter = ',' ,converters={'text': str}, encoding = "ISO-8859-1")
    return dataset

def class_hist(dataset):
    ##Class histgram
    dataset['choose_one'].hist()
    # Class percentage
    perc = dataset.choose_one.value_counts(normalize=True)
    # Provide descriptive statistics 
    pd.plotting.scatter_matrix(pd.DataFrame(dataset), alpha = 0.9, figsize = (8,6), diagonal = 'kde')
    return perc
    
def missing_values(dataset):    
    # Check for missing values
    print('Missing Values')
    display(dataset.isnull().sum())

def make_corpus():    
    corpus = []
    for i in range(0,79187):
        corpus.append(dataset.text[i])
    return corpus
 
def Bow_Split(corpus,dataset,max_features): #### 2-Bag of words model
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    #Count Vecorizer
    cv = CountVectorizer(max_features = (max_features))
    X = cv.fit_transform(corpus).toarray() 
    
    ####Tf-Idf Vectorizer
    #tf = TfidfVectorizer(max_features=(50))
    #X = tf.fit_transform(corpus).toarray()
    
    #Split Dataset to X and y
    y = dataset.iloc[: , 3].values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    return X,y

#read original dataset
dataset = read_dataset1()
cor = clean_text()
df = pd.DataFrame({'text':cor})
df['choose_one'] = dataset['choose_one']
df.to_csv('clean_df.csv', sep=',', encoding='utf-8')
# read cleaned dataset
dataset = read_dataset2()
#class_hist(dataset)
#missing_values(dataset)
#Make Word Corpus
corpus = make_corpus()
X,y= Bow_Split(corpus,dataset,max_features=500)
X_train, X_test, y_train, y_test = Test_Train_Split(X,y,test_size = 0.3)
