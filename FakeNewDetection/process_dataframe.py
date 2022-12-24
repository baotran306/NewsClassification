import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime
import string
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader import wordnet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler


def tag(t):
    if t.startswith("N"):
        return wordnet.NOUN
    elif t.startswith("V"):
        return wordnet.VERB
    elif t.startswith("R"):
        return wordnet.ADV
    elif t.startswith("J"):
        return wordnet.ADJ
    else:
        return wordnet.NOUN


def lemmatize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w, pos=tag(wn)) for w, wn in tokens]

    return " ".join(tokens)


def remove_num(text, num_only=False):
    if not num_only:
        # just remove number like 1, 123123, 55325794, ...
        token = [w for w in text if not w.isdigit()]
    else:
        # remove all word which contains num like 1, 213, as89, 12th, 3rd ...
        token = [w for w in text if not any(map(str.isdigit, w))]
    return ' '.join(token)


def remove_links(text):
    import re
    text = re.sub(r'http\S+', '', text)  # remove http links
    text = re.sub(r'bit.ly\S+', '', text)  # rempve bitly links
    text = text.strip('[link]')  # remove [links]
    return text


def remove_tags(text):
    import re
    text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', text) # remove tag at
    return text


def clean_text(text, num_only=False):
    """
    First of all, remove special character.
    It doesn't any extra functions. So remove it help reduce size of data
    string.punctuation contains all special character like @ $ # * & ...
    """
    stop_w = stopwords.words('english')
    translators = str.maketrans("", "", string.punctuation)
    text = text.translate(translators)

    """
    Next, i will lowercase character and remove number.
    If we don't lowercase, when count word, Basic and basic are 2 different word
    This avoid mutiplies copy in same word.
    """
    text = remove_tags(text)
    text = remove_links(text)
    words = word_tokenize(text.lower())
    tokens = word_tokenize(remove_num(words, False))

    """
    Then, remove stop word and word that length <= 1
    Because, word has length <= 1 usually dont have meaning
    """
    tokens = [w for w in tokens if w not in stop_w and len(w) > 1]

    return " ".join(tokens)


def pre_processing(df):
    df['clean_text'] = df['full_text'].apply(clean_text)
    df['clean_text'] = df['clean_text'].apply(lambda x: lemmatize(x))
    return df


def load_basic_feature(df):
    stop_w = stopwords.words('english')
    df['full_text'] = df['subject'] + ' ' + df['title'] + ' ' + df['text']
    df = df.drop(columns=['title', 'text', 'subject', 'date'])
    df['length_text'] = df['full_text'].apply(len)
    df['word_count'] = df['full_text'].apply(lambda x: len(x.split(" ")))
    df['num_stop_w'] = df['full_text'].apply(lambda x: len([word for word in x.split(" ") if word in stop_w]))
    df['num_special_char'] = df['full_text'].apply(
        lambda x: len([char for char in x if char in string.punctuation]))
    df['contains_num'] = df['full_text'].apply(
        lambda x: len([word for word in x if any(map(str.isdigit, word))]))
    return df


def load_tf_idf(df):
    tfidf = TfidfVectorizer(max_features=10, analyzer='word', ngram_range=(1, 1))
    tfidf_text_train = tfidf.fit_transform(df['clean_text']).toarray()
    tfidf_train_df = pd.DataFrame(tfidf_text_train, columns=tfidf.get_feature_names())
    tfidf_train_df.columns = ["word_" + str(x) for x in tfidf_train_df.columns]
    tfidf_train_df.index = df.index
    df = pd.concat([df, tfidf_train_df], axis=1)
    return df


def load_doc_2_vec(df):
    documents_train = [TaggedDocument(doc, [i]) for i, doc in
                       enumerate(df['clean_text'].apply(lambda x: x.split(" ")))]
    # Train a Doc2Vec model with out text data
    model_train = Doc2Vec(documents_train, vector_size=5, window=2, min_count=1, workers=4)
    # Transform each documnet into a vector data
    doc2vec_train_df = df["clean_text"].apply(lambda x: model_train.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_train_df.columns = ['doc2vec_vector_' + str(x) for x in doc2vec_train_df.columns]
    df = pd.concat([df, doc2vec_train_df], axis=1)
    return df


def process_df(df, type='train'):
    df_res = load_basic_feature(df)
    df_res = pre_processing(df_res)
    df_res = load_tf_idf(df_res)
    df_res = load_doc_2_vec(df_res)
    # print(df.columns)
    if type == 'train':
        drop_columns = ['full_text', 'clean_text', 'label']
    else:
        drop_columns = ['full_text', 'clean_text']
    df_res = df_res.drop(columns=drop_columns).to_numpy()
    scaler = StandardScaler().fit(df_res)
    df_res = scaler.transform(df_res)
    if type == 'train':
        df_label = df['label']
        return df_res, df_label
    return df_res
