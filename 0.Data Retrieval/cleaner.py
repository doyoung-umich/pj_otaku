import re
import numpy as np
import pandas as pd
import pickle
from IPython.display import display

import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

# NLTK and stopwords
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer

# language detection
from langdetect import detect

# ============================== functions ==============================

def clean_text(txt, stop_words):
    '''
    Gets rid of: html tags, source description, line breaks(\n), some specific expressions (ex: "notes"), punctuations, numbers
    :param txt: input text
    :param stop_words: list of stop words
    :return: cleaned text
    '''
    try:
        c_tag = re.compile("<.*?>") # html tags
        clean_txt = re.sub(c_tag, "", txt)
        c_source = re.compile("\((Source: .*?)\)") # source description
        clean_txt = re.sub(c_source, "", clean_txt)
        clean_txt = " ".join(clean_txt.split())
        c_notes = re.compile("Notes:.*|Note:.*|notes:.*|note:.*") # any "notes"
        clean_txt = re.sub(c_notes, '', clean_txt)
        # clean_txt = clean_txt.lower() # lowering characters
        clean_txt = re.sub("[^A-Za-z]+", " ", clean_txt) # remove punctuations and numbers
        clean_txt = [word for word in clean_txt.split() if word not in stop_words] # removing stopwords
        clean_txt = " ".join(clean_txt)
        return clean_txt
    except:
        print(txt) # nan can cause this error but if anything else happens

def extract_source(txt):
    '''
    Extracts where the synopsis comes from (source)
    :param txt: input text
    :return: source in string format
    '''
    try:
        match = re.findall("\((Source: .*?)\)", txt)
        source = re.findall("(?<=Source: ).*$", match[0])
        return source[0]
    except:
        return "no match"

def word_count(txt):
    '''
    word count of the input text
    :param txt: input text
    :return: word count in integer
    '''
    try:
        word_count = len(txt)
        return word_count
    except:
        return 0

def preprocess_text(txt, w2v_model):
    '''
    Remove named entities (by identifying capital letter) and names/words that are not in standard word2vec model
    :param txt: input text
    :param w2v_model: w2v_model that is referenced
    :return: tokens as list
    '''
    # text = re.sub("\S{2,}", " ", text) # remove single character words
    txt = re.sub("\s*[A-Z]\w*\s*", " ", txt) # remove named entities by capital letter (it also removes the first word of sentences)
    tokens = word_tokenize(txt)
    tokens = [x for x in tokens if x in w2v_model.wv.vocab] # exclude non-word2vec words -> exclude names and romaji (romaji: japanese words in alphabets)
    # tokens = [WordNetLemmatizer().lemmatize(t, pos='v') for t in tokens] # lemmatizer
    return tokens

def detect_language(txt):
    '''
    Detect which language the input text is written in
    :param txt: input text
    :return: type of language in string format
    '''
    try:
        return detect(txt)
    except:
        return "unknown or not given"


# # ============================== titles table: load and preprocess ==============================
# print("\n")
# print("~~~~~~~~~~~~~~~ start basic preprocessing ~~~~~~~~~~~~~~~")
#
# # Load data
# df_titles = pd.read_csv("dataset/titles_200p.csv")
#
# # Preprocess genre column: convert from string to array
# df_titles["genres"] = df_titles["genres"].apply(lambda x: re.findall("'(.*?)'", x))
#
# # Convert genre columns to OHE-like columns
# mlb = MultiLabelBinarizer()
# df_titles = df_titles.join(pd.DataFrame(mlb.fit_transform(df_titles["genres"]),columns=mlb.classes_))
# df_titles = df_titles.drop("genres", axis=1)
#
# # create new title with anime/manga indication at the end so that we can check for duplicate records
# df_titles["title_romaji_type"] = df_titles["title_romaji"] + "_" + df_titles["type"]
#
# # Remove duplicate rows and take only the first row of the duplicate records.
# print("rows before removing duplicates: ", len(df_titles))
# print("no. of titles before removing duplicates: ", len(df_titles["title_romaji_type"].unique()))
# df_titles = df_titles.drop_duplicates(subset=["title_romaji_type"])
# print("rows after removing duplicates: ", len(df_titles))
# print("no. of titles after removing duplicates: ", len(df_titles["title_romaji_type"].unique()))
#
#
# # ============================== titles table: clean text columns ==============================
# print("\n")
# print("~~~~~~~~~~~~~~~ start text column preprocessing ~~~~~~~~~~~~~~~")
#
# # Drop titles without synopsis
# print("no. of titles without synopsis: ", len(df_titles) - len(df_titles[df_titles["synopsis"].notna()]))
# df_titles = df_titles[df_titles["synopsis"].notna()]
# print("no. of titles after removing titles w/o synopsis: ", len(df_titles))
#
# # Check functions
# test_text = df_titles["synopsis"][0]
# # print("text before cleaning: ", test_text)
# print("test text after cleaning: ", clean_text(test_text, stopwords.words("english")))
# print("source of synopsis: ", extract_source(test_text))
# print("word count of cleaned synopsis: ", word_count(clean_text(test_text, stopwords.words("english"))))
#
# # Clean synopsis data
# df_titles["synopsis_cleaned"] = df_titles["synopsis"].apply(lambda x: clean_text(x, stopwords.words("english")))
# df_titles["synopsis_source"] = df_titles["synopsis"].apply(lambda x: extract_source(x))
# df_titles["synopsis_wc"] = df_titles["synopsis_cleaned"].apply(lambda x: word_count(x))
#
# # Use the text8_w2v model that was trained separately and saved under specified directory
# f = open("dataset/w2v_text8_model.pkl", "rb")
# w2v_model = pickle.load(f)
# f.close()
#
# # Further preprocessing
# df_titles["synopsis_cleaned_token"] = df_titles["synopsis_cleaned"].apply(lambda x: preprocess_text(x, w2v_model))
# display(df_titles.head())
#
#
#
# # ============================== reviews table: load and preprocess ==============================
# print("\n")
# print("~~~~~~~~~~~~~~~ start text column preprocessing for reviews table ~~~~~~~~~~~~~~~")
#
# # load data
# df_reviews = pd.read_csv("dataset/reviews_200p.csv")
# # df_reviews = df_reviews.iloc[:100, :]
#
# # Filter out non-English reviews
# df_reviews["lang"] = df_reviews["text_summary"].apply(lambda x: detect_language(x))
# print("language ratio: ", df_reviews["lang"].value_counts(normalize=True)) #
# print("len before filtering out non-English reviews: ", len(df_reviews))
# df_reviews = df_reviews[df_reviews["lang"] == "en"] # 8222 out of 9008 titles are in English (91.2%)
# print("len after filtering out non-English reviews: ", len(df_reviews))
#
# # Clean text
# # test_text = df_reviews["text_body"][50]
# # print("text before cleaning: ", test_text)
# # print("test text after cleaning: ", clean_text(test_text, stopwords.words("english")))
# df_reviews["text_body_cleaned"] = df_reviews["text_body"].apply(lambda x: clean_text(x, stopwords.words("english")))
#
# # Further preprocessing and tokenize
# # print("test text after further preprocessing: ", preprocess_text(test_text, w2v_model))
# df_reviews["text_body_cleaned_token"] = df_reviews["text_body_cleaned"].apply(lambda x: preprocess_text(x, w2v_model))
# display(df_reviews.head())
#
#
#
# # ============================== users table: load and preprocess ==============================
# print("\n")
# print("~~~~~~~~~~~~~~~ start text column preprocessing for users table ~~~~~~~~~~~~~~~")
#
# # load data
# df_users = pd.read_csv("dataset/users_200p.csv", encoding="ISO-8859-1")
# # df_users = df_users.iloc[:100, :]
#
# # Filter out non-English "about" texts
# df_users["about_lang"] = df_users["about"].apply(lambda x: detect_language(x))
# print("language ratio: ", df_users["about_lang"].value_counts(normalize=True)) #
# print("len before filtering out non-English 'abouts': ", len(df_users))
# df_users = df_users[df_users["about_lang"] == "en"]
# print("len after filtering out non-English 'abouts': ", len(df_users))
#
# # Clean text
# # test_text = df_users["about"][0]
# # print("text before cleaning: ", test_text)
# # print("test text after cleaning: ", clean_text(test_text, stopwords.words("english")))
# df_users["about_cleaned"] = df_users["about"].apply(lambda x: clean_text(x, stopwords.words("english")))
#
# # Further preprocessing and tokenize
# # print("test text after further preprocessing: ", preprocess_text(test_text, w2v_model))
# df_users["about_cleaned_token"] = df_users["about_cleaned"].apply(lambda x: preprocess_text(x, w2v_model))
# display(df_users.head())
#
#
# # ============================== save files ==============================
# df_titles.to_csv("./dataset/titles_200p_cleaned.csv", index=False)
# df_reviews.to_csv("./dataset/reviews_200p_cleaned.csv", index=False)
# df_users.to_csv("./dataset/users_200p_cleaned.csv", index=False)


# ============================== checking results ==============================
df_titles_cleaned = pd.read_csv("dataset/titles_200p_cleaned.csv")
df_reviews_cleaned = pd.read_csv("dataset/reviews_200p_cleaned.csv")
df_users_cleaned = pd.read_csv("dataset/users_200p_cleaned.csv")

display(df_titles_cleaned.head())
display(df_reviews_cleaned.head())
display(df_users_cleaned.head())
