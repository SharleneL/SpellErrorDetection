import nltk
import random
import os
import math
from nltk.tokenize import TweetTokenizer
from nltk.collocations import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn import svm

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

def main():
    utterances = []
    tknzr = TweetTokenizer()
    
    # FILE PATHS
    task_cat_path = '../Data/utterances/task/task-category.csv'
    task_utt_path = '../Data/utterances/task/task-correct.csv'
    nontask_cat_path = '../Data/utterances/nontask/nontask-category.csv'
    nontask_utt_path = '../Data/utterances/nontask/nontask-correct.csv'
    chat_path = '../Data/chat/chatbot.txt'

    # LOAD DATA
    # save the task utterances - (utterance_line, category)
    with open(task_cat_path) as f_cat, open(task_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip().decode('latin-1')
            utt_line = utt_line.strip().decode('latin-1')
            get_utterances(utterances, utt_line, cat_line, 1, 0)  
    # save the nontask utterances - (utterance_line, category)
    with open(nontask_cat_path) as f_cat, open(nontask_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip().decode('latin-1')
            utt_line = utt_line.strip().decode('latin-1')
            get_utterances(utterances, utt_line, cat_line, 1, 0)  
    # save the chat utterances - (utterance_line, category)
    with open(chat_path) as f:
        for line in f:
            line = line.strip().decode('latin-1')
            get_utterances(utterances, line, 'Chat', 1, 0)        

    random.shuffle(utterances)
#     print utterances[238]

    X = list(u[0] for u in utterances)
    y = list(u[1] for u in utterances)

    # convert to features
    X = [get_features(x) for x in X]
    v = DictVectorizer()
    X = v.fit_transform(X)

#     ==========/ Logistic Regression /========== 
    lr = LogisticRegression()
    scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    print scores.mean()


# OTHER FUNCTIONS
# convert to features
def get_features(value):
    words = set(value)
    features = {}
    for w in words:
        features[w] = 1
    return features

def get_utterances(utterances, line, category, wgram, cgram):
    tknzr = TweetTokenizer()
    gram_list = []
    # WORD GRAMS
    if wgram == 1:  # unigram
        wgram_list = tknzr.tokenize(line)
    elif wgram == 2:  # uni + bigram
        # unigram list
        tokens = nltk.wordpunct_tokenize(line)
        # bigram list
        finder = BigramCollocationFinder.from_words(tokens)
        scored = finder.score_ngrams(bigram_measures.raw_freq)
        bigram_list = sorted(bigram for bigram, score in scored)
        # res
        wgram_list = tknzr.tokenize(line) + bigram_list
    elif wgram == 3: # uni + bi + trigram
        # unigram list
        tokens = nltk.wordpunct_tokenize(line)
        # bigram list
        bi_finder = BigramCollocationFinder.from_words(tokens)
        bi_scored = bi_finder.score_ngrams(bigram_measures.raw_freq)
        bigram_list = sorted(bigram for bigram, biscore in bi_scored)  
        # trigram list
        tri_finder = TrigramCollocationFinder.from_words(tokens)
        tri_scored = tri_finder.score_ngrams(trigram_measures.raw_freq)
        trigram_list = sorted(trigram for trigram, triscore in tri_scored)
        # res
        wgram_list = tknzr.tokenize(line) + bigram_list + trigram_list
    
    # CHAR GRAMS
    cgram_list = []
    if cgram == 1:   # uni-chargram
        cgram_list = [line[i:i+1] for i in range(len(line)-1)]
    elif cgram == 2: # bi-chargram
        cgram_list = [line[i:i+2] for i in range(len(line)-1)]
    elif cgram == 3: # tri-chargram
        cgram_list = [line[i:i+3] for i in range(len(line)-1)]
        
    # RESULT
    if category == 'QA':            # non-task
        utterances.append((wgram_list + cgram_list, 0))
    elif category == 'Shopping':    # task
        utterances.append((wgram_list + cgram_list, 1))
    elif category == 'Travel':      # task
        utterances.append((wgram_list + cgram_list, 2))
    elif category == 'Hotel':       # task
        utterances.append((wgram_list + cgram_list, 3))
    elif category == 'Food':        # task
        utterances.append((wgram_list + cgram_list, 4))
    elif category == 'Art':         # task
        utterances.append((wgram_list + cgram_list, 5))
    elif category == 'Weather':     # task
        utterances.append((wgram_list + cgram_list, 6))
    elif category == 'Friends':     # task
        utterances.append((wgram_list + cgram_list, 7))
    elif category == 'Chat':        # chat
        utterances.append((wgram_list + cgram_list, 8))
    else:
        print utt_category,"ERROR"
        

if __name__ == '__main__':
    main()