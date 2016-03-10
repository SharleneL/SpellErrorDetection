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

import cPickle

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
%matplotlib inline


def main():

    task_cat_path = 'Data/utterances/task/task-category.csv'
    task_utt_path = 'Data/utterances/task/task-error.csv'
    nontask_cat_path = 'Data/utterances/nontask/nontask-category.csv'
    nontask_utt_path = 'Data/utterances/nontask/nontask-error.csv'
    
    utterances = []

    # ==========/ LOAD DATA INTO UTTERANCES[] /========== #
    # save the task utterances - (utterance_line, category)
    with open(task_cat_path) as f_cat, open(task_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip()
            utt_line = "^" + utt_line.strip().replace(" ", "#") + "$"
            # ==========/ binary classification - start /========== #
#             if(cat_line == "QA"):
#                 category = 'nontask'  # non-task
#             else:
#                 category = 'task'  # task
            # ==========/ binary classification - end /========== #
            # ==========/ multi classification - start /========== #
            category = cat_line
            # ==========/ multi classification - end /========== #
#             get_utterances(utterances, utt_line, category, 1, 3)
            utterances.append((utt_line, cat_line))
    print len(utterances)
    
    # save the non-task utterances - (utterance_line, category)
    with open(nontask_cat_path) as f_cat, open(nontask_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip()
            utt_line = utt_line.strip()
            # ==========/ binary classification - start /========== #
#             if(cat_line == "QA"):
#                 category = 'nontask'  # non-task
#             else:
#                 category = 'task'  # task
            # ==========/ binary classification - end /========== #
            category = cat_line
#             get_utterances(utterances, utt_line, category, 1, 3)
            utterances.append((utt_line, cat_line))
    print len(utterances)
    
    # shuffle
    random.shuffle(utterances)
    
    # ==========/ SPLIT CROSS-VALIDATION TRAIN-TEST SETS /========== #
    fold = 5  # 5-fold cv
    fold_len = len(utterances) / fold
    fold_data = []
    for i in range(0, 5):
        fold_data.append(utterances[i*fold_len: (i+1)*fold_len])

    # ==========/ GET FEATURES FOR CROSS-VALIDATION TRAIN-TEST SETS /========== #
    indexed_voc = dict()
    
    X_all = []
    y_all = []
    
    for i in range(0, fold):
        # get train & test sets for current iteration
        test_utterances = fold_data[i]
        train_utterances = []
        for j in range(0, fold):
            if j!=i:
                train_utterances += fold_data[j]
        # get the vocabulary of train & test sets
        train_voc = get_vocabulary(train_utterances)
        test_voc = get_vocabulary(test_utterances)
        # check whether testing tokens are all included in training tokens => decide whether to use ngram
        ngram = False   # to mark whether to use ngram-char or not
        for test_token in test_voc:
            if test_token not in train_voc:  # if any of the test token did not appear in the train set, use ngram char
                ngram = True
                break
        
        # get train & test features
        train_feature = get_features(train_utterances, ngram)
        test_feature = get_features(test_utterances, ngram)
        # split - train
        train_X = [k[0] for k in train_feature]
        train_X_indexed = get_indexed_X(train_X, indexed_voc)
        train_y = [k[1] for k in train_feature]
        # split - test
        test_X = [k[0] for k in test_feature]
        test_X_indexed = get_indexed_X(test_X, indexed_voc)
        test_y = [k[1] for k in test_feature]
        
        # save indexed feature to pickle file
#         print test_X_indexed[0]

        cPickle.dump(train_X_indexed, open("train_X_"+str(i)+".cPickle","w"))
        cPickle.dump(test_X_indexed, open("test_X_"+str(i)+".cPickle","w"))
        cPickle.dump(train_y, open("train_y_"+str(i)+".cPickle","w"))
        cPickle.dump(test_y, open("test_y_"+str(i)+".cPickle","w"))
#         X_all += test_X_indexed
#         y_all += test_y
# #     print X_all
#     lr = LogisticRegression()
#     scores = cross_val_score(lr, X_all, y_all, cv=5, scoring='accuracy')
#     print scores.mean()
        
        


        
# ==========/ HELPER FUNCTIONS /========== #
def get_vocabulary(utterances):
    tknzr = TweetTokenizer()
    token_list = []
    for utt in utterances:
        utt_content = utt[0]
        token_list += nltk.wordpunct_tokenize(utt_content)
    token_set = set(token_list)
    return token_set
    
    
def get_features(utterances, ngram):
    features = []
    tknzr = TweetTokenizer()
    for utt in utterances:
        utt_content = utt[0]  # text content of the utterance
        utt_category = utt[1]  
    
        if ngram:  # use bow & ngram as feature
            # bow_list & 3-gram char list
            bow_list = tknzr.tokenize(utt_content)
            cgram_list = [utt_content[i:i+3] for i in range(len(utt_content)-1)] # 3-gram list
            feature_list = bow_list + cgram_list
        else:  # only use bow as feature
            feature_list = tknzr.tokenize(utt_content)
            
        if utt_category == 'QA':  # non-task
            features.append((feature_list, 0))
        else:  # task
            features.append((feature_list, 1))
            
    return features


def get_indexed_X(features, index_voc):
    indexed_features = []
    for feature_list in features:
        indexed_feature_list = []
        for token in feature_list:
            if token in index_voc:
                indexed_feature_list.append(index_voc[token])
            else:
                index_voc[token] = len(index_voc)
                indexed_feature_list.append(index_voc[token])
        indexed_features.append(indexed_feature_list)
    return indexed_features
    
if __name__ == '__main__':
    main()