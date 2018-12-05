#!/usr/bin/env python3
import pickle
import sys
import re
import os
import string
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import ensemble, metrics, model_selection, naive_bayes 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import nltk
from nltk.corpus import stopwords

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument ('--train', required=True, help = 'traning data path')
parser.add_argument ('--test', required=True, help = 'test data path')
parser.add_argument ('--pred', required=True, help = 'prediction file path')
args = parser.parse_args()

param = {'objective': 'binary:logistic',
        'n_estimators':1000,
        'learning_rate': 0.001,
        'max_depth': 3,
        'silent': 1,
        'subsample': 0.5,
        'colsample_bytree': 0.60,
        'seed': 2017,
        'reg_lambda': 1,
        'reg_alpha': 1
}

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None

col=['company','label','text']

def main():
    #Set seed for reproducibility
    np.random.seed(0)
    os.system('date')
    print("Loading data...")
    #Load the data from the CSV files
    training_data = load_data(args.train,col)
    test_data = load_data(args.test,col)
    train_y = training_data['label'].values.astype(int)
    train_id = training_data['company'].values
    test_id = test_data['company'].values
    N = train_y.shape[0]
    N1 = np.sum(train_y == 1)
    N0 = N - N1
    model_xgb = xgb.XGBClassifier(scale_pos_weight=1.0*N0/N1,**param)
    model_mnb = naive_bayes.MultinomialNB()
    #Meta feature analysis
    print ('Meta feature statistical...')
    os.system('date')
    metaFeature(training_data)
    metaFeature(test_data)

    print ("TFIDF Vectorize data ...")
    os.system('date')
    full_tfidf_w,train_tfidf_w,test_tfidf_w = TfidfV(training_data,test_data,'word')
    print ("TFIDF MNB model on word")
    os.system('date')
    pred_train,pred_test=cv(model_mnb,train_tfidf_w,train_y,test_tfidf_w)
    training_data['nb_tfidf_w_0'] = pred_train[:,0]
    training_data['nb_tfidf_w_1'] = pred_train[:,1]
    test_data['nb_tfidf_w_0'] = pred_test[:,0]
    test_data['nb_tfidf_w_1'] = pred_test[:,1]
    print ("SVD on TFIDF word")
    os.system('date')
    training_data,test_data = SVD(training_data,test_data,'word',full_tfidf_w,train_tfidf_w,test_tfidf_w)
    print ('Count Vectorize data...')
    os.system('date')
    train_count_w,test_count_w = CountV(training_data,test_data,'word') 
    print ("CountV MNB model on word")
    os.system('date')
    pred_train,pred_test=cv(model_mnb,train_count_w,train_y,test_count_w)  
    training_data['nb_countv_w_0'] = pred_train[:,0]
    training_data['nb_countv_w_1'] = pred_train[:,1]
    test_data['nb_countv_w_0'] = pred_test[:,0]
    test_data['nb_countv_w_1'] = pred_test[:,1]

    os.system('date')
    print ('Xgboost...')
    cols_to_drop = ['company','text','label']
    train_X = training_data.drop(cols_to_drop, axis=1).as_matrix()
    test_X = test_data.drop(cols_to_drop, axis=1).as_matrix()
    pred_X,pred_x=cv(model_xgb,train_X,train_y,test_X)

    os.system('date')
    print ('Writing prediction to prediction.csv')
    '''
    train_X_df = pd.DataFrame(training_data.drop(cols_to_drop,axis=1))
    train_X_df.to_csv("xgb_input.csv",index=False)
    '''
    out_df = pd.DataFrame(pred_x[:,1])
    out_df.columns = ['label']
    out_df.insert(0, 'company',test_id )
    out_df.to_csv(args.pred, index=False)
    print ('Finished')
    os.system('date')

def load_data(file_name,column_name):
    load_df = pd.DataFrame(columns=column_name)
    for pred_df in pd.read_csv(file_name,header=0,chunksize = 500):
        load_df = pd.concat([load_df,pred_df],axis=0)
    return load_df

def metaFeature(d):
    # Number of words in the text ##
    d["num_words"] = d["text"].apply(lambda x: len(str(x).split()))
    # Number of unique words in the text ##
    d["num_unique_words"] = d["text"].apply(lambda x: len(set(str(x).split())))
    # Number of characters in the text ##
    d["num_chars"] = d["text"].apply(lambda x: len(str(x)))
    # Number of stopwords in the text ##
    d["num_stopwords"] = d["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    # Number of punctuations in the text ##
    d["num_punctuations"] = d['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    # Number of title case words in the text ##
    d["num_words_upper"] = d["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    ## Number of title case words in the text ##
    d["num_words_title"] = d["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    ## Average length of the words in the text ##
    d["mean_word_len"] = d["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

### Fit transform the tfidf vectorizer ###
def TfidfV(train_df,test_df,split_type):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1),analyzer=split_type)
    full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    dic = tfidf_vec.get_feature_names()
    a = []
    for w in dic:
        x = w.encode('ascii','ignore').decode('ascii')
        a.append(x)
    filename = "TFIDF_dictionary_"+split_type
    f = open (filename+'.txt','wb')
    pickle.dump(a,open(filename,'wb'))
    '''
    for temp in a:
        f.write("%s\n" % temp)
    '''
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
    return full_tfidf, train_tfidf, test_tfidf 

### Fit transform the count vectorizer ###
def CountV(train_df,test_df,split_type):
    count_vec = CountVectorizer(stop_words='english', ngram_range=(1,1),analyzer=split_type)
    count_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    dic = count_vec.get_feature_names()
    train_count = count_vec.transform(train_df['text'].values.tolist())
    test_count = count_vec.transform(test_df['text'].values.tolist())
    return train_count,test_count

### SVD on TFIDF 
def SVD(train_df,test_df,split_type,full_tfidf,train_tfidf,test_tfidf):
    n_comp = 100  #value for LSA
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    #np.save("TFIDF_SVD.npy",svd_obj.components_)
    #np.save("TFIDF_SVD_comp.npy",train_svd)
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
        
    train_svd.columns = ['svd_'+split_type+'_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_'+split_type+'_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    return train_df,test_df

def cv(model,X,Y,x):
    K = 5
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=2017)
    i = 0
    Y_com = np.array([], dtype=np.float64)
    pred_com = np.array([], dtype=np.float64)
    X_label = np.array([], dtype=np.float64)
    #for train, val in kf.split(X):
    for train, val in kf.split(X,Y):
        i += 1
        model.fit(X[train, :], Y[train])
        pred = model.predict_proba(X[val,:])[:,1]

        print("auc %d/%d:" % (i,K),metrics.roc_auc_score(Y[val],pred))
        Y_com = np.append(Y_com,Y[val])
        pred_com = np.append(pred_com,pred)
        #X_label = np.append(X_label,X_id[val])
    print ("AUC: ", metrics.roc_auc_score(Y_com,pred_com))
    model.fit(X,Y)
    
    print('Predicting...')
    Y_pre = model.predict_proba(X)
    y_pre = model.predict_proba(x)
    '''
    if re.match('XGB',str(model)):
        return X_label,pred_com 
    else:
    '''
    return Y_pre,y_pre

if __name__ == '__main__':
    main()
