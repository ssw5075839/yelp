# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:35:40 2016

@author: shiwei
"""
import argparse
import pandas as pd
import numpy as np
import time
import xgboost as xgb

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

#customized xgboost sklearn interface
class XGBLog (xgb.XGBModel):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0,threshold = 0.5,missing=np.nan):
        xgb.XGBModel.__init__(self, max_depth, learning_rate, n_estimators,
                 silent, objective, nthread, gamma, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, colsample_bylevel,
                 reg_alpha, reg_lambda, scale_pos_weight,
                 base_score ,seed, missing=np.nan)
        self.threshold = threshold
        self._estimator_type = "regressor"
    #customized prediction function for easier ensemble 
    def predict(self, data):
        return (xgb.XGBModel.predict(self, data) > self.threshold).astype(int)
    def predict_proba(self, data):
        p = xgb.XGBModel.predict(self, data)
        return np.vstack([1-p,p]).T

def parse_args():
    parser = argparse.ArgumentParser(description='ensemble xgboost and svm model on business features')
    parser.add_argument('--svm_pca_n', type=int, default=200,
                        help='the number of pca component you want to project to \
                        reduce dimension of features for svm. Default is 200 and it \
                        gives satisfactory results. It should be smaller than the \
                        CNN extracted feature dimension, 2048 in the case of res5c \
                        layer of ResNet-152')
    parser.add_argument('--coef_n', type=int, default=10,
                        help='how many coefficients do you want to search for the \
                        linear combination of xgb and svm. Deafult is 10 and will search 0, \
                        0.1, ..., 1.0')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='the probability threshold to determine if label is 1. Default is 0.5')
    parser.add_argument('--n_fold', type=int, default=10,
                        help='the number of folds to cross-validate. Default is 10')
    return parser.parse_args()
  
def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]

def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]

def f1(y_predicted, y_true, threshold):
    return ("f1_score",f1_score(y_true,(y_predicted>threshold).astype(int), average='weighted'))
            
if __name__ == '__main__':
    args = parse_args()
    
    #load data
    data_root = '/media/shiwei/data/yelp/'

    print 'loading data...'
    train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
    train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv', index_col='photo_id')
    
    train_df = pd.read_csv(data_root+"train_biz_res5c_features.csv")
    test_df  = pd.read_csv(data_root+"test_biz_res5c_features.csv")
    
    y_train = train_df['label'].values
    X_train = train_df['feature vector'].values
    X_test = test_df['feature vector'].values   
    
    y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
    X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
    X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])
    #Here, I find that PCA the business features actually help regularize svm and
    #and improve svm performance
    pca=PCA(n_components=args.svm_pca_n)
    pca.fit(X_train)
    X_t=pca.transform(X_train)
    X_te = pca.transform(X_test)
    
    mlb = MultiLabelBinarizer()
    y_Btrain= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix
    
    '''
    
    PART V: Exhaustive search on ensemble coefficient
    
    '''      
    #we are going to train an xgb and an svm model then linearly combine their predicted probability
    #We are going to use cross-validation and exhaustive search to find the best coefficient.
    threshold = args.threshold
    coef_n = args.coef_n
    coefs = np.linspace(0,1,1+coef_n)
    ensemble_score = np.zeros((args.n_fold,1+coef_n))

    print 'Start Kfolds cross-validation to find best combination coefficient:'
    seeds = [211,1018,1120,5075839,915] #my lucky number,you are free to edit here. Use multiple seeds to average random effects.
    for seed in seeds:
        kf = KFold(n_folds=args.n_fold,shuffle=True, random_state=seed)
        kf.get_n_splits(X_train)
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            X_train_fold = X_train[train_index]
            X_test_fold = X_train[test_index]
            y_train_fold = y_Btrain[train_index]
            y_test_fold = y_Btrain[test_index]
            X_t_fold = pca.transform(X_train_fold)
            X_te_fold = pca.transform(X_test_fold)
            #for each fold, train an xgb and svm and then record their results on test set with each combination coefficient
            #coef=0 means the svm only results and coef=1 means xgb only results
            
            #1.train xgb classifier
            #generally speaking, these parameters for xgboost need to be carefully tuned
            #and cross-validated. Here I just use what I get after hyper-parameter tuning.
            #for complete guide to xgboost hyper-parameter tuning, please refer to:
            #https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
            a= XGBLog(       max_depth=5, learning_rate=0.1, n_estimators=300,
                            silent=True, objective="binary:logistic",
                            nthread=4, gamma=0, min_child_weight=1, max_delta_step=0,
                            subsample=1, colsample_bytree=1, colsample_bylevel=1,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                            base_score=0.5, seed=0,threshold = 0.5, missing=np.nan
                   )
        
            random_state = np.random.RandomState(0)
            classifier = OneVsRestClassifier(a)
            classifier.fit(X_train_fold, y_train_fold)
        
            #y_predict_xgb is the xgboost probability prediction for each business    
            y_predict_xgb = classifier.predict_proba(X_test_fold)
            
            #2.train svm classifier
            #generally speaking, these parameters for xgboost need to be carefully tuned
            #and cross-validated. Here I just use what I get after hyper-parameter tuning.
            #for complete guide to xgboost hyper-parameter tuning, please refer to:
            #https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
            random_state = np.random.RandomState(0)
            classifier2 = OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=0.005,probability=True),n_jobs=4)
            classifier2.fit(X_t_fold, y_train_fold)
            y_predict_svm = classifier2.predict_proba(X_te_fold)
            
            #3linearly combine them
            for j,coef in enumerate(coefs):
                y_predict = coef*y_predict_xgb+(1-coef)*y_predict_svm
                ensemble_score[i,j] += f1(y_predict, y_test_fold, threshold)[1]
    ensemble_score /= len(seeds)        
    best_coef = coefs[np.argmax(ensemble_score.mean(axis=0))]
    print 'cross-validation score over different coefficient:'
    print ensemble_score.mean(axis=0)
    print '{0} folds cross-validation shows {1} is best combination \
    coefficient'.format(args.n_fold, best_coef)
    


    '''
    
    PART VI: Xgboost on business features
    
    '''
    
    t = time.time()
    #generally speaking, these parameters for xgboost need to be carefully tuned
    #and cross-validated. Here I just use what I get after hyper-parameter tuning.
    #for complete guide to xgboost hyper-parameter tuning, please refer to:
    #https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    a= XGBLog(       max_depth=5, learning_rate=0.1, n_estimators=300,
                     silent=True, objective="binary:logistic",
                     nthread=4, gamma=0, min_child_weight=1, max_delta_step=0,
                     subsample=1, colsample_bytree=1, colsample_bylevel=1,
                     reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                     base_score=0.5, seed=0,threshold = 0.5, missing=np.nan
            )
    
    random_state = np.random.RandomState(0)
    classifier = OneVsRestClassifier(a)
    classifier.fit(X_train, y_Btrain)
    
    #y_predict_xgb is the xgboost probability prediction for each business    
    y_predict_xgb = classifier.predict_proba(X_test)

    print "Time passed for Xgb: ", "{0:.1f}".format(time.time()-t), "sec"
    '''
    
    PART VII: SVM on business features
    
    '''  

    t = time.time()
       
    #generally speaking, svm hyper-parameter also need to tuned. Here I just post
    #what I got after tuning, for svm hyper-parameter tuning tutorial, please refer to:
    #http://scikit-learn.org/stable/modules/grid_search.html
    random_state = np.random.RandomState(0)
    classifier2 = OneVsRestClassifier(svm.SVC(kernel='rbf',gamma=0.005,probability=True),n_jobs=4)
    classifier2.fit(X_t, y_Btrain)
    y_predict_svm = classifier2.predict_proba(X_te)
    print "Time passed for svm: ", "{0:.1f}".format(time.time()-t), "sec"    
    
    '''
    
    PART VIII: combine xgb and svm results on the full training set and submit
    
    '''
    y_predict = ((best_coef*y_predict_xgb+(1-best_coef)*y_predict_svm)>threshold).astype(int)
    y_predict_label = mlb.inverse_transform(y_predict) #Convert binary matrix back to labels
    test_data_frame  = pd.read_csv(data_root+"test_biz_res5c_features.csv")
    df = pd.DataFrame(columns=['business_id','labels'])
    
    for i in range(len(test_data_frame)):
        biz = test_data_frame.loc[i]['business']
        label = y_predict_label[i]
        label = str(label)[1:-1].replace(",", " ")
        df.loc[i] = [str(biz), label]
    
    with open(data_root+"submission_res5c_xgb_svm_ensem.csv",'w') as f:
        df.to_csv(f, index=False)  