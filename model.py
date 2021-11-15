#pandas numpy
import pandas as pd
import numpy as np
from numpy import mean

#sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

#imblearn for imbalanced data
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#matplotlib
import matplotlib.pyplot as plt

#counters
from collections import Counter

#others (os, graphwiz, excels)
import graphviz 
import os
import xlrd
import openpyxl
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

global model

def func():
    global ensemble
    dir = 'c:\\Users\\huyho\\OneDrive\\Desktop\\stuff\\student-c'
    path = os.path.join(dir, "RFEP Worksheet Fall20-21 Sample.xlsx")

    xls = pd.ExcelFile(path)
    student_list_df = pd.read_excel(xls, sheet_name=1)
    wpa_data_df = pd.read_excel(xls, sheet_name=2)
    
    #x = features
    #y = target
    x = student_list_df[['Oral Level', 'Written Level', 'Listening', 'Speaking', 'Reading', 'Writing' , 'Current GPA', '18-19 Overall ELPAC']]
    y = student_list_df['RFEP Overall Eligible']
    x = x.fillna(0) #fill null values

    x = x.values
    y = y.values

    #Smote them due to imbalanced data
    over = SMOTE(sampling_strategy=0.1) #oversample the minority class to 1:10 ratio
    under = RandomUnderSampler(sampling_strategy=0.2) #undersample the majority class to a 1:2 ratio
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    x, y = pipeline.fit_resample(x, y)

    
    #split to training and testing
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
    
    #scale
    scaler = StandardScaler()
    scaler.fit(x)
    
    scaled_train = scaler.transform(xtrain)
    scaled_test = scaler.transform(xtest)

    #change rfep overall eligible from Yes -> 1 and No -> 0
    student_list_df['RFEP Overall Eligible'] = student_list_df['RFEP Overall Eligible'].replace({'Yes': 1, 'No': 0})
    
    #choose 3 models. Maybe 5 will be better idk
    e_svc = svm.SVC(kernel='linear')
    e_dectree = tree.DecisionTreeClassifier()
    e_NB = GaussianNB()
    e_Ada = AdaBoostClassifier()
    e_rf = RandomForestClassifier()
    
    #put the 3 models into an ensemble method 
    ensemble = VotingClassifier(estimators=[('svc', e_svc), ('dectree', e_dectree), 
                                            ('NB', e_NB), ('Ada', e_Ada), ('RF', e_rf)], voting='hard') #voting ='soft' will use probabilities (might use later?)
    ensemble.fit(scaled_train, ytrain)
    
    model = ensemble
    return model


        

