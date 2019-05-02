import csv
import os
import pickle
import sys

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def main():
    if(sys.argv[2] == "train" and sys.argv[1] =="--mode"):
       # print("in train mode")
        filename = sys.argv[4]
        file = open(filename, "r", encoding = 'UTF-8')
        #reader = csv.reader(file)
        data = pd.read_csv(file)

        feature = data['text']
        classcol = data['label']
        tf = TfidfVectorizer()
        #split data 75% for training and 25% for testing
        train_x, test_x, train_y, test_y = train_test_split(feature, classcol, train_size = .75,test_size = .25)
        
        #transform data for accurate classification
        train_x = tf.fit_transform(train_x)
        test_x = tf.transform(test_x)
        joblib.dump(tf,"tfidf.pkl")

        #Testing other classification methods
        #knn = KNeighborsClassifier(n_neighbors=5)
        #knn.fit(train_x,train_y)
        #tree = DecisionTreeClassifier()
        #tree.fit(train_x,train_y)

        #use linear support vector classification and save to pkl file
        linearsvm = LinearSVC().fit(train_x, train_y)
        joblib.dump(linearsvm, 'SVM.pkl')
        pred_y = linearsvm.predict(test_x)

        #print report and confusion matrix
        print("\nClassification Report:\n")
        print(metrics.classification_report(test_y,pred_y))
        prediction = linearsvm.predict(test_x)
        print("\nConfusion matrix:\n")
        print(pd.crosstab(test_y, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
        file.close()

    elif(sys.argv[2] == "cross_val" and sys.argv[1] =="--mode"):
        #print("in cross_validation mode")
        filename = sys.argv[4]
        file = open(filename, "r", encoding = 'UTF-8')
        data = pd.read_csv(file)

        feature = data['text']
        classcol = data['label']
        tf = TfidfVectorizer()
        train_x = tf.fit_transform(feature)
        linearsvm = LinearSVC()
        
        #use 10 fold cross validation with linear support vector model and print results
        scores = cross_val_score(linearsvm,train_x,classcol,cv=10)
        print("\nCross-validation scores: {}".format(scores))
        print("\nAverage cross-validation score: {:.2f}\n".format(scores.mean()))
        file.close()


    elif(sys.argv[2] == "predict" and sys.argv[1] =="--mode"):
        #print("in predict mode")
        sentence = sys.argv[4]

        #load pkl data
        trained_model = joblib.load('SVM.pkl')
        tf = joblib.load("tfidf.pkl")
        sentence = [sentence]
        sentence = tf.transform(sentence)

        #use predict function on sentence to get class prediction
        prediction = trained_model.predict(sentence)
        print()
        for k in prediction:
            print(k)

main()

#examples from assignment page
#naive bayes example
def naivebayes():
    cancer = load_breast_cancer()
    train_feature, test_feature, train_class, test_class = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
    nb = GaussianNB().fit(train_feature, train_class)
    print("Test set score: {:.3f}".format(nb.score(test_feature, test_class)))
#linear support vector example
def linsupvec():
    cancer = load_breast_cancer()
    train_feature, test_feature, train_class, test_class = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
    linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class)
    print("Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))

#k nearest neighbor and graph example
def knearneighbor():
    cancer = load_breast_cancer()
    print("cancer.keys(): {}".format(cancer.keys()))
    print("Shape of cancer data: {}".format(cancer.data.shape))
    print("Sample counts per class:\n{}".format( {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

    print(cancer.feature_names,cancer.target_names)
    for i in range(0,3):
        print(cancer.data[i], cancer.target[i])

    train_feature, test_feature, train_class, test_class = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_feature,train_class)
    print("Test set predictions:\n{}".format(knn.predict(test_feature)))
    print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))

    cancer = load_breast_cancer()
    train_feature, test_feature, train_class, test_class = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
    training_accuracy = []
    test_accuracy = []
    # try n_neighbors from 1 to 10.
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # build the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_feature, train_class)
        # record training set accuracy
        training_accuracy.append(knn.score(train_feature, train_class))
        # record generalization accuracy
        test_accuracy.append(knn.score(test_feature, test_class))

    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()
