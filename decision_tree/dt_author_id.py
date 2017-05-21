#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import collections
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
print((int)(len(features_train[0]))) # number of features
# len2 = (int)(len(labels_train)/100)
#features_train = features_train[:len1] 
#labels_train = labels_train[:len2] 
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
t1 = time()
predictions = clf.predict(features_test)
print(collections.Counter(predictions).get(1))
print("prediction time:", round(time()-t1, 3), "s")
print (clf.score(features_test, labels_test))

#print((time()-t0))





#########################################################
### your code goes here ###


#########################################################


