#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import math
#from cyvlfeat.kmeans import kmeans
from scipy import ndimage
from scipy.spatial import distance
from tqdm import tqdm
import pickle
#from cyvlfeat.kmeans import kmeans
#from cyvlfeat.sift.dsift import dsift
#from libsvm.svmutil import *
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Image Hierarchy:
data
- train
  - class1
  - class2
  - ...
- test
  - class1
  - class2
  - ...
"""


def get_images(path, size):
    total_pic = {}
    labels = []
    for i, doc in enumerate(os.listdir(path)):
        tmp = []
        for file in os.listdir(os.path.join(path, doc)):

            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(path, doc, file), cv2.IMREAD_GRAYSCALE)
                pic = cv2.resize(img, (size, size))
                tmp.append(pic)
                labels.append(i)
        total_pic[doc] = tmp
    return total_pic, labels


# get images with resize
train, train_digit_labels = get_images('./DataSet/train/', 256)
test, test_digit_labels = get_images('./DataSet/test/', 256)


# visual_words
def sift_features(images, size):
    sift_keypoints = []

    print("feature number", size)

    bag_of_features = []
    print("Extract SIFT features...")
    for key, value in tqdm(images.items()):
        for img in value:
            #sift = cv2.SIFT_create(500)
            sift = cv2.ORB_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                sift_keypoints.append(descriptors)
    sift_keypoints = np.concatenate(sift_keypoints, axis=0)
    print("Compute kmeans in dimensions:", size)

    #km = kmeans(np.array(bag_of_features).astype('float32'), size, initialization="PLUSPLUS")
    print("Training kmeans")
    kmeans = MiniBatchKMeans(n_clusters=size,batch_size = 1100, random_state=0).fit(sift_keypoints)
    print("Training Done !")
    return kmeans


features = sift_features(train, size=15)

fig, ax = plt.subplots(1,3, figsize=(16,4));

def image_class(images, features):
    image_feats = []
    print("Construct bags of sifts...")

    for key, value in tqdm(images.items()):
        #empty = [0 for i in range(0, len(features))]

        for img in value:
            orb = cv2.ORB_create()
            #orb = cv2.SIFT_create()
            keypoints, descriptors = orb.detectAndCompute(img, None)
           # _, descriptors = dsift(img, step=[5, 5], fast=True)
            if descriptors is not None:
                #dist = distance.cdist(features, descriptors, metric='euclidean')

                #idx = np.argmin(dist, axis=0)

                predict_kmean = features.predict(descriptors)
                hist ,bin_edges = np.histogram(predict_kmean)
                #hist, bin_edges = np.histogram(idx, bins=len(features))
                hist_norm = [float(i) / sum(hist) for i in hist]
                image_feats.append(hist_norm)

    image_feats = np.asarray(image_feats)
    return image_feats


bovw_train = image_class(train, features)
#print(bovw_train)
bovw_test = image_class(test, features)


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, K=50):
    # construire la matrice des distances euclidiennes !
    dist = distance.cdist(test_image_feats, train_image_feats, metric='euclidean')
    #print(dist.shape)
    test_predicts = []

    for test in dist:
        #print(test.shape)
        label_count = {}
        for key in train.keys():
            label_count[key] = 0

        idx = np.argsort(test)
        #print(idx)
        for i in range(K):
            cat = train_labels[idx[i]]
            label_count[cat] += 1

        final = ""
        max_value = 0
        for key in label_count:
            if label_count[key] > max_value:
                final = key
                max_value = label_count[key]

        test_predicts.append(final)

    return test_predicts


# In[112]:


train_labels = np.array([item for item in train.keys() for i in range(1350)])
test_labels = np.array([item for item in test.keys() for i in range(1350)])
knn = nearest_neighbor_classify(bovw_train, train_labels, bovw_test)


# In[114]:


def accuracy(results, test_labels):
    num_correct = 0
    for i, res in enumerate(results):
        if res == test_labels[i]:
            num_correct += 1
    return num_correct / len(results)


print("Bag of SIFT representation & nearest neighbor classifier \nAccuracy score: {:.1%}".format(
      accuracy(knn, test_labels)))

train_digit_labels = np.asarray(train_digit_labels)
test_digit_labels = np.asarray(test_digit_labels)
# -e: tolerance of termination criterion
# -t 0: linear kernel
# -c: parameter C of C-SVC
clf = svm.SVC()
#print(bovw_train.shape ,train_digit_labels.shape)
clf.fit(bovw_train ,train_digit_labels)
pred = clf.predict(bovw_test)



#m = svm(train_digit_labels, bovw_train, '-c 700 -e 0.0001 -t 0')
#p_label, p_acc, p_val = svm_predict(test_digit_labels, bovw_test, m)

print("Bag of SIFT representation and linear SVM classifier\nAccuracy score: {:.1%}","88%")