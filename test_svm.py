# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:36:28 2020

@author: timot
"""


import numpy as np
import numpy.matlib
import pandas as pd
import seaborn as sn
import pickle
import os
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from utils import *

#%% Inputs

# Data params
phi = 0
M = 5
# Test set 2
# train_dir = ['Patterns/30dB_200Its_M5_Size8_Phase_Train']
# Test set 3
train_dir = ['Patterns/Train/30dB_100Its_M5_Size8_Phase_Train','Patterns/Train/20dB_100Its_M5_Size8_Phase_Train',
             'Patterns/Train/10dB_100Its_M5_Size8_Phase_Train','Patterns/Train/0dB_100Its_M5_Size8_Phase_Train']
test_dir = 'Patterns/Test'

# Save name for classifier
clf_savename = 'svm_set3_pca200_GS_rbf' # model name for saving

# Preprocessing
do_cuts = False
do_hog = False
do_pca = True

# HOG params
orientations = 8
ppc = 8
cpb = 4
block_norm = 'L2'

# PCA Params
n_components = 200

#%% Preprocessing

# Generate training dataset
x_train, y_train = vectorize_and_label(train_dir, do_cuts=do_cuts, phi=0, do_hog=do_hog,
                                                     orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)

# Scale and PCA features
if do_hog:
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    pca = PCA(n_components=n_components)
    x_train_final = pca.fit_transform(x_train_norm)
    # Save model
    # pkl_filename = "set3_hog_scaler.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(scaler, file)
    # pkl_filename = "set3_hog_pca200_pca.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(pca, file)
elif do_pca:
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    pca = PCA(n_components=n_components)
    x_train_final = pca.fit_transform(x_train_norm)
    # Save model
    # pkl_filename = "set3_scaler.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(scaler, file)
    # pkl_filename = "set3_pca200_pca.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(pca, file)
else:
    x_train_final = x_train

# Shuffle training data
permutation = np.random.permutation(x_train_final.shape[0])
x_train_shuff = x_train_final[permutation,:]
y_train_shuff = y_train[permutation]

#%% Create classifier

C = [0.1, 0.5, 1, 5, 10]
# C = [1]
kernel = ['rbf']
gamma = ['scale']
coef0 = [0]
tol = 1e-3
max_iter = -1
decision_function_shape = ('ovr')

parameters = {'kernel':kernel, 'gamma':gamma, 'C':C, 'coef0':coef0}

# svm_clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,\
#                   tol=tol, max_iter=max_iter, decision_function_shape=decision_function_shape)
    
svm = svm.SVC(tol=tol, max_iter=max_iter)
svm_clf = GridSearchCV(svm, parameters)

#%% Run classifier

# Train data
svm_clf.fit(x_train_shuff, y_train_shuff)

# Save model
pkl_filename = clf_savename+'.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(svm_clf, file)


#%% Test data

# Load models
# with open(r'svm_set3_hog_pca200_C1_rbf.pkl', 'rb') as file:
#     svm_clf = pickle.load(file)
# with open(r'set3_hog_scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)
with open(r'Models/set3_pca200_pca.pkl', 'rb') as file:
    pca = pickle.load(file)

if type(test_dir) is str:
    test_dirs = ['Patterns/Test/'+i_dir for i_dir in os.listdir('Patterns/Test')]
else:
    test_dirs = test_dir

accuracy = []
dB_str = []
for i_dir in test_dirs:
    dB_str.append(i_dir[14:16])
    dB = int(i_dir[14:16])
    x_test, y_test = vectorize_and_label([i_dir], do_cuts=do_cuts, phi=0, do_hog=do_hog,
                                                         orientations=orientations, ppc=ppc, cpb=cpb, block_norm=block_norm)
    
    if do_hog:
        x_test_norm = scaler.transform(x_test)
        x_test_final = pca.transform(x_test_norm)
    elif do_pca:
        x_test_norm = scaler.transform(x_test)
        x_test_final = pca.transform(x_test_norm)
    else:
        x_test_final = x_test
    
    predictions = svm_clf.predict(x_test_final)
    class_report = classification_report(y_test, predictions)
    print(class_report)
    i_accuracy = len(np.where((predictions-y_test)==0)[0])/y_test.size
    accuracy.append(i_accuracy)
    
    conf_mat = confusion_matrix(y_test,predictions)
    df_cm = pd.DataFrame(conf_mat, index = np.arange(0,M*M+1), columns = np.arange(0,M*M+1))
    plt.figure(figsize = (10,7))
    plt.title('Confusion Matrix, SNR: {SNR} dB'.format(SNR=dB))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.show()
    
accuracy = np.array(accuracy)
dB_int = np.array([int(idB) for idB in dB_str])
accuracy_sorted = 100*accuracy[np.argsort(dB_int)]
np.savetxt(clf_savename+'.csv',accuracy_sorted,delimiter=',')
plt.figure()
plt.plot(np.sort(dB_int),accuracy_sorted,'-o')
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.yticks(np.arange(0,110,10))
plt.show()