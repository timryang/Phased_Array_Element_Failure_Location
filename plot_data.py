# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:54:46 2020

@author: timot
"""


import numpy as np
from matplotlib import pyplot as plt

svm_cuts = np.genfromtxt('Data\svm_set3_cuts_C1_rbf.csv', delimiter=',')
svm_pca = np.genfromtxt('Data\svm_set3_pca200_C1_rbf.csv', delimiter=',')
svm_pca_gs = np.genfromtxt('Data\svm_set3_pca200_GS_rbf.csv', delimiter=',')
svm_hog_pca = np.genfromtxt('Data\svm_set3_hog_pca200_C1_rbf.csv', delimiter=',')
svm_hog_pca_gs = np.genfromtxt('Data\svm_set3_hog_pca200_GS_rbf.csv', delimiter=',')
cnn = np.genfromtxt('Data\cnn_vals.csv', delimiter=',')
mlp = np.genfromtxt(r'Data\accuracy_mlp.csv', delimiter=',')
rfc = np.genfromtxt(r'Data\accuracy_rfc.csv', delimiter=',')

SNR = np.arange(-4,22,2)

plt.figure()
plt.grid()
plt.plot(SNR,svm_cuts,label='Cuts')
plt.plot(SNR,svm_pca,label='PCA')
plt.plot(SNR,svm_hog_pca,label='HOG')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.yticks(np.arange(0,110,10))
plt.title('SVM Preprocessing Performance')
plt.legend()
plt.show()

plt.figure()
plt.grid()
plt.plot(SNR,svm_pca,label='PCA')
plt.plot(SNR,svm_pca_gs,label='PCA w/GS')
plt.plot(SNR,svm_hog_pca,label='HOG')
plt.plot(SNR,svm_hog_pca_gs,label='HOG w/GS')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.yticks(np.arange(0,110,10))
plt.title('SVM GridSearch Performance')
plt.legend()
plt.show()

plt.figure()
plt.grid()
plt.plot(SNR,svm_pca_gs,label='SVM')
plt.plot(SNR,cnn,label='AlexNet')
plt.plot(SNR,mlp,label='Multilayer Perceptron')
plt.plot(SNR,rfc,label='Random Forest')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy (%)')
plt.yticks(np.arange(0,110,10))
plt.title('Model Performance')
plt.legend()
plt.show()