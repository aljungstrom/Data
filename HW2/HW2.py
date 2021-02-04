#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:57:57 2021

@author: axel
"""

import numpy as np

def read_file(f):
    mat = []
    count = False
    for r in f :
        if count:
            r_s = (r.replace('\t','').replace('\n', '')).split(" ")
            vec = []
            for s in r_s:
                vec.append(float(s))
            mat.append(vec)
        count = True
    return np.array(mat)

def center_mat(mat):
    height = len(mat[:,0])
    width = len(mat[0])
    sum = 0
    for i in range(width):
        for j in range(height):
            sum += mat[j,i]
        for j in range(height):
            mat[j,i]= mat[j,i] - sum/height
        sum = 0
    return mat

def split_at_first_row(mat):
    return (mat.transpose()[1:len(mat[0])]).transpose() , (np.array([mat[:,0]])).transpose()

def standardise_and_center(mat):
    (X,Y) = split_at_first_row(mat)
    return standardise_mat(center_mat(mat)) , standardise_mat(center_mat(Y))

def standardise_mat(mat):
    height = len(mat[:,0])
    width = len(mat[0])
    mat_copy = np.zeros((height,width))
    sum = 0
    for i in range(width):
        for j in range(height):
            sum += (mat[j,i])**2
            #print(sum)
        if sum > 0:
            for j in range(height):
                mat_copy[j,i]= mat[j,i]/(sum**.5)
        sum = 0
    return mat_copy

def sum_vector_mat(vec):
    sum = 0
    for x in vec:
        sum += x
    return sum

# Soft threshold:
# ------
def sign(x):
    if x < 0:
        return -1
    else:
        return 1

def trunc(x):
    if x < 0:
        return 0
    else:
        return x


def soft_threshold(x,lam):
    return sign(x)*trunc(abs(x)-lam)
# ------

#--- (full) residual
#---
def res(Y,X,beta):
    N = len(Y[:,0])
    r = []
    for i in range(N):
        sumi = 0
        for j in range(len(X[0])):
            sumi = sumi + beta[0][j]*X[i][j]
        r.append(Y[i][0] - sumi)
    return np.array([r])

def l2(mat1,mat2):
    z = mat1[0] - mat2[0]
    return ((z*z.transpose())[0])**.5 

def close_to_zero(mat1,mat2,d):
    if l2(mat1,mat2) > d:
        return False
    return True

def CCD_aux(Y,X,b,lam):
    r = res(Y,X,b)
    N = len(Y[:,0])
    s = []
    for i in range(len(b[0])):
        s.append(soft_threshold((b[0][i] + (X[:,i].dot((r[0].transpose()))) /N), lam))
    return np.array([s])
           
def CCD(Y,X,beta,lam):
    new_beta = CCD_aux(Y, X, beta, lam)
    if close_to_zero(beta , new_beta, .0005):
        return new_beta
    else: 
        return CCD(Y,X,new_beta,lam)
        
def delete_row(A,i):
    B = []
    for s in range(len(A)):
        if s != i:
            B.append(A[s])
    return np.array(B)

def delete_column(A,i):
    return delete_row(A.transpose(), i).transpose()
            
def choose_row(A,i):
    return np.array([A[i]])

def choose_column(A,j):
    return np.array([A[:,j]])

def lambdamax(X,Y):
    p = len(X[0])
    N = len(X[:,0])
    val = 0
    for i in range(p):
        Xi = choose_column(X, i)
        new_val = abs(Xi.dot(Y)[0][0])
        if new_val > val:
            val = new_val
    return val/N

def eval_model(X,Y,lam):
    N = len(X[:,0])
    beta = np.array([[0,0,0,0,0,0]])
    devs = []
    for i in range(N):
        Xi = delete_row(X, i)
        Yi = delete_row(Y,i)
        b = CCD(Yi,Xi,beta,lam)
        print(b)
        y_pred = (b[0]).dot(X[i])

def main():
    f = open("crime","r").readlines()
    mat = read_file(f)
    (XX,Y) = standardise_and_center(mat)
    X = (XX.transpose()[1:7]).transpose()
    beta = np.array([[0,0,0,0,0,0]])
    eval_model(X,Y,.1)
    
    print(CCD(Y, X, beta, 1/10))
    
main()