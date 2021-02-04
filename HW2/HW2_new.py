#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:33:40 2021

@author: axel
"""
import matplotlib.pyplot as plt
import numpy as np

# Read file
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

#basic vector stuff
def ones(v):
    return np.ones(len(v))

def vec_sum(v):
    sum = 0
    for x in v:
        sum += x
    return sum

#L2 norm of vector
def vec_len(v):
    return (v.dot(v))**.5

#checks whether distance between two vectors is
#less than or equal to some (small) distance d
def close(v,u,d):
    if vec_len(v - u) > d:
        return False
    return True

#finds min of a vector
def min_vec(v):
    z = v[0]
    for i in range(len(v)):
        if v[i] < z:
            z = v[i]
            pos = i
    return (z , pos)

#deleting rows/columns
def delete_row(M,i):
    s = []
    for j in range(len(M)):
        if j != i:
          s.append(M[j])
    return np.array(s)

def delete_col(M,i):
    return (delete_row(M.transpose(),i)).transpose()
        
#picking out rows/columns as vectors
def choose_row(M,i):
    return M[i]

def choose_col(M,i):
    return choose_row(M.transpose(),i)

#standardisation/normalisation for matrices
def standardise(mat):
    s = []
    for v in mat.transpose():
        s.append(v / vec_len(v))
    return np.array(s).transpose()


def center(mat):
    s = []
    for v in mat.transpose():
        l = vec_sum(v)/len(v) * ones(v)
        s.append(v - l)
    return np.array(s).transpose()

#soft thesholding
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

# (full) residual
def res(X,Y,beta):
    r = []
    N = len(Y)
    p = len(X.transpose())
    for i in range(N):
        sum_i = 0
        for j in range(p):
            sum_i += X[i,j] * beta[j]
        r.append(Y[i]-sum_i)
    return np.array(r)

#finds "\lambda_\max" for X and Y
def lambdamax(X,Y):
    N = len(Y)
    p = len(X.transpose())
    max_val = 0
    for j in range(p):
        c = abs(choose_col(X, j).dot(Y))
        if c > max_val:
            max_val = c
    return max_val/N


#CCD as a recursive function
def CCD_recursor(X,Y,beta,lam):
    b = []
    N = len(Y)
    p = len(X.transpose())
    r = res(X,Y,beta)
    for j in range(p):
        c = beta[j] + 1/N * choose_col(X,j).dot(r)
        b.append(soft_threshold(c, lam))
    return np.array(b)

def CCD(X,Y,beta,lam):
    new_beta = CCD_recursor(X,Y,beta,lam)
    if close(beta,new_beta,.1/10):
        return new_beta
    else:
        return CCD(X,Y,new_beta,lam)


def leave_out(X,Y,beta,lam):
    pred = []
    N = len(Y)
    for i in range(N):
        Xi = delete_row(X, i)
        Yi = np.concatenate((Y[0:i], Y[i+1:N]))
        new_beta = CCD(Xi,Yi,beta,lam)
        z = new_beta.dot(choose_row(X, i)) - Y[i]
        pred.append(z)
    return vec_len(np.array(pred))/N

def find_lambda(X,Y):
    beta = np.zeros(len(X.transpose()))
    vals = []
    z = lambdamax(X, Y)
    for i in range(1,41):
        print(i)
        k = i*z/40
        vals.append((leave_out(X, Y, beta, k)))
    return vals

def sort_insert(v,x):
    binbag1 = []
    binbag2 = []
    inserted = False
    if v == []:
        return [[x[0]], [x[1]]]
    for i in range(len(v[0])):
        if x[0] < v[0][i] and inserted == False:
            binbag1.append(x[0])
            binbag2.append(x[1])
            inserted = True
        binbag1.append(v[0][i])
        binbag2.append(v[1][i])
    if inserted == False:
            binbag1.append(x[0])
            binbag2.append(x[1])
    return [binbag1, binbag2]
    
def sort(v):
    binbag = []
    for i in range(len(v[0])):
        binbag = sort_insert(binbag , (v[0][i] , v[1][i]))
    return binbag

    
def pick_smallest(X,Y,v):
    v_copy = v
    ind = 1
    smallest = v_copy[0]
    for i in range(len(v)):
        if v_copy[i] < smallest:
            smallest = v_copy[i]
            ind = i + 1
    return (lambdamax(X, Y) * ind / 40)

def main():
    f = open("crime","r").readlines()
    mat = standardise(center(read_file(f)))
    Y = choose_col(mat,0)
    X = delete_col(delete_col(mat,0),0)
    
    lambda_devs = find_lambda(X, Y)
    
    plt.xlabel("i")
    plt.ylabel('Average deviation for $ \\frac{\lambda_\max / i } {40}$')
    plt.plot(range(1,len(lambda_devs)+1) , lambda_devs)
    plt.xticks([1] + list(map(lambda x : 2*x, range(1,21))))
    plt.xlim(xmin=1)
    plt.show()
    plt.savefig('lambdas.jpg')
    
    
    lambdamin = pick_smallest(X, Y, lambda_devs)
    beta = CCD(X, Y, np.array([0,0,0,0,0]), lambdamin)
    res = list(map(lambda s : s.dot(beta) , X))
    
    #sorts Y and the "resut" list res, so that res in increasing. Cooler plot...
    resY = sort([res,Y])
    
    plt.scatter(range(1,len(Y)+1),resY[1], label='Data')
    plt.plot(range(1,len(Y)+1),resY[0] , color = "r" , label = "Prediction")
    plt.legend()
    plt.savefig('compare.jpg')
    plt.show()

main()
