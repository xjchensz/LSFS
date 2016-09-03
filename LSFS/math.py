#!usr/bin/python
# -*- coding:utf-8 -*-

def norm_2_1(a):
    """
    对每行的向量求第二范数，然后对所有范数求和
    """
    return np.sum(np.linalg.norm(a, ord = 2, axis=1))




"""
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
"""
def EProjSimplex_new(v, k=1):    
    v = np.matrix(v)
    ft = 1;
    n = np.max(v.shape)
    
#    if len(v.A[0]) == 0:
#        return v, ft
 
    if np.min(v.shape) == 0:
        return v, ft
    
#    print('n : ', n)
#    print(v.shape)
#    print('v :  ', v)
    
    v0 = v - np.mean(v) + k/n
    
#    print('v0 :  ', v0)
    
    vmin = np.min(v0)
    
    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10**-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if ft > 100:
                v1[v1<0] = 0.0
                break
        x = v1.copy()
        x[x<0] = 0.0
    else:
        x = v0
    return x,ft




