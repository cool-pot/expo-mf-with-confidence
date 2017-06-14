#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:03:54 2017

@author: zhouyi
"""
ndcg=[0,0.0112,0.0179,0.0208,0.0207,0.0217,0.0229,0.0237,0.0239,0.0246,0.0247,0.0248,0.0251,0.0251,0.0254,0.0254,0.0256,0.0255,0.0253,0.0255,0.0255]

ndcg_im=[0.0236]*21

t = np.arange(0,21,1)


plt.figure()
plt.plot(t,ndcg,"r-",label="ExpoMF",linewidth=2)
plt.plot(t,ndcg_im,"b-x",label="WMF",linewidth=2)

plt.axis([0,21,0.00,0.03])
plt.xlabel("Iteration")
plt.ylabel("NDCG@20")
plt.title("Performance for Steam Validation data when trainning")

plt.grid(True)
plt.legend()
plt.show()