#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:46:28 2017

@author: zhouyi
"""

ndcg_im=[0.0225859638359]*16
ndcg_ExpoMF=[0,0.0114,
0.0160,
0.0168,
0.0177,
0.0191,
0.0194,
0.0201,
0.0203,
0.0199,
0.0196,
0.0194,
0.0195,
0.0190,
0.0190,
0.0189]
ndcg_ExpoMF_conf=[0,0.02807,
0.05114,
0.05427,
0.05419,
0.05336,
0.05306,
0.05281,
0.05254,
0.05245,
0.05251,
0.05247,
0.05265,
0.05309,
0.05312,
0.05332]
t = np.arange(0,16,1)


plt.figure()
plt.plot(t,ndcg_ExpoMF,"g-.",label="ExpoMF",linewidth=2)
plt.plot(t,ndcg_ExpoMF_conf,"r-",label="ExpoMF with confidence",linewidth=2)
plt.plot(t,ndcg_im,"b-x",label="WMF",linewidth=2)

plt.axis([0,16,0.00,0.04])
plt.xlabel("Iteration")
plt.ylabel("NDCG@20")
plt.title("Performance for TPS Validation data when trainning")

plt.grid(True)
plt.legend()
plt.show()