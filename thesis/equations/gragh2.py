#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:16:27 2017

@author: zhouyi
"""
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

'''
fig, ax = plt.subplots(nrows=2,ncols=2,sharey=True)

x = np.linspace(-1,
                3, 100)

plt.subplot(2,2,1)
plt.plot(x, np.power(norm.pdf(x,loc=1),1),
       'r-', lw=1, alpha=0.6, label='norm pdf')
plt.subplot(2,2,2)
plt.plot(x, np.power(norm.pdf(x,loc=1),0.75),
       'r-', lw=1, alpha=0.6, label='norm pdf')
plt.subplot(2,2,3)
plt.plot(x, np.power(norm.pdf(x,loc=1),0.5),
       'r-', lw=1, alpha=0.6, label='norm pdf')
plt.subplot(2,2,4)
plt.plot(x, np.power(norm.pdf(x,loc=1),0.25),
       'r-', lw=1, alpha=0.6, label='norm pdf')
'''

fig, ax = plt.subplots(nrows=1,ncols=1)
x = np.linspace(-1,3, 100)
plt.plot(x, np.power(norm.pdf(x,loc=1),1),
       'red', lw=1, alpha=0.6, label='z=1')

plt.plot(x, np.power(norm.pdf(x,loc=1),0.75),
       'black', lw=1, alpha=0.6, label='z=0.75')
plt.plot(x, np.power(norm.pdf(x,loc=1),0.5),
       'green', lw=1, alpha=0.6, label='z=0.5')
plt.plot(x, np.power(norm.pdf(x,loc=1),0.25),
       'blue', lw=1, alpha=0.6, label='z=0.25')
plt.legend()