#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 02:12:34 2017

@author: zhouyi
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.xlabel('Users')
plt.ylabel('Movies')
plt.title('Expected Exposure priors $\mu_{ui}$') 
sns.heatmap(exposure[1:30,1:30])
#plt.savefig('snsfig)