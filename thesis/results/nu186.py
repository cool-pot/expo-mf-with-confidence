#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 02:04:40 2017

@author: zhouyi
"""

#268::F::18::12::29708
#女性，18-25岁，程序员
#nu[186]

import numpy as np  
import matplotlib.pyplot as plt 

genre=["Action",
"Adventure",
"Animation",
"Children's",
"Comedy",
"Crime",
"Documentary",
"Drama",
"Fantasy",
"Film-Noir",
"Horror",
"Musical",
"Mystery",
"Romance",
"Sci-Fi",
"Thriller",
"War",
"Western"]

bar_width = 0.8
n_groups = 18 
fig, ax = plt.subplots(figsize=(18, 6))  
index = np.arange(n_groups)  

plt.bar(index,nu186,bar_width,color='b',alpha=0.5)
plt.xlabel('Movie genres')
plt.ylabel('Expected exposure rate to movie genres')
plt.xticks(index, genre)
plt.title('Female,Programmer,Age 18-25') 