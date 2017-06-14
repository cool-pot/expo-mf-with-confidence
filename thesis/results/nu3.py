#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 01:35:53 2017

@author: zhouyi
"""

#nu[3]
#5::M::25::20::55455
# 男性，25-34，作家
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

plt.bar(index,nu3,bar_width,color='r',alpha=0.5)
plt.xlabel('Movie genres')
plt.ylabel('Expected exposure rate to movie genres')
plt.xticks(index, genre)
plt.title('Male,Write,Age 25-34') 