#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:50:36 2017

@author: zhouyi
"""

import numpy as np



'''
    Calculate the normalized discounting gain for matrix factorization
    
    Refer to:
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
        
    Args:
        train_data: Matrix,shape(n_users,n_items). The data you use to train 
        held_out_data: Matrix,shape(n_users,n_items). The data you use to calculate NDCG scores
        theta: Matrix,shape(n_users,n_components), user factors
        beta: Matrix,shape(n_items,n_components), user factors
        mu: Matrix,shape(n_users,n_items), weight vectors for Rating ,if necessary
        k: Number of results to consider
        (R[u,i]=theta[u]*beta[i]*mu[u,i] or R[u,i]=theta[u]*beta[i])
    
    Return:
        average NDCG across users
        
'''

def NDCG_at_k(train_data,held_out_data,theta,beta,k,mu=None):
    
    assert train_data.shape == held_out_data.shape
    assert theta.shape[1] == beta.shape[1]
    assert theta.shape[0] == train_data.shape[0]
    assert beta.shape[0] == train_data.shape[1]
    
    n_users,n_items=train_data.shape
    
    user_NDCG_table=np.zeros(n_users)
    
    for u in range(n_users):
        DCG_at_k=0
        IDCG_at_k=0
        Rank_list=[]
        for i in range(n_items):
            '''To use the np.sort, preference should be negative'''
            Rank_list.append((i,0-theta[u].dot(beta[i])))
        table_dtype = [('item_index', int), ('preference', float)]
        Rank_Table=np.array(Rank_list,dtype=table_dtype)
        Rank_Table=np.sort(Rank_Table, order=['preference'])
        #print Rank_Table
        
        for top_k in range(k):        
            item_index=Rank_Table[top_k][0]
            #print (u,item_index)
            ''''held_out_data[u,item_index]!=0'''
            if held_out_data[u,item_index]> 0.1 :
                DCG_at_k+=1.0/np.log2(top_k+1+1)
            IDCG_at_k+=1.0/np.log2(top_k+1+1)
        
        user_NDCG_table[u]=DCG_at_k/IDCG_at_k
    
    #print user_NDCG_table
    return sum(user_NDCG_table)/n_users  

def weighted_NDCG_at_k(train_data,held_out_data,theta,beta,k,mu):
    
    assert train_data.shape == held_out_data.shape
    assert train_data.shape == mu.shape
    assert theta.shape[1] == beta.shape[1]
    assert theta.shape[0] == train_data.shape[0]
    assert beta.shape[0] == train_data.shape[1]
    
    n_users,n_items=train_data.shape
    
    user_NDCG_table=np.zeros(n_users)
    
    for u in range(n_users):
        DCG_at_k=0
        IDCG_at_k=0
        Rank_list=[]
        for i in range(n_items):
            '''To use the np.sort, preference should be negative'''
            Rank_list.append((i,0-theta[u].dot(beta[i])*mu[u,i]))
        table_dtype = [('item_index', int), ('preference', float)]
        Rank_Table=np.array(Rank_list,dtype=table_dtype)
        Rank_Table=np.sort(Rank_Table, order=['preference'])
        #print Rank_Table
        
        for top_k in range(k):        
            item_index=Rank_Table[top_k][0]
            #print (u,item_index)
            ''''held_out_data[u,item_index]!=0'''
            if held_out_data[u,item_index]> 0.1 :
                DCG_at_k+=1.0/np.log2(top_k+1+1)
            IDCG_at_k+=1.0/np.log2(top_k+1+1)
        
        user_NDCG_table[u]=DCG_at_k/IDCG_at_k
    
    #print user_NDCG_table
    return sum(user_NDCG_table)/n_users  









'''
    Calculate the Mean Average Precision for matrix factorization
    
    Refer to:
        https://www.kaggle.com/wiki/MeanAveragePrecision
        
        assuming: k << m
        ap@k=∑Precision(n)/k  , n=1...k
        map@k=Average(ap@k)
        
    Args:
        train_data: Matrix,shape(n_users,n_items). The data you use to train 
        held_out_data: Matrix,shape(n_users,n_items). The data you use to calculate MAP scores
        theta: Matrix,shape(n_users,n_components), user factors
        beta: Matrix,shape(n_items,n_components), user factors
        mu: Matrix,shape(n_users,n_items), weight vectors for Rating ,if necessary
        k: Number of results to consider
        (R[u,i]=theta[u]*beta[i]*mu[u,i] or R[u,i]=theta[u]*beta[i])
    
    Return:
        average ap@k across users
        
'''    
def MAP_at_k(train_data,held_out_data,theta,beta,k,mu=None):
    assert train_data.shape == held_out_data.shape
    assert theta.shape[1] == beta.shape[1]
    assert theta.shape[0] == train_data.shape[0]
    assert beta.shape[0] == train_data.shape[1]
    
    n_users,n_items=train_data.shape
    
    user_AP_table=np.zeros(n_users)
    
    for u in range(n_users):

        Rank_list=[]
        for i in range(n_items):
            '''To use the np.sort, preference should be negative'''
            Rank_list.append((i,0-theta[u].dot(beta[i])))
        table_dtype = [('item_index', int), ('preference', float)]
        Rank_Table=np.array(Rank_list,dtype=table_dtype)
        Rank_Table=np.sort(Rank_Table, order=['preference'])
        #print Rank_Table
        
        count=0.0
        AP_k=0.0
        for top_k in range(k):        
            item_index=Rank_Table[top_k][0]
            #print (u,item_index)
            ''''held_out_data[u,item_index]!=0'''
            if held_out_data[u,item_index]> 0.1 :
                count+=1.0
            Precesion_top_k=count/(top_k+1)
            AP_k+=Precesion_top_k/k
        user_AP_table[u]=AP_k
    
    #print user_AP_table
    return sum(user_AP_table)/n_users
    
def weighted_MAP_at_k(train_data,held_out_data,theta,beta,k,mu):
    assert train_data.shape == held_out_data.shape
    assert theta.shape[1] == beta.shape[1]
    assert theta.shape[0] == train_data.shape[0]
    assert beta.shape[0] == train_data.shape[1]
    
    n_users,n_items=train_data.shape
    
    user_AP_table=np.zeros(n_users)
    
    for u in range(n_users):

        Rank_list=[]
        for i in range(n_items):
            '''To use the np.sort, preference should be negative'''
            Rank_list.append((i,0-theta[u].dot(beta[i])*mu[u,i]))
        table_dtype = [('item_index', int), ('preference', float)]
        Rank_Table=np.array(Rank_list,dtype=table_dtype)
        Rank_Table=np.sort(Rank_Table, order=['preference'])
        #print Rank_Table
        
        count=0.0
        AP_k=0.0
        for top_k in range(k):        
            item_index=Rank_Table[top_k][0]
            #print (u,item_index)
            ''''held_out_data[u,item_index]!=0'''
            if held_out_data[u,item_index]> 0.1 :
                count+=1.0
            Precesion_top_k=count/(top_k+1)
            AP_k+=Precesion_top_k/k
        user_AP_table[u]=AP_k
    
    #print user_AP_table
    return sum(user_AP_table)/n_users
            
'''
    Calculate the Mean Recall for matrix factorization
    
    Refer to:
        https://en.wikipedia.org/wiki/Precision_and_recall#Recall
        
        Recall in information retrieval is the fraction of the documents that are 
        relevant to the query that are successfully retrieved.
        
        
        
        
    Args:
        train_data: Matrix,shape(n_users,n_items). The data you use to train 
        held_out_data: Matrix,shape(n_users,n_items). The data you use to calculate RECALL scores
        theta: Matrix,shape(n_users,n_components), user factors
        beta: Matrix,shape(n_items,n_components), user factors
        mu: Matrix,shape(n_users,n_items), weight vectors for Rating ,if necessary
        k: Number of results to consider
        (R[u,i]=theta[u]*beta[i]*mu[u,i] or R[u,i]=theta[u]*beta[i])
    
    Return:
        average Recall@k across users
        
'''           
def Recall_at_k(train_data,held_out_data,theta,beta,k,mu=None):
    assert train_data.shape == held_out_data.shape
    assert theta.shape[1] == beta.shape[1]
    assert theta.shape[0] == train_data.shape[0]
    assert beta.shape[0] == train_data.shape[1]
    
    n_users,n_items=train_data.shape
    
    user_Recall_table=np.zeros(n_users)
    
    for u in range(n_users):
        #print "user:",u

        Rank_list=[]
        for i in range(n_items):
            '''To use the np.sort, preference should be negative'''
            Rank_list.append((i,0-theta[u].dot(beta[i])))
        table_dtype = [('item_index', int), ('preference', float)]
        Rank_Table=np.array(Rank_list,dtype=table_dtype)
        Rank_Table=np.sort(Rank_Table, order=['preference'])
        #print Rank_Table
        
        rank_k_set=set()
        for top_k in range(k):        
            item_index=Rank_Table[top_k][0]
            rank_k_set.add(item_index)
        #print rank_k_set
    
        count_in_rank_k_set=0.0
        count_in_held_out_data=0.0
        for i in range(n_items):
            if held_out_data[u,i]> 0.1 :
                count_in_held_out_data+=1
                if i in rank_k_set:
                    count_in_rank_k_set+=1
        size=min(k,count_in_held_out_data)
        
        if size > 0.1:
            user_Recall_table[u]=count_in_rank_k_set/size
        else:
            user_Recall_table[u]=0
    
    #print user_Recall_table
    return sum(user_Recall_table)/n_users
            
def weighted_Recall_at_k(train_data,held_out_data,theta,beta,k,mu):
    assert train_data.shape == held_out_data.shape
    assert theta.shape[1] == beta.shape[1]
    assert theta.shape[0] == train_data.shape[0]
    assert beta.shape[0] == train_data.shape[1]
    
    n_users,n_items=train_data.shape
    
    user_Recall_table=np.zeros(n_users)
    
    for u in range(n_users):
        #print "user:",u

        Rank_list=[]
        for i in range(n_items):
            '''To use the np.sort, preference should be negative'''
            Rank_list.append((i,0-theta[u].dot(beta[i])*mu[u,i]))
        table_dtype = [('item_index', int), ('preference', float)]
        Rank_Table=np.array(Rank_list,dtype=table_dtype)
        Rank_Table=np.sort(Rank_Table, order=['preference'])
        #print Rank_Table
        
        rank_k_set=set()
        for top_k in range(k):        
            item_index=Rank_Table[top_k][0]
            rank_k_set.add(item_index)
        #print rank_k_set
    
        count_in_rank_k_set=0.0
        count_in_held_out_data=0.0
        for i in range(n_items):
            if held_out_data[u,i]> 0.1 :
                count_in_held_out_data+=1
                if i in rank_k_set:
                    count_in_rank_k_set+=1
        size=min(k,count_in_held_out_data)
        
        if size > 0.1:
            user_Recall_table[u]=count_in_rank_k_set/size
        else:
            user_Recall_table[u]=0
    
    #print user_Recall_table
    return sum(user_Recall_table)/n_users
        
        
            
'''                
#Test Script           
#3 user, 4 items, 5 components
train_data=np.random.rand(3,4)
mu=np.random.rand(3,4)
held_out_data=np.array([[1,1,1,0],[0,0,1,0],[1,0,1,0]])
theta=np.random.rand(3,5)
beta=np.random.rand(4,5)

print weighted_Recall_at_k(train_data,held_out_data,theta,beta,2,mu)
'''