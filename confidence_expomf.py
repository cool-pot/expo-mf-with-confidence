#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:31:35 2017

@author: zhouyi
"""

import numpy as np
from math import sqrt
import sys
import evaluation

#A naive implementation

floatX = np.float32



class CONFIDENCE_EXPO_MF():
    def __init__(self, n_components=100, init_std=0.01, lam_y=1.0, lam_theta=1e-5, lam_beta=1e-5, init_mu=0.01):
        '''
        Exposure matrix factorization
        
        Parameters
        ---------
        n_components : int
            Number of latent factors

        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        '''
        self.n_components=n_components
        self.init_std=init_std
        
        '''
        Hyperparameters
        ---------
        lambda_theta, lambda_beta: float
            Regularization parameter for user (lambda_theta) and item factors (
            lambda_beta). Default value 1e-5. Since in implicit feedback all
            the n_users-by-n_items data points are used for training,
            overfitting is almost never an issue
        lambda_y: float
            inverse variance on the observational model. Default value 1.0
        init_mu: float
            All the \mu_{i} will be initialized as init_mu. Default value is
            0.01. This number should change according to the sparsity of the
            data (sparser data with smaller init_mu). In the experiment, we
            select the value from validation set
        '''
        self.lam_y=lam_y
        self.lam_theta=lam_theta
        self.lam_beta=lam_beta
        self.init_mu=init_mu

    def _init_params(self, n_users, n_items):
        self.theta = self.init_std * \
            np.random.randn(n_users, self.n_components).astype(floatX)
        self.beta = self.init_std * \
            np.random.randn(n_items, self.n_components).astype(floatX)
        self.mu = self.init_mu * np.ones(n_items, dtype=floatX)

        self.A = np.random.randn(n_users, n_items).astype(floatX)
    
    '''
        X is the binarized train data 
        Raw is the confidence matrix
    '''
    def fit(self, X ,vad_data=None,MAX_ITERATION=15):
        n_users, n_items = X.shape
        self._init_params(n_users, n_items)
        self.show_params()
        
        
        RECORD_VAD=np.zeros((MAX_ITERATION,3))
        
        
        for i in range(MAX_ITERATION):
            print "ROUND %d" %i
            self._update_A(X)
            self._update_factors(X)
            self._update_perItem_U(X)
            RECORD_VAD[i]=self._validate(X,vad_data)
            
        return RECORD_VAD   
            
    def fit_with_confidence(self, X, Raw=None, vad_data=None, MAX_ITERATION=15):           
        n_users, n_items = X.shape
        self._init_params(n_users, n_items)
        self.show_params()
        
        
        RECORD_VAD=np.zeros((MAX_ITERATION,3))
        
        
        for i in range(MAX_ITERATION):
            print "ROUND %d" %i
            self._update_A(X)
            self._update_factors(X)
            self._update_confidence_perIterm_U(X,Raw)
            RECORD_VAD[i]=self._validate(X,vad_data)
            
        return RECORD_VAD 
        
        

    def show_params(self):
        print "model.theta.shape",self.theta.shape
        print "model.beta.shape",self.beta.shape
        print "model.mu.shape",self.mu.shape
        print "model.A.shape",self.A.shape

    def _update_A(self, X):
        '''E-step, update the expected exposure'''
        
        n_users,n_items=X.shape
        
        print "---updating A---"

        for u in range(n_users):
            for i in range(n_items):
                if X[u,i] == 1:
                    self.A[u,i] = 1;

                else:
                    norm_zero_giveSituation = sqrt(self.lam_y / 2 * np.pi) * \
                                           np.exp(-self.lam_y * self.theta[u].dot(self.beta[i])**2 / 2)
                    self.A[u,i] = self.mu[i] * norm_zero_giveSituation / \
                                (self.mu[i] * norm_zero_giveSituation + 1 -self.mu[i])
        
    def _update_factors(self,X):
        '''M-step,update the theta and beta factors '''
        n_users,n_items=X.shape
        
        print "---updating factors!---"
        
        old_theta=self.theta
        
        '''update theta'''
        for u in range(n_users):
            
            #print "---updating theta[%d]---" %u
            pBeta=np.diag(self.A[u]).dot(self.beta)
            B=self.beta.T.dot(pBeta)*self.lam_y+self.lam_theta*np.eye(self.n_components)
            
            a=np.zeros(self.n_components)
            for i in range(n_items):
                a+=self.lam_y*self.A[u,i]*X[u,i]*self.beta[i]
            
            #print "a.shape",a.shape
            #print "B.shape",B.shape
            #print "pBeta.shape",pBeta.shape
            #print "model.beta.T.shape",model.beta.T.shape
            
            self.theta[u]=np.linalg.solve(B,a)
        
        '''update beta'''
        
        for i in range(n_items):
            #print "---updating beta[%d]---" %i
            pTheta=np.diag(self.A[:,i]).dot(old_theta)  
            B=old_theta.T.dot(pTheta)*self.lam_y+self.lam_beta*np.eye(self.n_components)   
            
            a=np.zeros(self.n_components)
            for u in range(n_users):
                a+=self.lam_y*self.A[u,i]*X[u,i]*old_theta[u]            
            self.beta[i]=np.linalg.solve(B,a)
            
    def _update_perItem_U(self,X):
        print "---updating perItem_U!---"
        n_users,n_items=X.shape
        for i in range(n_items):
            self.mu[i]=sum(self.A[:,i])/n_users
                   
    def _update_confidence_perIterm_U(self,X,Raw):
        print "---updating confidence_perItem_U!---"
        n_users,n_items=X.shape
        for i in range(n_items):
            sum_Expo_Confidence=0
            sum_Confidence=0
            for u in range(n_users):
                sum_Expo_Confidence+=self.A[u,i]*self.log_confidence(Raw[u,i])
                sum_Confidence+=self.linear_confidence(Raw[u,i])
            self.mu[i]=sum_Expo_Confidence/sum_Confidence

                
        
    def _validate(self, X, vad_data):
        '''Compute validation metric '''
        
        vad_recall_at_k = evaluation.Recall_at_k(X, vad_data,
                                                self.theta,
                                                self.beta,
                                                k=20)
        
        vad_map_at_k = evaluation.MAP_at_k(X, vad_data,
                                                self.theta,
                                                self.beta,
                                                k=20)
        
        vad_ndcg_at_k = evaluation.NDCG_at_k(X, vad_data,
                                                self.theta,
                                                self.beta,
                                                k=20)
        
        if True:
            print('\tValidation NDCG@k: %.5f' % vad_ndcg_at_k)
            print('\tValidation Recall@k: %.5f' % vad_recall_at_k)
            print('\tValidation MAP@k: %.5f' % vad_map_at_k)
            sys.stdout.flush()
            
        
        return vad_ndcg_at_k,vad_recall_at_k,vad_map_at_k
    
    '''
      Utility function:
          Raw should be a sparse matrix [scipy.sparse.csr_matrix]
    '''
    
    
    def binarize(self,Raw):
        bi=Raw
        bi.data=np.ones_like(bi.data)
        return bi
    
    '''c = weight * r + 1'''
    def linear_confidence(self,raw_data_entry, weight=5):
        
        return raw_data_entry*weight+1
    
    '''c = 1 + weight * log ( 1 + r/b )'''
    def log_confidence(self,raw_data_entry,weight=10,b=10):
        return weight*np.log(1+raw_data_entry/b)+1
    
    
        
            