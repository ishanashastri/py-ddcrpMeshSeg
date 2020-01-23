import math

import numpy as np


class MNIWlik: #Matrix Normal Inverse Wishart Likelihood class

    I = np.identity(4)

    def __init__(self, scale, S0, n0, M, K, rofg):    
        self.scale = scale #scale
        self.S0 = S0 #expected variance
        self.n0 = n0 #dof
        self.M = M #mean transformation
        self.K = K #precision matrix. higher = smaller var        
            
         # #---Precomputations useful for Marginal likelihood computation--- # #
         #Precomputed Ratio of Gammas (\Gamma_3((N_k+n0)/2)/\Gamma_3(n0/2))
        self.rofg = rofg 

        # precompute values needed for Marginal likelihood computations
        L1 = np.linalg.cholesky(self.S0)
        logdet_S0 = 2*sum(np.log(np.diag(L1))) 
        L1 = np.linalg.cholesky(self.K) 
        logdet_K = 2*np.sum(np.log(np.diag(L1))) 
        self.MKMt = M.dot(self.K)
        self.MKMt = self.MKMt.dot(M.transpose())
        self.MK = M.dot(self.K)
        self.sumKS0 = (3/2)*logdet_K + (n0/2)*logdet_S0 

    def computeMarginalLik(self, Y_p, X):
        # I/P - hyper-params and data
        # O/P - log marginal likelihood
        
        #X \in R^4*N, Y_p \in R^3*N*num_poses
           
        num_poses = len(Y_p[0][0]) 
        
        N = len(Y_p[0])
        
        # Sxx = X*X'+ I 
        Sxx = np.dot(X,X.transpose()) + self.K 
        L = np.linalg.cholesky(Sxx) 
        invL = np.linalg.solve(L,self.I) 
        T = np.dot(invL, X) 
        b = np.dot(invL,self.MK.transpose()) 
        B = self.MKMt - np.dot(b.transpose(),b)
        
        #compute pose independent terms of mlik 
        
        C1 = self.sumKS0-(3/2)* 2*sum(np.log(np.diag(L))) 
        #ratio  #of 3 dimensional gamma functions
        C2 = self.rofg[0][N-1] #-1 for zero indexing
        C = C1+C2 
        #compute pose dependent terms of the mlik.
        poselik = np.zeros((1,num_poses))
        
        for p in range(0,num_poses):
            t = np.dot(Y_p[:,:,p],T.transpose())
            btcrossterm = np.dot(t,b) 
            Syx = np.dot(Y_p[:,:,p],Y_p[:,:,p].transpose()) + B -np.dot(t,t.transpose()) -btcrossterm.transpose() - btcrossterm   # This is what needs to be computed for each pose Syx = Y*Y' + MKMt -t*t' -b'*b - (t*b)'-(t*b) 
            #Syx = Y_p(:,:,p)*Y_p(:,:,p)' + B -t*t'   # This is what needs to be computed for each pose Syx = Y*Y' + MKMt -t*t' -b'*b 
            
            L1 = np.linalg.cholesky(self.S0+Syx) 
            logdet_S0Syx = 2*np.sum(np.log(np.diag(L1))) 
            
            poselik[0,p] = C-((N+self.n0)/2)*logdet_S0Syx 

        llik = np.sum(poselik)  
        llik = llik - 1.5*N*num_poses*np.log(math.pi) 

        return llik
