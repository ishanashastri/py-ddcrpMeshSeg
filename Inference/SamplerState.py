import os
import numpy as np
import time 
import pickle as pickle
from random import sample
from scipy.sparse import csr_matrix
from Util.fast_cc import fast_cc
from numpy.random import *
np.random.seed(2)

class SamplerState:
    
    def __init__(self, data, num_iter):
        # number of mesh faces (data instances)
        self.num_data=int(len(data["X"]))
        self.c = np.full((num_iter, self.num_data), np.nan)
        self.t = np.full((num_iter, self.num_data), np.nan)
        self.T = np.full((1, num_iter), np.nan)
        self.c[0, :] = np.arange(self.num_data, dtype=np.int64) # initialize each meshface as its own part.
        self.t[0, :] = np.arange(self.num_data, dtype=np.int64)
        self.T[0,0]=self.num_data

    def run_sampler(self,prior_param,mniw,data,bookkeeper,num_iter):
        savedir = bookkeeper.savedir

        for iter in range(1, num_iter):
            self.c[iter,:]=self.c[iter-1,:]
            self.t[iter,:]=self.t[iter-1,:]
            self.T[0,iter]=self.T[0,iter-1]
            
            t0 = time.clock()
            for i_s in range(0,self.num_data):
                [self,bookkeeper] = self.sample_links(iter,i_s,bookkeeper, prior_param,mniw,data)
            t1 = time.clock()
            
            print(' ')
            print('\nIteration - %d took %0.03f seconds to complete\n' % (iter, (t1-t0)))
            if(iter%20==0):
                pickle.dump(self, open(os.path.join(savedir,('Intermediate_MCMC_sample%d.pkl'%iter)),"wb"))
                print('Iteration - %d took %0.03f seconds to complete\n' % (iter,(t1-t0)))
        return self

    def sample_links(self,iter,i_s,bookkeeper,prior_param,mniw,data):
        # # #  Sample p(c_i|c_ni,X,Y)

        runCC = self.check_shallow_cycles(self.c[iter,:],i_s) 

        temp_c=self.c[iter,:] 
        #reset the current link.
        temp_c[i_s] = i_s 
        self.c[iter,i_s] = i_s 

        if(runCC):
            # should only have to compute c_to_t for the current table.
            table = self.t[iter,i_s] 
            table_members = np.nonzero(self.t[iter,:]==table)[0]
            inverted_index = csr_matrix((np.arange(0,len(table_members)), (np.zeros(len(table_members),dtype=np.int64),table_members)))
            
            #temp_c(table_members) are the customers tab_members are pointing to.
            #remapped_table_members contains these customers mapped to 1:table_members
            remapped_table_members = inverted_index[0,temp_c[table_members]].sorted_indices()

            #Perform a connected components operation and determine if resetting the
            #current link splits the corresponding table.
            [tables,split] = fast_cc(remapped_table_members.shape[1],remapped_table_members.toarray()) 
        else:
            #shallow cycle found
            split = 1 
        
        if(split>2):
            print("Err - Cant split table more than two ways") 

        elif(split==2):
            split_customers = table_members[np.nonzero(tables==2)[0]]
            self.t[iter,split_customers] = max(self.t[iter,:])+1 
            num_tables = int(self.T[0,iter])+1
            self.T[0,iter] = num_tables 
            bookkeeper.valid_clusters[num_tables-1] = num_tables-1
            #reset the likelihood of the split tables
            if (bookkeeper.table_lik[bookkeeper.valid_clusters[int(self.T[0,iter])]]):       
                bookkeeper.table_lik[bookkeeper.valid_clusters[int(self.T[0,iter])]] = 0  
            bookkeeper.table_lik[int(table)] = 0 
            
            # reset pairwise table_lik    
            bookkeeper.pairwise_table_lik[bookkeeper.valid_clusters[int(self.T[0,iter])],:] = bookkeeper.reset_vec 
            bookkeeper.pairwise_table_lik[bookkeeper.valid_clusters[int(table)],:] = bookkeeper.reset_vec 
            bookkeeper.pairwise_table_lik[:,bookkeeper.valid_clusters[int(self.T[0,iter])]] = np.squeeze(bookkeeper.reset_vec.transpose())
            bookkeeper.pairwise_table_lik[:,bookkeeper.valid_clusters[int(table)]] = np.squeeze(bookkeeper.reset_vec.transpose())
        
        # sample new link for the current mesh face 
        nbor = np.nonzero(prior_param.d2_mat[:,i_s])[0]
        prior = []
        for n in nbor:
            prior.append(prior_param.d2_mat[n,i_s])
        logposterior = np.log(prior) 

        t_curr = self.t[iter,i_s] 
        t_curr_members = self.t[iter,:]==t_curr #returns 1 if sampler_state[iter,:] member ==t_curr, else 0
        for ii in range(0,len(nbor)):
            t_prop=self.t[iter,nbor[ii]] 
            t_prop_members = self.t[iter,:]==t_prop
            [loglik_ratio,bookkeeper]=self.compute_likelihood_ratio(mniw,data,bookkeeper,t_prop,t_curr,t_prop_members,t_curr_members) 
            logposterior[ii] = loglik_ratio+logposterior[ii] 

        logposterior=logposterior-max(logposterior) 
        logposterior=np.exp(logposterior) 
        logposterior=logposterior/np.sum(logposterior) 

        c_ji_new = np.random.choice(nbor, size = 1, p = logposterior)[0]
        #c_ji_new = nbor[np.nonzero(np.random.multinomial(1,logposterior))][0]
        #c_ji_new = nbor[int(len(nbor)/2)] #for debugging
        
        #update customer links
        self.c[iter,i_s] = c_ji_new 

        #update tables
        link_table = self.t[iter,c_ji_new]
        curr_tab = self.t[iter,i_s]  #note this won't be the same as tab, if a split occurs.
        if(link_table!=curr_tab): # do nothing if it is equal, linking to a customer in the current table
            large_idx = int(max(link_table,curr_tab))
            small_idx = int(min(link_table,curr_tab))
            ids = self.t[iter,:]==large_idx #must return boolean array 1x157866 
            self.t[iter,ids] = small_idx 
        
            ids = self.t[iter,:]>large_idx 
            self.t[iter,ids] = self.t[iter,ids] - 1 
            bookkeeper.valid_clusters[np.where(bookkeeper.valid_clusters>large_idx)[0]] = bookkeeper.valid_clusters[np.where(bookkeeper.valid_clusters>large_idx)[0]] - 1 
            
            # Modify table_lik to reflect the merge.
            # reset the likelihood of the merged table.
            loc = bookkeeper.valid_clusters[int(small_idx)]
            bookkeeper.table_lik[loc] = 0 

            bookkeeper.valid_clusters[large_idx:int(self.T[0,iter])-1] = bookkeeper.valid_clusters[large_idx+1:int(self.T[0,iter])]
            #print(bookkeeper.valid_clusters[large_idx+1:int(self.T[0,iter])])  
            bookkeeper.pairwise_table_lik[:,loc] = np.ravel(bookkeeper.reset_vec.transpose())
            bookkeeper.pairwise_table_lik[loc,:] = np.ravel(bookkeeper.reset_vec)
            
            self.T[0,iter] = self.T[0,iter]-1 

        if (self.T[0,iter]==0):
            print('something is wrong') 
       
        return (self,bookkeeper)

    def check_shallow_cycles(self, curr_c,i_s):

         # check for shallow cycles. Presence of a cycle ensures that removing the
         # current link won't cause a split. Saves us the expensive connected
         # components operation.
               
        curr_c = curr_c.astype(int)
        runCC = 1 
        if(curr_c[i_s]==i_s):
             # if the existing link is a self-link, removing it can't split a
             # component.
            runCC = 0 
        else:
             # # # check shallow cycles
             # A->B  B->A, removing either link can't split the component.
            if(curr_c[curr_c[i_s]]==i_s):
                runCC = 0 
                
                 # A->B->C  C->A, removing any of these links can't split the component.
            elif(curr_c[curr_c[curr_c[i_s]]]==i_s):
                runCC = 0 
                
                 # A->B->C->D  D->A, removing any of these links can't split the component.
            elif(curr_c[curr_c[curr_c[curr_c[i_s]]]]==i_s):
                runCC = 0 
                
                 # A->B->C->D->E  E->A, removing any of these links can't split the component.
            elif(curr_c[curr_c[curr_c[curr_c[curr_c[i_s]]]]]==i_s):
                runCC = 0 
                
                 # 6 cycle, removing any of these links can't split the component.
            elif(curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[i_s]]]]]]==i_s):
                runCC = 0 
                
                 # 7 cycle
            elif(curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[i_s]]]]]]]==i_s):
                runCC = 0 
                
                 # 8 cycle
            elif(curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[i_s]]]]]]]]==i_s):
                runCC = 0 
                
                 # 9 cycle
            elif(curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[i_s]]]]]]]]]==i_s):
                runCC = 0 
                
                 # 10 cycle
            elif(curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[curr_c[i_s]]]]]]]]]]==i_s):
                runCC = 0 
       
        return runCC


    def compute_likelihood_ratio(self, mniw,data,bookkeeper,t_prop,t_curr,t_prop_members,t_curr_members):
        #t_curr is the table that super-pixel i currently belongs to, t_prop is the
        #table being proposed for i.

        if(t_prop==t_curr):
            #the new link doesn't merge tables
            #li (log-likelihood) will be zero. and we will sample only on the basis of the prior
            li = 0
        else:
            t_prop = int(t_prop)
            t_curr = int(t_curr)
            # if we don't already have table likelihoods compute and return them along with their counts.
            #bookkeeper.valid_clusters = np.reshape(bookkeeper.valid_clusters,(-1,1))
            if(not bookkeeper.table_lik[bookkeeper.valid_clusters[t_prop]]):
                bookkeeper.table_lik[bookkeeper.valid_clusters[t_prop]] = self.compute_table_lik(mniw,data,t_prop_members)
            if(not bookkeeper.table_lik[bookkeeper.valid_clusters[t_curr]]):
                bookkeeper.table_lik[bookkeeper.valid_clusters[t_curr]] = self.compute_table_lik(mniw,data,t_curr_members)
            min_t = np.minimum(bookkeeper.valid_clusters[t_prop],bookkeeper.valid_clusters[t_curr])
            max_t = np.maximum(bookkeeper.valid_clusters[t_prop],bookkeeper.valid_clusters[t_curr])
            if(bookkeeper.pairwise_table_lik[min_t,max_t]):
                merged_lik = bookkeeper.pairwise_table_lik[min_t,max_t]
            else:
                #merged_lik not on record. recompute.
                #note that for merged_lik computation the table_counts have already been computed.

                merged_lik_bodies = np.empty((1,data["num_bodies"]))
                for i in range (0,data["num_bodies"]):
                    #compute likelihoods across different body shapes, each
                    #having its own reference pose.
                    merged_lik_bodies[0,i] = mniw.computeMarginalLik(np.concatenate((data["Y"][:,t_prop_members,:] , data["Y"][:,t_curr_members,:]),axis=1), np.concatenate((data["X"][t_prop_members,:].transpose(),data["X"][t_curr_members,:].transpose()),axis=1))
                #compute part likelihood by summing over all bodies.
                merged_lik = np.sum(merged_lik_bodies)
                bookkeeper.pairwise_table_lik[min_t,max_t] = merged_lik
            li = merged_lik-bookkeeper.table_lik[bookkeeper.valid_clusters[t_prop]] -bookkeeper.table_lik[bookkeeper.valid_clusters[t_curr]] # p(t_old U t_new)/p(t_old)*p(t_new)

        return (li, bookkeeper)

    def compute_table_lik(self, mniw,data,table_members):
        #computes table marginal likelihood.

        table_lik_bodies = np.zeros((1,data["num_bodies"]))
        for i in range(0,data["num_bodies"]): # each body refers to a different person
            #different body shape and hence different reference body.
            #X = data["X"][table_members,:].transpose()
            #Y = data["Y"][:,table_members,:]

            table_lik_bodies[0,i] = mniw.computeMarginalLik(data["Y"][:,table_members,:],data["X"][table_members,:].transpose())
            #[table_lik_bodies(i),~] = computeMultiPoseMNIWlik(Y,X,param);


            #part likelihood is computed by summing over all bodies.
            table_lik = np.sum(table_lik_bodies)

        return table_lik