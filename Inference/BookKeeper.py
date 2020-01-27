import numpy as np
from scipy.sparse import csr_matrix


class BookKeeper:

    def __init__(self, sampler_state, savedir):
        self.valid_clusters = np.arange(sampler_state.num_data)
        #self.table_lik = csr_matrix((1,sampler_state.num_data))
        #self.pairwise_table_lik = csr_matrix((sampler_state.num_data,sampler_state.num_data))
        #self.reset_vec = csr_matrix((1,self.pairwise_table_lik.shape[0]))
        
        self.table_lik = np.zeros((1,sampler_state.num_data))[0]
        self.pairwise_table_lik = np.zeros((sampler_state.num_data,sampler_state.num_data))
        self.reset_vec = np.zeros((1,len(self.pairwise_table_lik[0])))
        self.savedir = savedir

