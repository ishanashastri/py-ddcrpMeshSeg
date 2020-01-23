import numpy as np
import math

def trace_plot(iter,mniw,data,sampler_state,D):

    ln_map_e_store = np.zeros((1,int(sampler_state.T[0,iter])))
    p = np.zeros((sampler_state.num_data,1))
    ctr = 0
    for t in np.unique(sampler_state.t[iter,:]):
        tab_members = sampler_state.t[iter,:]==t
        ln_map_e_store[0, ctr] = sampler_state.compute_table_lik(mniw,data,tab_members)
        ctr+=1

    nc = np.sum(D,axis=1)
    for i in range(0,sampler_state.num_data):
        p[i] = D[i,int(sampler_state.c[iter,i])]
    
    prior = np.log(np.divide(np.squeeze(p),nc)) 
    # for i = 1:length(param.c)
    #     c = param.c(i)
    #     nc = nnz(D(i,:))
    #     prior(i) = log(D(i,c)/nc)
    # 
    lp = np.sum(prior)

    ll = np.sum(ln_map_e_store)

    ln_map_e = ll + lp #- 1.5*sampler_state.num_data*log(pi)

    return (ln_map_e,ll,lp)