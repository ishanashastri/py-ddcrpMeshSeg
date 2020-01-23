import numpy as np
from igraph import *

def fast_cc(N, assig): 
#computes table assignments from inputted links
#inputs: assig is pointer vector (ndarray), N is number of nodes
#output: tables is the membership for the clusters, largest is the total number of clusters

    edges = []
    
    for p in range(0, len(assig[0])):
        edges.append((p, assig[0][p]))

    adj = np.zeros((N,N)) #create n x n sparse matrix 
    for i in range(0, N): #add ones to create adjacency matrix
        adj[edges[i][0],edges[i][1]]=1

    adj = np.maximum(adj, adj.transpose()) #make symmetric (undirected)        

    iG = Graph.Adjacency(adj.tolist())
    tables  = np.asarray(iG.components(mode=WEAK).membership)+1
    tables = np.reshape(tables, (-1,1))
    largest = max(tables)

    return (tables, largest)