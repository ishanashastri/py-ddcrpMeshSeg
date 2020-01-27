import numpy as np
from scipy.sparse import csr_matrix
from graph_tool.all import *

def fast_cc(N, assig): 
#computes table assignments from inputted links
#inputs: assig is pointer vector (ndarray), N is number of nodes
#output: tables is the membership for the clusters, largest is the total number of clusters

    edges = []
    
    for p in range(0, len(assig[0])):
        edges.append((p, assig[0][p]))
        if(p != assig[0][p]):
            edges.append((assig[0][p], p)) #make symmetric    
    gG = Graph()
    gG.add_edge_list(edges)
    tables  = label_components(gG,directed=False)[0].a+1
    largest = max(tables)

    return (tables, largest)