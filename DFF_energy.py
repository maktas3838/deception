
"""
Created on Wed Dec  1 11:04:35 2021

@author: maktas
"""

# In this code, we first rank vertices based on DFF centrality, then divide the vertices into 10 equal parts based on their ranking.
# In each part, we randomly assign honest, pro and anti equally.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import math

def w_ji(i,j,opinion,tau,h):
    if h==0: # honest
        w=opinion[i]
    elif h==1: # prosocial
        w=tau*opinion[i]+(1-tau)*opinion[j]
    elif h==2: # antisocial
        w=tau*opinion[i]-(1-tau)*opinion[j]
    return w

def y_i(i,opinion,tau,G,h):
    nbors=G.neighbors(i)
    public=[]
    iind=0
    for n in nbors:
        public.append(w_ji(i,n,opinion,tau,h))
        iind=iind+1
    if len(public)==0:
        y=0
    else:
        y=sum(public)/len(public)
    return y
    

def info_dif(i,j,opinion,tau,perception,h):
    if h==0: # honest
        w=1
    elif h==1: # prosocial
        w=tau+(1-tau)*(perception[j]/opinion[i])
    elif h==2: # antisocial
        w=tau-(1-tau)*(perception[j]/opinion[i])
    return w

def dff_centrality(Laplacian, prob, t):
    """ Laplacian   = Laplacian for all simplices 
        prob        = the probability distribution over all simplices
        t           = the scale parameter value of the diffusion Frechet function"""
    # compute the down and up Laplacian
    evalues,evectors=np.linalg.eigh(Laplacian)
    #normalizing eigenfunctions 
    for i in range(1,evalues.shape[0]):
        evectors[:,i]=evectors[:,i]*math.pow(np.dot(evectors[:,i],evectors[:,i]),-0.5)

    size=evalues.shape[0]
    dist_mat=np.zeros([size,size])
    for i in range(size):
        for j in range(i+1,size):
            dist_mat[i,j]=diffusion_distance(i,j,evalues,evectors,t)
            dist_mat[j,i]=dist_mat[i,j]
    return np.dot(dist_mat,prob)

def diffusion_distance(ind1,ind2,evalues,evectors,t):
    """calculates the difussion distances (squared) between 2 nodes on the graph with the discrete laplacian"""
    aux1=(evectors[ind1,:]-evectors[ind2,:])**2
    aux2=np.exp(-2*t*evalues)
    return np.dot(aux1,aux2)

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


dataset='Netscience'
adjname='Datasets/'+dataset+'_adj.csv'
tlist=np.linspace(0,1,41)

adj_mtx=[]
with open(adjname) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        adj_mtx.append(row)

nn=len(adj_mtx)

adjmatrix = np.full((nn, nn), 0)

for i in range(nn):
    for j in range(i+1,nn):
        if float(adj_mtx[i][j])!=0:
            adjmatrix[i][j]=1#int(adj_mtx[i][j]) # making unweighted
            adjmatrix[j][i]=1#int(adj_mtx[j][i])

G = nx.Graph(adjmatrix)

nn=200
G=nx.erdos_renyi_graph(nn,0.05)
#G=nx.wheel_graph(nn)
#G=nx.cycle_graph(nn)
#G=nx.turan_graph(nn,3)
#G=nx.path_graph(nn)
#G=nx.star_graph(101)
#G=nx.circulant_graph(nn,[5])
#G=nx.dorogovtsev_goltsev_mendes_graph(nn)

# Laplacian centrality

t=0.01
prob=[1/nn for i in range(nn)]  # uniform probability distribution over simplices 

L=np.array(nx.laplacian_matrix(G).todense())
rank2=dff_centrality(L,prob,t)

rr=np.full((nn,), 0, dtype=float)
for i in range(nn):
    rr[i]=rank2[i]
rankd=np.argsort(rr)


ngroup=10

parts=np.array_split(rankd, ngroup)

num=100

honest_tau=[]
pro_tau=[]
anti_tau=[]



nlist=list(range(nn))


for tau in tlist:
    print(tau)
    honest_all=[]
    pro_all=[]
    anti_all=[]

    for ii in range(num):
        nbb=np.full((nn,), 0)
        
        for i in range(ngroup):
            pp=parts[i]
            random.shuffle(pp)
            rrr=[0,1,2]
            random.shuffle(rrr)
            lp=np.array_split(pp, 3)
            for j in range(3):
                for k in lp[j]:
                    nbb[k]=rrr[j]
        
        edges=G.edges()
        incidence=np.full((G.number_of_edges(),nn), 0,dtype=float)
        info=np.full((nn,nn), 0,dtype=float)
        opinion=[]
        for i in range(nn):
            n=2*random.random()-1
            opinion.append(n)
        ind=0
        
        perception=[]
        for i in range(nn):
            perception.append(y_i(i,opinion,tau,G,nbb[i]))
        
        for e in edges:
            i=e[0]
            j=e[1]
            wi=info_dif(i, j, opinion, tau, perception, nbb[i])
            wj=info_dif(j, i, opinion, tau, perception, nbb[j])
            incidence[ind][i]=wi
            incidence[ind][j]=-wj
            ind=ind+1
        
        L=np.matmul(np.transpose(incidence),incidence)
        
        dff_cent=dff_centrality(L,prob,t)
        honest=[]
        pro=[]
        anti=[]
        n_honest=0
        n_pro=0
        n_anti=0
        for i in range(nn):
            if nbb[i]==0:
                honest.append(dff_cent[i])
                n_honest=n_honest+1
            elif nbb[i]==1:
                pro.append(dff_cent[i])
                n_pro=n_pro+1
            elif nbb[i]==2:
                anti.append(dff_cent[i])
                n_anti=n_anti+1
        honest_all.append(sum(honest)/n_honest)
        pro_all.append(sum(pro)/n_pro)
        anti_all.append(sum(anti)/n_anti)
    honest_tau.append((sum(honest_all)/num)/nn)
    pro_tau.append((sum(pro_all)/num)/nn)
    anti_tau.append((sum(anti_all)/num)/nn)


topr=[honest_tau,pro_tau,anti_tau]

markers=["o","v",".","^","*","s","p","+","x","D","8","P","1"]



totalm=3
fig = plt.figure()
plt.subplot(111)
ax=plt.subplot(111)
#cm = plt.get_cmap('jet')
#ax.set_color_cycle([cm(1.*i/totalm) for i in range(totalm)])
label_str=["Honest","Pro","Anti"]
for j in range(totalm):
    ax=plt.plot(tlist,topr[j][:],label=label_str[j],marker=markers[j])
plt.title(dataset)
plt.legend(bbox_to_anchor=(0.8, -0.1), ncol=3)

'''
filename=dataset+'_energy_DFF_40.csv'
with open(filename, 'w',newline='') as myFile:
    writer = csv.writer(myFile)
    writer.writerows(topr)
'''
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

# Put a legend below current axis
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=3)   
