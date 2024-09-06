# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:46:21 2022

@author: maktas
"""


# In this code, we first rank vertices based on Laplacian centrality, then divide the vertices into 10 equal parts based on their ranking.
# In rach part, we randomly assign honest, pro and anti equally.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

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
    for n in nbors:
        public.append(w_ji(i,n,opinion,tau,h))
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

def laplacian_centrality(L):
    # L is the laplacian
    LC=np.full((len(L),), 0,dtype=float)
    ll=laplacian_energy(L)
    for i in range(len(L)):
        Lcopy=np.copy(L)
        #print(i)
        z=np.array(L[i])
        edge=np.argwhere(abs(z)>0)
        for j in edge:
            Lcopy[j[0]][j[0]]=Lcopy[j[0]][j[0]]-1
        L_del=np.delete(Lcopy,i,0)
        L_del=np.delete(L_del,i,1)
        le=laplacian_energy(L_del)
        LC[i]=(ll-le)/ll
    return LC
      

def laplacian_energy(L):
    evalues,evectors=np.linalg.eigh(L)
    #print(evalues)
    le=0
    for i in evalues:
        le = le + i*i
    return le

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


dataset='facebook'
adjname='Datasets/'+dataset+'_adj.csv'
tlist=np.linspace(0,1,11)

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

#nn=200
#G=nx.erdos_renyi_graph(nn,0.05)
#G=nx.wheel_graph(nn)
#G=nx.cycle_graph(nn)
#G=nx.turan_graph(nn,3)
#G=nx.path_graph(nn)
#G=nx.star_graph(101)
#G=nx.circulant_graph(nn,[5])
#G=nx.dorogovtsev_goltsev_mendes_graph(nn)

# Laplacian centrality
L=np.array(nx.laplacian_matrix(G).todense())
rank2=laplacian_centrality(L)

rr=np.full((nn,), 0,dtype=float)
for i in range(nn):
    rr[i]=rank2[i]
rankd=np.argsort(-1*rr)




ngroup=10

parts=np.array_split(rankd, ngroup)

num=10

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
        
        lap_centrality=laplacian_centrality(L)
        honest=[]
        pro=[]
        anti=[]
        n_honest=0
        n_pro=0
        n_anti=0
        for i in range(nn):
            if nbb[i]==0:
                honest.append(lap_centrality[i])
                n_honest=n_honest+1
            elif nbb[i]==1:
                pro.append(lap_centrality[i])
                n_pro=n_pro+1
            elif nbb[i]==2:
                anti.append(lap_centrality[i])
                n_anti=n_anti+1
        honest_all.append(sum(honest)/n_honest)
        pro_all.append(sum(pro)/n_pro)
        anti_all.append(sum(anti)/n_anti)
    honest_tau.append(sum(honest_all)/num)
    pro_tau.append(sum(pro_all)/num)
    anti_tau.append(sum(anti_all)/num)


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


filename=dataset+'_Lap_energy.csv'
#filename='ER_Lap_200_005.csv'
with open(filename, 'w',newline='') as myFile:
    writer = csv.writer(myFile)
    writer.writerows(topr)

#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

# Put a legend below current axis
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=3)   
    