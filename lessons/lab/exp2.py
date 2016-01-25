#!/usr/bin/python

import sys
import math
from lesson1 import randomGraph, diameter, averageClustering
from lesson2 import *

eps = 1e-12

rep = int(sys.argv[1]) #number of repetitions
nodes = int(sys.argv[2]) #number of nodes
sep = float(sys.argv[3]) #experiments for p = sep, 2sep, 3sep, ..., 1
k_threshold=int(sys.argv[4]) #the maximum number of random links

# experiments with random graphs
print("experiments w/ random graphs")
p = sep
while p < 1:
  diam = 0
  aclust = 0
  not_connected = 0
  for i in range(rep): # for each p we make rep simulations
    graph = randomGraph(nodes,p)
    edges=countEdges(graph)
    tmp_diam=diameter(graph)
    if tmp_diam > 0: #if diameter is -1, then the graph is not connected
      diam += tmp_diam
    else:
      not_connected += 1
    aclust += averageClustering(graph)
  print(p, edges, not_connected, diam/(rep-not_connected), aclust/rep)
  p += sep

print("experiments w/ Watts-Strogatz graphs (on grid)")
# experiments with Watts-Strogatz graphs (on grid)
line=int(math.sqrt(nodes))
for r in range(1,line):
  for k in range(1,k_threshold+1):
    diam = 0
    aclust = 0
    not_connected = 0
    for i in range(rep):
      graph = WSGridGraph(nodes,r,k)
      edges=countEdges(graph)
      tmp_diam=diameter(graph)
      if tmp_diam > 0:
        diam += tmp_diam
      else:
        not_connected += 1
      aclust += averageClustering(graph)
    print(r, k, edges, not_connected, diam/(rep-not_connected+eps), aclust/rep)

print("experiments w/ Watts-Strogatz graphs (on 2D grid)")
# experiments with Watts-Strogatz graphs (on 2D plane)
for r in range(1,line):
  for k in range(1,k_threshold+1):
    diam = 0
    aclust = 0
    not_connected = 0
    for i in range(rep):
      graph = WS2DGraph(nodes,r,k)
      edges=countEdges(graph)
      tmp_diam=diameter(graph)
      if tmp_diam > 0:
        diam += tmp_diam
      else:
        not_connected += 1
      aclust += averageClustering(graph)
    print(r, k, edges, not_connected, diam/(rep-not_connected+eps), aclust/rep)