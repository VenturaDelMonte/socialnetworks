#!/usr/bin/python

from lesson4 import betweenness

# Computes the eigenvector centrality of the graph in input
def eigenvector(graph,confidence):
  nodes = graph.keys()
  done = 0
  
  eigen = dict()
  for i in nodes:
    eigen[i] = 1/len(nodes)
  
  tmp = dict() # I first compute this temporary value for eigenvector centrality. The real value consists of a normalization of this temporary value
  while not done: # I repeat the process until the centrality vector does not change anymore.
    max_tmp = 0
    for i in nodes:
      tmp[i] = eigen[i] # Even if this is not standard, this is necessary in order that the algorithm converge for any graph
      for j in graph[i]:
        tmp[i] += eigen[j]
      if tmp[i] > max_tmp:
        max_tmp = tmp[i]
        
    diff = 0
    for i in nodes:
      diff += abs(eigen[i]-float(tmp[i])/max_tmp) # Distance between old and new centrality vector
      eigen[i] = float(tmp[i])/max_tmp
    
    if diff < confidence:
      done = 1
  return eigen

# Returns the k most central nodes in the network
def top(graph,k,centrality,confidence):
  
  if centrality == "b":
    b = betweenness(graph)
  elif centrality == "e":
    b = eigenvector(graph,float(confidence))
    
  top = []
  top_values = []
  for i in b.keys():
    added = 0
    for j in range(min(len(top),int(k))):
      if b[top[j]] < b[i]:
        top.insert(j,i)
        added = 1
        break
    if added == 0:
      top.append(i)
    if len(top) > int(k):
      top.pop()
  for i in range(len(top)):
    top_values.append(b[top[i]])
  return top, top_values