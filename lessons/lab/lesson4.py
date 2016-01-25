#!/usr/bin/python

def betweenness(graph, verbose=False):
  cb = {} #betweenness centrality

  for i in graph.keys():
    cb[i] = 0

  for i in graph.keys(): #check the max distance for every node i
    visited = [] #keep a list of visited nodes to check if the graph is connected

    P = {}
    for j in graph.keys():
      P[j] = [] # j's parents list

    sigma = {}
    for j in graph.keys():
      sigma[j] = 0
    sigma[i] = 1

    ###BFS###
    queue = [i]
    distance = {}
    for j in graph.keys():
      distance[j] = -1

    distance[i] = 0
    while queue != []:
      s = queue.pop(0)
      visited.append(s)
      for j in graph[s]: # foreach neighbor w of v do
        # w found for the first time?
        if distance[j] < 0:
          queue.append(j)
          distance[j] = distance[s] + 1
        # shortest path to w via v?
        if distance[j] == distance[s] + 1:
          sigma[j] = sigma[j] + sigma[s]
          P[j].append(s)
    ###END###
    if verbose:
      print 'Sigma: ', sigma
      print 'Visited: ', visited
      print 'i: ', i, ' Parents: ', P

    delta = {}
    for j in graph.keys():
      delta[j] = 0
    # visited returns vertices in order of non-increasing distance from s
    while visited != []:
      w = visited.pop()
      for v in P[w]:
        delta[v] += (float(sigma[v]) / sigma[w]) * (1 + delta[w])
      if w != i:
        cb[w] += delta[w]
  return cb

def topk(cb, k):
  return sorted(cb.items(), key = lambda x : x[1], reverse=True)[:k]

def lastk(cb, k):
  return sorted(cb.items(), key = lambda x : x[1], reverse=True)[-k:]

def cb_max(cb):
  node = -1
  maxvalue = -1
  for v in cb.keys():
    if cb[v] > maxvalue:
      maxvalue = cb[v]
      node = v
  return (node, maxvalue)
