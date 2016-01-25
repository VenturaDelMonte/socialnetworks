#!/usr/bin/python

import random

def randomGraph(n, p):
  graph = dict() #we represent a graph as a list of adjacency lists
  for i in range(n):
    graph[i] = [] #there are n nodes (n adjacency lists)
  for i in range(n):
    for j in range(i+1,n): #we start from i+1 because graph is undirected
      r = random.random()
      if r <= p : #the probability that a random value in [0,1] is at most p is exactly p
        graph[i].append(j)
        graph[j].append(i) #graph is undirected, thus if j is a neighbor of i, then i is a neighbor of j
  return graph

def countTriangles(graph):
  triangles = 0
  for i in graph.keys():
    for j in graph[i]:
      for k in graph[j]:
        if k in graph[i]:
          triangles += 1
  return int(triangles/6)

def diameter(graph):
  n = len(graph)
  diameter = -1
  for i in graph.keys(): #check the max distance for every node i
    visited = [] #keep a list of visited nodes to check if the graph is connected
    max_distance = 0 #keep the max distance from i
    ###BFS###
    queue = [i]
    distance = dict()
    for j in graph.keys():
      distance[j] = -1
    distance[i] = 0
    while queue != []:
      s = queue.pop(0)
      visited.append(s)
      for j in graph[s]:
        if not type(j) is str and distance[j] < 0:
          queue.append(j)
          distance[j] = distance[s] + 1
          if distance[j] > max_distance:
            max_distance = distance[j]
    ###END###
    if len(visited) < n: #graph is not connected
      break
    elif max_distance > diameter:
      diameter = max_distance
  return diameter

def averageClustering(graph):
  n = len(graph)
  total=0
  for i in graph.keys():
    triangles = 0
    neigh_pairs = (len(graph[i])*(len(graph[i])-1))/2 #number of pairs of neighbors of node i
    for j in graph[i]:
      for k in graph[i]:
        if not type(j) is str and k in graph[j]:
          triangles += 1 #number of pairs of neighbors of node i that are adjacent
    if neigh_pairs > 0:
      total += float(triangles)/(2*neigh_pairs) # triangles/neighbors is the individual clustering of node i
  return float(total)/n #the average clustering is the average of individual clusterings