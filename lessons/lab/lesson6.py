#!/usr/bin/python

# counts the number of neighbors of i adopting alternative "1"
def OneNeighbors(i, profile, graph):
  oneNeigh=0
  for j in graph[i]:
    if profile[j] == 1:
      oneNeigh += 1
  return oneNeigh

# check if there are nodes for which it is a best response to adopt alternative "1"
def isEquilibrium(profile,graph,a,b,seed, withBR):
  nodes=set(graph.keys())
  equilibrium = 1
  for i in nodes.difference(seed): # It will never happen that a node with alternative "1" prefers to adopt alternative "0"
    if float(OneNeighbors(i,profile,graph))/len(graph[i]) >= float(b)/(a + b): # In case of ties we assume that agent adopt a new technology (no switching cost)
      equilibrium = 0
      withBR += [i]
  return equilibrium

# simulate the process until an equilibrium is reached
def simulate_game(graph,a,b,seed):
  withBR = []
  profile = dict()
  for i in graph.keys():
    profile[i] = 0
  for i in seed:
    profile[i] = 1
  while not isEquilibrium(profile,graph,a,b,seed, withBR):
    for i in withBR:
      profile[i] = 1
      seed.add(i)
    withBR = []
  return profile